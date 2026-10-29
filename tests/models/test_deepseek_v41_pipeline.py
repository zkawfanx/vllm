# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_name", ["deepseek_v4", "deepseek_v4_1"])
@pytest.mark.parametrize("num_redundant_experts", [0, 2, 6])
def test_expert_mapping_loads_every_physical_slot(model_name, num_redundant_experts):
    """Replica slots must load the checkpoint weights of their logical expert."""
    model_module = import_module(f"vllm.models.{model_name}.nvidia.model")
    model = torch.nn.Module()
    model.config = SimpleNamespace(n_routed_experts=4)
    model.start_layer, model.end_layer = 1, 2
    model.layers = [
        None,
        SimpleNamespace(
            ffn=SimpleNamespace(
                use_mega_moe=False, n_redundant_experts=num_redundant_experts
            )
        ),
    ]

    mapping = model_module.DeepseekV4Model.get_expert_mapping(model)

    assert mapping == [
        (
            "experts.routed_experts.w2_"
            if shard == "w2"
            else "experts.routed_experts.w13_",
            f"experts.{physical_id % 4}.{shard}.",
            physical_id,
            shard,
        )
        for physical_id in range(4 + num_redundant_experts)
        for shard in ("w1", "w2", "w3")
    ]


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_name", ["deepseek_v4", "deepseek_v4_1"])
@pytest.mark.parametrize("num_redundant_experts", [0, 2, 6])
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_expert_loader_initializes_all_local_replicas(
    monkeypatch, model_name, num_redundant_experts, ep_rank
):
    """Every local physical copy receives all three checkpoint projections."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    model_module = import_module(f"vllm.models.{model_name}.nvidia.model")
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        model_module,
        "is_pp_missing_parameter",
        lambda name, _: name.startswith("remote."),
    )
    num_local_experts = (4 + num_redundant_experts) // 2
    start = ep_rank * num_local_experts
    experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(experts)
    experts.quant_config = None
    experts.quant_method = SimpleNamespace()
    experts.moe_config = SimpleNamespace(
        tp_rank=0, is_act_and_mul=True, moe_parallel_config=SimpleNamespace(tp_size=1)
    )
    experts.expert_map_manager = SimpleNamespace(
        map_global_to_local=lambda expert: (
            expert - start if start <= expert < start + num_local_experts else -1
        )
    )
    params = {
        f"ffn.experts.routed_experts.{name}": torch.nn.Parameter(
            torch.full((num_local_experts, rows, 2), float("nan")), requires_grad=False
        )
        for name, rows in (("w13_weight", 4), ("w2_weight", 2))
    }
    for param in params.values():
        param.weight_loader = experts.weight_loader
    model = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=1, n_routed_experts=4),
        quant_config=None,
        use_sequence_parallel=False,
        named_parameters=lambda: params.items(),
    )
    model.get_expert_mapping = lambda: RoutedExperts.make_expert_params_mapping(
        model, "w1", "w2", "w3", 4, num_redundant_experts
    )
    weights = {
        f"ffn.experts.{expert}.{shard}.weight": torch.full(
            (2, 2), float(10 * expert + i)
        )
        for expert in range(4)
        for i, shard in enumerate(("w1", "w2", "w3"), start=1)
    }
    loaded = model_module.DeepseekV4Model.load_weights(model, weights.items())
    assert loaded == set(params)
    for local_id in range(num_local_experts):
        logical_id = (start + local_id) % 4
        for shard, param_name, rows in (
            ("w1", "w13_weight", slice(0, 2)),
            ("w3", "w13_weight", slice(2, 4)),
            ("w2", "w2_weight", slice(None)),
        ):
            torch.testing.assert_close(
                params[f"ffn.experts.routed_experts.{param_name}"][local_id, rows],
                weights[f"ffn.experts.{logical_id}.{shard}.weight"],
            )
    assert not model_module.DeepseekV4Model.load_weights(
        model, ((f"remote.{name}", weight) for name, weight in weights.items())
    )


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("num_tokens", [0, 1, 5, 8])
def test_sequence_parallel_token_ids_stay_aligned_with_hidden_rows(
    monkeypatch, num_tokens
):
    """Target and draft token-ID vectors must shard like hidden-state matrices."""
    from vllm.models.common.ops import sequence_parallel as sp

    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: 4)
    token_ids = torch.arange(num_tokens, dtype=torch.int64) + 101
    hidden_states = torch.stack((token_ids, token_ids + 100), dim=-1)
    shards = []
    for rank in range(4):
        monkeypatch.setattr(
            sp, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        ids = sp.sp_shard(token_ids)
        assert ids.shape == ((num_tokens + 3) // 4,)
        torch.testing.assert_close(ids, sp.sp_shard(hidden_states)[:, 0])
        shards.append(ids)
    joined = torch.cat(shards)
    torch.testing.assert_close(joined[:num_tokens], token_ids)
    assert not joined[num_tokens:].any()


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_stage_entry_injects_engram_before_mhc_pre(monkeypatch):
    """Pipeline entry must apply Engram before computing attention mixing."""
    model_module = import_module("vllm.models.deepseek_v4_1.nvidia.model")

    class StopAfterMhcPre(Exception):
        pass

    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    hashes = torch.arange(6).reshape(2, 3)
    mask = torch.tensor([True, False])
    pre_mix = torch.randn(2, 4)
    engram_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    mhc_pre_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class Engram:
        layer_hash_index = 1

        def __call__(self, residual, layer_hashes, engram_mask):
            engram_calls.append((residual, layer_hashes, engram_mask))
            return residual + 10

    def mhc_pre(residual, *args, pre_mix=None, **kwargs):
        mhc_pre_calls.append((residual, pre_mix))
        raise StopAfterMhcPre

    layer = SimpleNamespace(
        engram=Engram(),
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
    )
    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", mhc_pre)

    with pytest.raises(StopAfterMhcPre):
        model_module.DeepseekV4DecoderLayer.forward(
            layer,
            hidden,
            torch.arange(2),
            None,
            pre_mix=pre_mix,
            engram_hashes=hashes,
            engram_mask=mask,
        )

    assert len(engram_calls) == len(mhc_pre_calls) == 1
    engram_residual, layer_hashes, engram_mask = engram_calls[0]
    assert engram_residual is hidden
    assert engram_mask is mask
    torch.testing.assert_close(layer_hashes, hashes[:, 1])
    mhc_residual, mhc_pre_mix = mhc_pre_calls[0]
    torch.testing.assert_close(mhc_residual, hidden + 10)
    assert mhc_pre_mix is pre_mix


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("owned_layer", [1, 14, None])
def test_pipeline_loads_only_local_engram_projection(monkeypatch, owned_layer):
    """PP placeholders skip remote Engrams before their TP weight loader runs."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.models import utils
    from vllm.model_executor.models.utils import PPMissingLayer

    monkeypatch.setattr(utils, "_model_to_pp_missing_layer_names", {})
    model_module = import_module("vllm.models.deepseek_v4_1.nvidia.model")
    for module in (model_module, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
    model = model_module.DeepseekV4Model.__new__(model_module.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_attention_heads=2)
    model.quant_config = None
    model.use_sequence_parallel = False
    model.get_expert_mapping = lambda: []
    model.layers = torch.nn.ModuleList([PPMissingLayer() for _ in range(15)])
    if owned_layer is not None:
        layer = torch.nn.Module()
        layer.engram = torch.nn.Module()
        layer.engram.wkv = ColumnParallelLinear(
            4,
            8,
            bias=False,
            gather_output=True,
            return_bias=False,
            tp_rank=1,
            tp_size=2,
        )
        model.layers[owned_layer] = layer
    weights = {
        f"layers.{index}.engram.wkv.weight": torch.arange(
            32, dtype=torch.float32
        ).reshape(8, 4)
        + index
        for index in (1, 14)
    }
    loaded = model.load_weights(weights.items())
    if owned_layer is None:
        assert not loaded
    else:
        name = f"layers.{owned_layer}.engram.wkv.weight"
        assert loaded == {name}
        torch.testing.assert_close(layer.engram.wkv.weight, weights[name][4:])
