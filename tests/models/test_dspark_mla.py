# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    assert set(K3DSparkForCausalLM.checkpoint_skip_substrs) == {
        "confidence_head",
        "embed_tokens",
        "lm_head",
    }


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        # K3DSparkModel.__init__ sizes its context-KV buffer from the scheduler
        # budget (dspark_mla.py:150-152), so the stand-in config has to carry
        # it. The value is arbitrary here -- nothing in this test reaches the
        # buffer -- but it must exist.
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1


@pytest.mark.cpu_test
@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_v41_context_wkv_loads_weights_and_scales_into_both_projections(
    monkeypatch, scale_name
):
    """Replicated context shards retain layer order without consuming query WKV."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.models.deepseek_v4_1.nvidia import dspark

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **k: []
    )
    projections = [
        MergedColumnParallelLinear(
            5, sizes, bias=False, disable_tp=True, params_dtype=torch.float32
        )
        for sizes in ([2, 4], [2, 4], [2, 4], [4, 4, 4])
    ]
    params = {}
    for i, projection in enumerate(projections):
        prefix = (
            f"model.layers.{i}.attn.fused_wqa_wkv"
            if i < 3
            else "model.context_wkv_proj"
        )
        projection.weight.data.fill_(-1)
        params[f"{prefix}.weight"] = projection.weight
        projection.weight_block_size = [2, 1]
        scale = parameter.BlockQuantScaleParameter(
            data=torch.full((sum(projection.output_sizes) // 2, 5), -1.0),
            input_dim=1,
            output_dim=0,
            weight_loader=projection.weight_loader_v2,
        )
        projection.register_parameter(scale_name, scale)
        projection.update_param_tp_status()
        params[f"{prefix}.{scale_name}"] = scale
    draft = SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=1, num_attention_heads=4),
        linear_scale_name=scale_name,
        pad_shared_expert=False,
        model=SimpleNamespace(
            layers=[SimpleNamespace(ffn=SimpleNamespace(use_mega_moe=False))] * 3,
            confidence_head=None,
        ),
        named_parameters=lambda: params.items(),
        process_weights_after_loading=lambda: None,
    )
    draft._remap_dspark_name = lambda name: (
        dspark.DSparkDeepseekV4ForCausalLM._remap_dspark_name(draft, name)
    )
    weights = [torch.arange(20).view(4, 5).float() + i * 20 for i in range(3)]
    checkpoint = [
        (f"mtp.{i}.attn.wkv.{suffix}", value)
        for i in (2, 0, 1)
        for suffix, value in (
            ("weight", weights[i]),
            ("scale", torch.full((2, 5), i + 0.5)),
        )
    ]
    loaded = dspark.DSparkDeepseekV4ForCausalLM.load_weights(draft, iter(checkpoint))

    assert loaded == set(params)
    torch.testing.assert_close(projections[3].weight, torch.cat(weights))
    torch.testing.assert_close(
        params[f"model.context_wkv_proj.{scale_name}"],
        torch.tensor([0.5, 1.5, 2.5]).repeat_interleave(2)[:, None].expand(6, 5),
    )
    for i, projection in enumerate(projections[:3]):
        torch.testing.assert_close(projection.weight[2:], weights[i])
        assert torch.all(projection.weight[:2] == -1)
        torch.testing.assert_close(
            params[f"model.layers.{i}.attn.fused_wqa_wkv.{scale_name}"],
            torch.tensor([-1.0, i + 0.5, i + 0.5])[:, None].expand(3, 5),
        )
    assert all(not hasattr(weight, "shard_id") for _, weight in checkpoint)


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_tokens", [0, 2])
@pytest.mark.parametrize("profiling", [False, True])
def test_v41_context_kv_stacked_projection_preserves_layer_outputs(
    monkeypatch, num_tokens, profiling
):
    """One KV-only GEMM must match separate projections, including profiling."""
    from vllm.models.deepseek_v4_1.nvidia import dspark

    x = torch.arange(num_tokens * 5, dtype=torch.float32).reshape(num_tokens, 5)
    weights = [torch.arange(20).reshape(4, 5).float() + i for i in range(3)]
    calls = []
    inserted = []

    def project(hidden):
        calls.append(hidden)
        return hidden @ torch.cat(weights).T

    layers = [
        SimpleNamespace(attn=SimpleNamespace(kv_norm=nn.Identity())) for _ in weights
    ]
    model = SimpleNamespace(
        layers=layers,
        context_wkv_proj=project,
        num_dspark_layers=3,
        config=SimpleNamespace(head_dim=4),
    )
    positions = torch.arange(num_tokens)
    mappings = None if profiling else [positions, None, positions + 10]
    monkeypatch.setattr(
        dspark,
        "_insert_context_kv",
        lambda attn, kv, pos, slots: inserted.append((attn, kv, pos, slots)),
    )
    dspark.DSparkDeepseekV4Model.precompute_and_store_context_kv(
        model, x, positions, mappings
    )
    assert len(calls) == 1 and calls[0] is x
    assert len(inserted) == (0 if profiling else 2)
    for (attn, kv, pos, slots), i in zip(inserted, (0, 2)):
        assert attn is layers[i].attn
        torch.testing.assert_close(kv, x @ weights[i].T)
        assert pos is positions and slots is mappings[i]


@pytest.mark.cpu_test
@pytest.mark.parametrize("mismatch", [None, "unquantized", "method", "config"])
def test_v41_context_projection_rejects_incompatible_quantization(
    monkeypatch, mismatch
):
    """Prefix exclusions must fail before checkpoint tensors can be misinterpreted."""
    from vllm.models.deepseek_v4_1.nvidia import dspark

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class QuantMethod:
        def __init__(self, config):
            self.quant_config = config

    class ReachedHeads(Exception):
        pass

    quant_config = object()
    context_method = QuantMethod(quant_config)
    query_methods = [QuantMethod(quant_config), QuantMethod(quant_config)]
    if mismatch == "unquantized":
        context_method = object()
        query_methods = [object(), object()]
    elif mismatch == "method":
        query_methods[1] = object()
    elif mismatch == "config":
        query_methods[1] = QuantMethod(object())
    layers = []
    for method in query_methods:
        layer = DummyModule()
        layer.attn = SimpleNamespace(
            fused_wqa_wkv=SimpleNamespace(
                quant_method=method, prefix="model.layers.2.attn.fused_wqa_wkv"
            )
        )
        layers.append(layer)
    layer_iter = iter(layers)
    projection = DummyModule()
    projection.quant_method = context_method
    monkeypatch.setattr(
        dspark, "DeepseekV4DecoderLayer", lambda *a, **k: next(layer_iter)
    )

    def make_projection(*args, **kwargs):
        assert kwargs["prefix"] == "model.layers.2.attn.fused_wqa_wkv"
        assert kwargs["disable_tp"]
        return projection

    monkeypatch.setattr(dspark, "MergedColumnParallelLinear", make_projection)
    monkeypatch.setattr(dspark, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark, "VocabParallelEmbedding", DummyModule)
    monkeypatch.setattr(dspark, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark, "_use_sequence_parallel", lambda _: False)
    monkeypatch.setattr(dspark, "get_current_vllm_config", lambda: None)

    def stop_at_heads(*args, **kwargs):
        raise ReachedHeads

    monkeypatch.setattr(dspark, "DSparkMarkovHead", stop_at_heads)
    config = SimpleNamespace(
        hidden_size=4,
        head_dim=2,
        hc_mult=1,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        num_hidden_layers=2,
        dspark_target_layer_ids=[0],
        n_mtp_layers=2,
        vocab_size=8,
        index_topk=2,
        dspark_markov_rank=2,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        quant_config=quant_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2),
    )
    expected = ReachedHeads if mismatch in (None, "unquantized") else ValueError
    with pytest.raises(expected):
        dspark.DSparkDeepseekV4Model(vllm_config=vllm_config)
