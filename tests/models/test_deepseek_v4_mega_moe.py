# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MegaMoEExperts,
    DeepseekV4MoE,
    make_deepseek_v4_expert_params_mapping,
)
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DeepSeek V4 MegaMoE requires CUDA",
)


def test_deepseek_v4_mega_moe_expert_mapping():
    mapping = make_deepseek_v4_expert_params_mapping(2)

    assert mapping == [
        ("experts.w13_", "experts.0.w1.", 0, "w1"),
        ("experts.w2_", "experts.0.w2.", 0, "w2"),
        ("experts.w13_", "experts.0.w3.", 0, "w3"),
        ("experts.w13_", "experts.1.w1.", 1, "w1"),
        ("experts.w2_", "experts.1.w2.", 1, "w2"),
        ("experts.w13_", "experts.1.w3.", 1, "w3"),
    ]


def test_deepseek_v4_mega_moe_ue8m0_uint8_to_float():
    raw = torch.tensor([0, 126, 127, 128], dtype=torch.uint8)

    decoded = DeepseekV4MegaMoEExperts._ue8m0_uint8_to_float(raw)

    assert torch.equal(decoded.view(torch.int32), raw.to(torch.int32) << 23)
    assert decoded[0].item() == 0.0
    assert decoded[1].item() == 0.5
    assert decoded[2].item() == 1.0
    assert decoded[3].item() == 2.0


def test_deepseek_v4_mega_moe_weight_loader_uses_ep_expert_ownership():
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
    )

    nonlocal_weight = torch.ones(128, 64, dtype=torch.uint8)
    assert (
        experts.weight_loader(
            experts.w13_weight,
            nonlocal_weight,
            "experts.w13_weight",
            shard_id="w1",
            expert_id=1,
            return_success=True,
        )
        is False
    )

    w1 = torch.full((128, 64), 3, dtype=torch.uint8)
    w3 = torch.full((128, 64), 7, dtype=torch.uint8)
    w2 = torch.full((128, 64), 11, dtype=torch.uint8)

    assert experts.weight_loader(
        experts.w13_weight,
        w1,
        "experts.w13_weight",
        shard_id="w1",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight,
        w3,
        "experts.w13_weight",
        shard_id="w3",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight,
        w2,
        "experts.w2_weight",
        shard_id="w2",
        expert_id=2,
        return_success=True,
    )

    assert torch.equal(experts.w13_weight[0, :128], w1)
    assert torch.equal(experts.w13_weight[0, 128:], w3)
    assert torch.equal(experts.w2_weight[0], w2)
    assert torch.count_nonzero(experts.w13_weight[1]) == 0


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,block_size,fused",
    [
        pytest.param(128, 512, 128, True, id="aligned-block128"),
        pytest.param(5120, 2304, 32, False, id="deepseek-v41-flash"),
    ],
)
def test_deepseek_v4_mega_moe_finalizes_native_shared_expert_weights(
    monkeypatch, hidden_size, intermediate_size, block_size, fused
):
    """V4.1-Flash's routed padding must preserve the serial shared-expert fallback."""

    class FakeDeepGemm:
        transformed_dims: list[tuple[int, int]] = []
        scale_inputs: list[tuple[int, ...]] = []

        @staticmethod
        def get_symm_buffer_for_mega_moe(*args, num_shared_experts=0, **kwargs):
            return None

        @staticmethod
        def get_block_m_for_mega_moe(*args, **kwargs):
            return 128

        @staticmethod
        def fp8_fp4_mega_moe(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            shared_l1_weights=None,
            shared_l2_weights=None,
            **kwargs,
        ):
            return None

        @classmethod
        def transform_sf_into_required_layout(cls, sf, mn, k, *args, **kwargs):
            cls.scale_inputs.append(tuple(sf.shape))
            return torch.empty((sf.shape[0], mn, k // 128), dtype=torch.int32)

        @classmethod
        def transform_weights_for_mega_moe(cls, l1_weights, l2_weights):
            cls.transformed_dims.append((l1_weights[0].dim(), l2_weights[0].dim()))
            if l1_weights[0].dim() == 2:
                return (l1_weights[0].clone(), l1_weights[1]), l2_weights
            return l1_weights, l2_weights

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=2,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_shared_experts=1,
    )
    experts._check_runtime_supported = lambda: None

    def fp8_parameter(*shape):
        return torch.nn.Parameter(
            torch.empty(*shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )

    def scale_parameter(*shape, dtype=torch.int32):
        return torch.nn.Parameter(torch.ones(*shape, dtype=dtype), requires_grad=False)

    shared_experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=fp8_parameter(2 * intermediate_size, hidden_size),
            weight_block_size=(block_size, block_size),
            weight_scale_inv=scale_parameter(
                2 * intermediate_size // block_size,
                hidden_size // block_size,
                dtype=torch.float8_e8m0fnu,
            ),
        ),
        down_proj=SimpleNamespace(
            weight=fp8_parameter(hidden_size, intermediate_size),
            weight_block_size=(block_size, block_size),
            weight_scale_inv=scale_parameter(
                hidden_size // block_size,
                intermediate_size // block_size,
                dtype=torch.float8_e8m0fnu,
            ),
        ),
    )
    monkeypatch.setattr("vllm.utils.deep_gemm._import_deep_gemm", lambda: FakeDeepGemm)

    original_gate_up_ptr = shared_experts.gate_up_proj.weight.data_ptr()
    original_down_ptr = shared_experts.down_proj.weight.data_ptr()
    experts.finalize_weights(shared_experts)

    assert FakeDeepGemm.scale_inputs[-2:] == [
        (1, 2 * intermediate_size, hidden_size // 32),
        (1, hidden_size, intermediate_size // 32),
    ]
    assert experts.has_fused_shared_experts is fused
    if not fused:
        assert experts.intermediate_size == 2560
        assert FakeDeepGemm.transformed_dims == [(3, 3)]
        assert experts.num_shared_experts == 0
        assert shared_experts.gate_up_proj.weight.data_ptr() == original_gate_up_ptr
        assert shared_experts.down_proj.weight.data_ptr() == original_down_ptr
        assert shared_experts.gate_up_proj.weight.shape == (
            2 * intermediate_size,
            hidden_size,
        )
        assert shared_experts.down_proj.weight.shape == (hidden_size, intermediate_size)
        return

    assert FakeDeepGemm.transformed_dims == [(3, 3), (2, 2)]
    assert shared_experts.gate_up_proj.weight.data_ptr() != original_gate_up_ptr
    assert (
        experts._transformed_shared_l1_weights[0].data_ptr()
        == shared_experts.gate_up_proj.weight.data_ptr()
    )
    assert (
        experts._transformed_shared_l2_weights[0].data_ptr()
        == shared_experts.down_proj.weight.data_ptr()
    )


@pytest.mark.parametrize("fused", [False, True])
def test_deepseek_v4_mega_moe_does_not_double_add_fused_shared_expert(
    monkeypatch, fused
):
    class FakeGate(torch.nn.Module):
        tid2eid = None
        e_score_correction_bias = None

        def forward(self, hidden_states):
            return torch.empty(hidden_states.shape[0], 2), None

    class FakeExperts(torch.nn.Module):
        has_fused_shared_experts = fused

        def forward(self, hidden_states, *args, **kwargs):
            return torch.ones_like(hidden_states)

    class FakeSharedExperts(torch.nn.Module):
        calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.full_like(hidden_states, 2)

    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.use_mega_moe = True
    moe.gate = FakeGate()
    moe.experts = FakeExperts()
    moe.shared_experts = FakeSharedExperts()
    moe.scoring_func = "sqrtsoftplus"
    moe.n_activated_experts = 1
    moe.renormalize = True
    moe.hash_indices_dtype = torch.int64
    moe.routed_scaling_factor = 1.0
    moe.swiglu_limit = 10.0
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.model.fused_topk_bias",
        lambda **kwargs: (
            torch.ones(kwargs["hidden_states"].shape[0], 1),
            torch.zeros(kwargs["hidden_states"].shape[0], 1, dtype=torch.int64),
        ),
    )

    output = moe(torch.zeros(2, 128))

    expected = 1 if fused else 3
    assert torch.all(output == expected)
    assert moe.shared_experts.calls == (0 if fused else 1)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
@pytest.mark.skipif(
    not has_deep_gemm(),
    reason="the reference per_token_cast_to_fp8 comes from deep_gemm, which "
    "production itself treats as optional (is_deep_gemm_supported)",
)
def test_deepseek_v4_mega_moe_fused_input_staging_is_bitwise_exact():
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 17.0
    ).to(torch.bfloat16)
    hidden_states[0, :32] = 0
    hidden_states[1, 32:64] = 1.0e-6
    hidden_states[2, 64:96] = -1.0e-6

    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_weights = topk_weights.clone()

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 MegaMoE fused input staging requires CUDA.",
)
@pytest.mark.skipif(
    not has_deep_gemm(),
    reason="the reference per_token_cast_to_fp8 comes from deep_gemm, which "
    "production itself treats as optional (is_deep_gemm_supported)",
)
def test_deepseek_v4_mega_moe_fused_input_staging_masks_padding():
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    num_tokens = 7
    hidden_size = 256
    top_k = 8

    generator = torch.Generator(device=device)
    generator.manual_seed(1)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = torch.randint(
        0,
        256,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.randn(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    is_padding = torch.tensor(
        [False, True, False, False, True, False, True],
        device=device,
    )

    ref_x, ref_x_sf = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    ref_topk_idx = topk_ids.to(torch.int64)
    ref_topk_idx[is_padding] = -1
    ref_topk_weights = topk_weights.clone()
    ref_topk_weights[is_padding] = 0.0

    fused_x = torch.empty_like(ref_x)
    fused_x_sf = torch.empty_like(ref_x_sf)
    fused_topk_idx = torch.empty_like(ref_topk_idx)
    fused_topk_weights = torch.empty_like(ref_topk_weights)

    prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        fused_x,
        fused_x_sf,
        fused_topk_idx,
        fused_topk_weights,
        is_padding=is_padding,
    )
    torch.accelerator.synchronize()

    assert torch.equal(fused_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(fused_x_sf, ref_x_sf)
    assert torch.equal(fused_topk_idx, ref_topk_idx)
    assert torch.equal(
        fused_topk_weights.view(torch.uint8),
        ref_topk_weights.view(torch.uint8),
    )
