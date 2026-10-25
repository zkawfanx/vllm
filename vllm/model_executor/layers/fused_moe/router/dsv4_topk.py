# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_TOPK = 6

# Adapted from:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/moe_fused_gate.py


def can_use_dsv4_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    indices_dtype: torch.dtype,
) -> bool:
    return (
        current_platform.is_cuda()
        and gating_output.dtype == torch.float32
        and gating_output.ndim == 2
        and gating_output.shape[1] in (256, 384)
        and gating_output.is_contiguous()
        and correction_bias is not None
        and correction_bias.dtype == torch.float32
        and correction_bias.shape == (gating_output.shape[1],)
        and correction_bias.is_contiguous()
        and topk == _TOPK
        and renormalize
        and indices_dtype in (torch.int32, torch.int64)
    )


if current_platform.is_cuda():

    @triton.jit
    def _dsv4_topk_kernel(
        gating_output_ptr,
        correction_bias_ptr,
        bias_vl_ptr,
        input_tokens_ptr,
        is_padding_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        routed_scaling_factor,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        HAS_VISION: tl.constexpr,
        HAS_CORRECTION_BIAS: tl.constexpr,
        IMAGE_SENTINEL_LO: tl.constexpr,
        TOPK: tl.constexpr,
        TOPK_BLOCK: tl.constexpr,
        RENORMALIZE: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        row = tl.program_id(0)
        expert_offsets = tl.arange(0, BLOCK_N)
        expert_mask = expert_offsets < NUM_EXPERTS
        bias = tl.full([BLOCK_N], 0.0, tl.float32)
        if HAS_CORRECTION_BIAS:
            bias = tl.load(
                correction_bias_ptr + expert_offsets, mask=expert_mask, other=0.0
            ).to(tl.float32)
        if HAS_VISION:
            token = tl.load(input_tokens_ptr + row)
            is_image = (token >= IMAGE_SENTINEL_LO) & (token < IMAGE_SENTINEL_LO + 5)
            vision_bias = tl.load(
                bias_vl_ptr + expert_offsets, mask=expert_mask, other=0.0
            ).to(tl.float32)
            bias = tl.where(is_image, vision_bias, bias)

        if launch_pdl:
            tl.extra.cuda.gdc_wait()

        logits = tl.load(
            gating_output_ptr + row * NUM_EXPERTS + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_VISION:
            # Avoid cancellation and exp underflow before the square root.
            weights = tl.where(
                logits < -20.0,
                tl.exp(logits * 0.5),
                tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits)))),
            )
        else:
            weights = tl.sqrt(
                tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits)))
            )
        current = tl.where(expert_mask, weights + bias, -float("inf"))
        current = tl.where(current == current, current, -1e30)

        topk_offsets = tl.arange(0, TOPK_BLOCK)
        selected_weights = tl.zeros([TOPK_BLOCK], dtype=tl.float32)
        selected_ids = tl.zeros([TOPK_BLOCK], dtype=tl.int32)
        for slot in tl.static_range(TOPK):
            max_value = tl.max(current, axis=0)
            candidate = tl.where(current == max_value, expert_offsets, NUM_EXPERTS)
            expert_id = tl.min(candidate, axis=0).to(tl.int32)
            selected_weight = tl.sum(
                tl.where(expert_offsets == expert_id, weights, 0.0), axis=0
            )
            is_slot = topk_offsets == slot
            selected_weights = tl.where(is_slot, selected_weight, selected_weights)
            selected_ids = tl.where(is_slot, expert_id, selected_ids)
            current = tl.where(expert_offsets == expert_id, -float("inf"), current)

        if RENORMALIZE:
            weight_sum = tl.sum(selected_weights, axis=0)
            selected_weights *= routed_scaling_factor / tl.where(
                weight_sum > 0.0, weight_sum, 1.0
            )
        else:
            selected_weights *= routed_scaling_factor
        output_mask = topk_offsets < TOPK
        output_offsets = row * TOPK + topk_offsets

        if HAS_PADDING:
            is_padding = tl.load(is_padding_ptr + row)
            selected_weights = tl.where(is_padding, 0.0, selected_weights)
            selected_ids = tl.where(is_padding, -1, selected_ids)

        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()

        tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
        tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)


def dsv4_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    indices_dtype: torch.dtype,
    routed_scaling_factor: float,
    is_padding: torch.Tensor | None = None,
    *,
    topk: int = _TOPK,
    renormalize: bool = True,
    bias_vl: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = gating_output.shape
    has_vision = bias_vl is not None and image_sentinel_lo > 0
    if has_vision:
        assert input_tokens is not None
        assert input_tokens.shape == (num_tokens,)
        assert input_tokens.dtype in (torch.int32, torch.int64)
        assert input_tokens.device == gating_output.device
        assert input_tokens.is_contiguous() and gating_output.is_contiguous()
        for bias in (correction_bias, bias_vl):
            if bias is not None:
                assert bias.shape == (num_experts,)
                assert bias.dtype == torch.float32
                assert bias.device == gating_output.device
                assert bias.is_contiguous()
    if is_padding is not None:
        assert is_padding.dtype == torch.bool
        assert is_padding.shape == (num_tokens,)
        assert is_padding.device == gating_output.device
        assert is_padding.is_contiguous()
        assert indices_dtype in (torch.int32, torch.int64), (
            "Padding requires a signed indices dtype for the -1 sentinel."
        )
    shape = (num_tokens, topk)
    topk_weights = gating_output.new_empty(shape, dtype=torch.float32)
    topk_ids = gating_output.new_empty(shape, dtype=indices_dtype)
    if num_tokens > 0:
        _dsv4_topk_kernel[(num_tokens,)](
            gating_output,
            correction_bias,
            bias_vl,
            input_tokens,
            is_padding,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
            NUM_EXPERTS=num_experts,
            BLOCK_N=triton.next_power_of_2(num_experts),
            HAS_PADDING=is_padding is not None,
            HAS_VISION=has_vision,
            HAS_CORRECTION_BIAS=correction_bias is not None,
            IMAGE_SENTINEL_LO=image_sentinel_lo,
            TOPK=topk,
            TOPK_BLOCK=triton.next_power_of_2(topk),
            RENORMALIZE=renormalize,
            num_warps=1,
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
    return topk_weights, topk_ids
