# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 sparse-MLA decode for DeepSeek V4.1 on Ampere."""

import torch

from vllm.models.deepseek_v4_1.ampere.sparse_attn import ampere_sparse_attn
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_sm80 import (
    _decode_fp8_f32,
    native_fp8_cast_supported,
)

_BYTES_PER_TOKEN = 584
_FP8_DIM = 448
_BF16_DIM = 64
_SCALE_DIM = 8
_QUANT_BLOCK = 64
_TOKEN_DATA_SIZE = _FP8_DIM + _BF16_DIM * 2
_HEAD_DIM = 512


@triton.jit(do_not_specialize=["num_rows"])
def _gather_dequant_slots_kernel(
    cache_ptr,
    indices_ptr,
    out_ptr,
    num_rows,
    block_stride,
    WIDTH: tl.constexpr,
    TOTAL: tl.constexpr,
    COL_OFFSET: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    TOKEN_DATA_SIZE: tl.constexpr,
    FP8_DIM: tl.constexpr,
    BF16_DIM: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_QUANT_BLOCKS: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(indices_ptr + row).to(tl.int64)
    token = row // WIDTH
    column = row % WIDTH
    out_row_ptr = out_ptr + (token * TOTAL + COL_OFFSET + column) * HEAD_DIM
    cols = tl.arange(0, HEAD_DIM)

    if slot < 0:
        tl.store(out_row_ptr + cols, tl.zeros((HEAD_DIM,), dtype=tl.bfloat16))
    else:
        block_idx = slot // CACHE_BLOCK_SIZE
        pos_in_block = slot % CACHE_BLOCK_SIZE
        cache_block_ptr = cache_ptr + block_idx * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * TOKEN_DATA_SIZE
        token_scale_ptr = (
            cache_block_ptr
            + CACHE_BLOCK_SIZE * TOKEN_DATA_SIZE
            + pos_in_block * SCALE_DIM
        )

        for qblock_idx in tl.static_range(N_QUANT_BLOCKS):
            qblock_start = qblock_idx * QUANT_BLOCK
            if qblock_start < FP8_DIM:
                offsets = qblock_start + tl.arange(0, QUANT_BLOCK)
                mask = offsets < FP8_DIM
                raw = tl.load(token_data_ptr + offsets, mask=mask, other=0)
                values = _decode_fp8_f32(raw, False)
                encoded = tl.load(token_scale_ptr + qblock_idx)
                scale = tl.exp2(encoded.to(tl.float32) - 127.0)
                tl.store(
                    out_row_ptr + offsets,
                    (values * scale).to(tl.bfloat16),
                    mask=mask,
                )

        bf16_ptr = (token_data_ptr + FP8_DIM).to(tl.pointer_type(tl.bfloat16))
        tail = tl.arange(0, BF16_DIM)
        tl.store(out_row_ptr + FP8_DIM + tail, tl.load(bf16_ptr + tail))


def _gather_dequant(
    cache: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    total: int,
    col_offset: int,
) -> None:
    if cache.shape[-1] != _BYTES_PER_TOKEN:
        raise NotImplementedError(
            "DeepSeek V4.1 Ampere decode expects the 584-byte KV record; "
            f"got {cache.shape[-1]} bytes."
        )
    num_rows = indices.numel()
    if num_rows == 0:
        return
    _gather_dequant_slots_kernel[(num_rows,)](
        cache,
        indices,
        out,
        num_rows,
        cache.stride(0),
        WIDTH=indices.shape[1],
        TOTAL=total,
        COL_OFFSET=col_offset,
        CACHE_BLOCK_SIZE=cache.shape[1],
        TOKEN_DATA_SIZE=_TOKEN_DATA_SIZE,
        FP8_DIM=_FP8_DIM,
        BF16_DIM=_BF16_DIM,
        SCALE_DIM=_SCALE_DIM,
        QUANT_BLOCK=_QUANT_BLOCK,
        HEAD_DIM=_HEAD_DIM,
        N_QUANT_BLOCKS=_SCALE_DIM,
        NATIVE_FP8=native_fp8_cast_supported(),
        num_warps=4,
    )


def ampere_sparse_decode(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    compressed_cache: torch.Tensor | None,
    topk_indices: torch.Tensor | None,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Gather both paged caches and run one BF16 sparse attention."""
    num_tokens = q.shape[0]
    parts = [(swa_cache, swa_indices.reshape(num_tokens, -1).contiguous())]
    if compressed_cache is not None and topk_indices is not None:
        parts.append(
            (compressed_cache, topk_indices.reshape(num_tokens, -1).contiguous())
        )

    total = sum(indices.shape[1] for _, indices in parts)
    kv = torch.empty(
        (num_tokens * total, _HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )
    flat_indices = torch.arange(
        num_tokens * total, device=q.device, dtype=torch.int32
    ).view(num_tokens, total)

    col_offset = 0
    for cache, indices in parts:
        width = indices.shape[1]
        _gather_dequant(cache, indices, kv, total, col_offset)
        window = flat_indices[:, col_offset : col_offset + width]
        window.copy_(torch.where(indices >= 0, window, -1))
        col_offset += width

    return ampere_sparse_attn(
        q,
        kv.unsqueeze(1),
        flat_indices.unsqueeze(1),
        sm_scale,
        attn_sink,
        out=out,
    )
