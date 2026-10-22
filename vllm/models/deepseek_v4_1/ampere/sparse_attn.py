# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 sparse-MLA attention for DeepSeek V4.1 on Ampere."""

import torch

from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface


def ampere_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    d_v: int = 512,
) -> torch.Tensor:
    """Run sparse attention over gathered BF16 KV rows."""
    # The generic Triton kernel uses -1 sentinels rather than a separate
    # top-k-length argument. The shared combine kernel initializes reused
    # workspace tails to that sentinel before filling valid entries.
    del topk_length
    block_dpe = q.shape[-1] - d_v
    acc, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v=d_v, block_dpe=block_dpe
    )

    # The Triton kernel returns the usual sink-free normalized value. Recover
    # the sink-aware normalization from its log-sum-exp and running maximum.
    exp_sum = torch.exp(lse - max_logits)
    empty = ~torch.isfinite(exp_sum) | (exp_sum <= 0)
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).view(1, -1).expand_as(max_logits)
        sink_scale = exp_sum / (exp_sum + torch.exp(sink - max_logits))
        acc = acc * sink_scale.unsqueeze(-1).to(acc.dtype)

    acc = torch.where(empty.unsqueeze(-1), torch.zeros_like(acc), acc)
    if out is not None:
        out.copy_(acc)
        return out
    return acc
