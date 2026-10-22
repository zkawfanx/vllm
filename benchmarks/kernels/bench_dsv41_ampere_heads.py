# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare eight real Ampere heads with FlashMLA's 64-head padding."""

import json
import statistics
from functools import partial

import torch

from vllm.models.deepseek_v4_1.ampere.sparse_attn import ampere_sparse_attn
from vllm.triton_utils import triton


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    for tokens in (1, 6, 48, 128):
        for topk in (128, 640):
            q = torch.randn(tokens, 8, 512, dtype=torch.bfloat16, device="cuda")
            kv = torch.randn(2048, 1, 512, dtype=q.dtype, device=q.device)
            indices = torch.randint(
                2048, (tokens, 1, topk), dtype=torch.int32, device=q.device
            )
            indices[:, :, -16:] = -1
            sink = torch.randn(8, device=q.device)
            padded_q = torch.nn.functional.pad(q, (0, 0, 0, 56))
            padded_sink = torch.nn.functional.pad(sink, (0, 56), value=-float("inf"))
            calls = {
                "8": partial(ampere_sparse_attn, q, kv, indices, 512**-0.5, sink),
                "64": partial(
                    ampere_sparse_attn, padded_q, kv, indices, 512**-0.5, padded_sink
                ),
            }
            timings = {heads: [] for heads in calls}
            for repeat in range(5):
                for heads in ("8", "64") if repeat % 2 else ("64", "8"):
                    timings[heads].append(
                        1000 * triton.testing.do_bench_cudagraph(calls[heads], rep=100)
                    )
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "topk": topk,
                        "microseconds": timings,
                        "median_us": {
                            heads: statistics.median(values)
                            for heads, values in timings.items()
                        },
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
