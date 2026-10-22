# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure packed-FP8 DSpark context insertion with and without dummy queries."""

import json
import statistics
from functools import partial
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4_1.nvidia.dspark import _insert_context_kv
from vllm.triton_utils import triton


def legacy_insert(kv, cache, slots, positions, cos_sin, block_size):
    query = torch.zeros(kv.shape[0], 8, 512, dtype=kv.dtype, device=kv.device)
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        query,
        kv,
        cache.view(cache.shape[0], -1),
        slots,
        positions,
        cos_sin,
        64,
        1e-20,
        block_size,
    )


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    for tokens in (8, 128, 1024, 8192):
        block_size = 256
        kv = torch.randn(tokens, 512, dtype=torch.bfloat16, device="cuda")
        cache = torch.zeros(
            (tokens + 255) // 256, block_size, 584, dtype=torch.uint8, device=kv.device
        )
        positions = torch.arange(tokens, device=kv.device)
        slots = positions.clone()
        slots[::7] = -1
        angles = torch.randn(tokens, 32, device=kv.device)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
        attn = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=block_size),
            rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
        )

        calls = {
            "legacy": partial(
                legacy_insert, kv, cache, slots, positions, cos_sin, block_size
            ),
            "kv_only": partial(_insert_context_kv, attn, kv, positions, slots),
        }
        allocations = {}
        for name, call in calls.items():
            torch.accelerator.synchronize()
            before = torch.accelerator.memory_allocated()
            torch.accelerator.reset_peak_memory_stats()
            call()
            torch.accelerator.synchronize()
            allocations[name] = torch.accelerator.max_memory_allocated() - before
        timings = {name: [] for name in calls}
        for repeat in range(5):
            for name in ("legacy", "kv_only") if repeat % 2 else ("kv_only", "legacy"):
                timings[name].append(
                    1000 * triton.testing.do_bench_cudagraph(calls[name], rep=100)
                )
        print(
            json.dumps(
                {
                    "tokens": tokens,
                    "microseconds": timings,
                    "median_us": {
                        name: statistics.median(values)
                        for name, values in timings.items()
                    },
                    "peak_extra_allocated_bytes": allocations,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
