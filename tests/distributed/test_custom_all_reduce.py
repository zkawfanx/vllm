# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import ray
import torch
import torch.distributed as dist

import vllm.distributed.device_communicators.custom_all_reduce as car
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

random.seed(42)
test_sizes = [random.randint(1024, 2048 * 1024) for _ in range(8)]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


@ray.remote(num_gpus=1, max_calls=1)
def graph_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    # use integers so result matches NCCL exactly
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(1, 16, (sz,), dtype=dtype, device=device_idx)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for i in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                torch.testing.assert_close(out1, inp1)
                torch.testing.assert_close(out2, inp2)


@ray.remote(num_gpus=1, max_calls=1)
def eager_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1
        sz = 1024
        fa = get_tp_group().device_communicator.ca_comm
        inp = torch.ones(sz, dtype=torch.float32, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))

        inp = torch.ones(sz * 4, dtype=torch.bfloat16, device=device)
        out = inp
        for _ in range(num_communication):
            out = fa.all_reduce(out, registered=False)
        torch.testing.assert_close(out, inp * (tp_size**num_communication))


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [eager_allreduce, graph_allreduce])
def test_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)


def _order_sensitive_inputs(
    numel: int, dtype: torch.dtype, world_size: int, seed: int
) -> torch.Tensor:
    """Per-rank inputs, shape [numel, world_size], whose sum depends on the
    order the ranks are accumulated in.

    Each element holds +B, -B and small terms t in a random rank order, with t
    below the float32 accumulator's resolution at B: adding t to B loses it,
    cancelling +B and -B first keeps it, and the difference survives the
    downcast. Ordinary random fp16/bf16 data cannot detect a changed order,
    because its float32 partial sums are exact.
    """
    log2_big, log2_tiny = {
        torch.float16: (13, -13),
        torch.bfloat16: (20, -8),
        torch.float32: (20, -8),
    }[dtype]
    gen = torch.Generator().manual_seed(seed)
    big = (torch.rand(numel, generator=gen, dtype=torch.float64) + 1) * 2.0**log2_big
    tiny = torch.rand(numel, world_size - 2, generator=gen, dtype=torch.float64) + 1
    sign = torch.randint(0, 2, tiny.shape, generator=gen) * 2 - 1
    tiny *= sign * 2.0**log2_tiny
    terms = torch.cat([big[:, None], -big[:, None], tiny], dim=1)
    order = torch.argsort(torch.rand(numel, world_size, generator=gen), dim=1)
    return torch.gather(terms, 1, order).to(dtype)


@ray.remote(num_gpus=1, max_calls=1)
def one_and_two_stage_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        # Test both custom kernels, without symmetric-memory cap clamping.
        m.setenv("VLLM_BATCH_INVARIANT", "0")
        m.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
        m.setenv("VLLM_MAX_SIZE_MB_CUSTOM_ALL_REDUCE", "8")
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        fa = get_tp_group().device_communicator.ca_comm
        assert fa is not None and not fa.disabled, "custom all-reduce is disabled"

        # Sizes count 16-byte packs. They bracket the fork's 512 KiB crossover
        # and cover both even and uneven partitions.
        for dtype, bits in [
            (torch.float16, torch.int16),
            (torch.bfloat16, torch.int16),
            (torch.float32, torch.int32),
        ]:
            for packed in [1021, 32767, 32768, 32771, 65536, 262139]:
                numel = packed * (16 // torch.empty((), dtype=dtype).element_size())
                case = f"{dtype}, {packed * 16} bytes"

                ints = torch.randint(
                    -8, 9, (numel, tp_size), generator=torch.Generator().manual_seed(0)
                ).to(dtype)
                exact = ints.sum(dim=1).to(device)
                inputs = _order_sensitive_inputs(numel, dtype, tp_size, seed=packed)
                inp = inputs[:, rank].contiguous().to(device)
                assert fa.should_custom_ar(inp), case
                outs = {}
                for algo in ["1stage", "2stage"]:
                    m.setenv("VLLM_CUSTOM_ALLREDUCE_ALGO", algo)
                    summed = fa.all_reduce(
                        ints[:, rank].contiguous().to(device), registered=False
                    )
                    torch.testing.assert_close(summed, exact, rtol=0, atol=0)
                    outs[algo] = fa.all_reduce(inp, registered=False)
                torch.accelerator.synchronize()

                differ = outs["1stage"].view(bits) != outs["2stage"].view(bits)
                assert not differ.any(), (
                    f"{case}: 2-stage differs from 1-stage in "
                    f"{int(differ.sum())}/{numel} elements"
                )


@pytest.mark.parametrize("tp_size", [4, 8])
def test_one_and_two_stage_allreduce_bitwise_identical(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
):
    """Needs at least 4 ranks: at 2, the 2-stage owner rotation only turns a+b into b+a,
    which is commutative, so no 2-rank run can tell the orders apart.
    """
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    if not car.custom_ar:
        pytest.skip("Custom all-reduce ops are not available.")
    physical_ids = [
        current_platform.visible_device_id_to_physical_device_id(i)
        for i in range(tp_size)
    ]
    if not current_platform.is_fully_connected(physical_ids):
        pytest.skip("Custom all-reduce needs fully connected GPUs above 2 ranks.")
    multi_process_parallel(monkeypatch, tp_size, 1, one_and_two_stage_allreduce)
