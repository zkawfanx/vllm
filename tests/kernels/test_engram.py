# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1.common.engram import Engram, ParallelEngramEmbedding
from vllm.platforms import current_platform


def _reference_lookup(weight, scale_inv, ids, start, end, block=32):
    """The torch expression the fused kernel replaces."""
    mask = (ids < start) | (ids >= end)
    local = (ids - start).masked_fill(mask, 0).long()
    values = torch.nn.functional.embedding(local, weight)
    scales = torch.nn.functional.embedding(local, scale_inv)
    scales = (scales.to(torch.int32) << 23).view(torch.float32)
    values = values.float().unflatten(-1, (-1, block))
    values = (values * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    return values.masked_fill(mask.unsqueeze(-1), 0)


def _make_embedding(cpu_offload, rows=4096, dim=256, block=32):
    layer = ParallelEngramEmbedding.__new__(ParallelEngramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.dim, layer.block_size, layer.tp_size = dim, block, 1
    layer.n_hash_cols = layer.part_n_hash_cols = 24
    layer.head_start = 0
    layer.cpu_offload = cpu_offload
    layer.part_num_embeddings = rows
    # A window strictly inside the table, so unowned rows are exercised too.
    layer.vocab_start_idx, layer.vocab_end_idx = rows // 4, rows // 4 + rows // 2
    layer._views = layer._view_src = None
    layer._num_sms = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).multi_processor_count
    kwargs = (
        {"device": "cpu", "pin_memory": True} if cpu_offload else {"device": "cuda"}
    )
    owned = layer.vocab_end_idx - layer.vocab_start_idx
    layer.weight = torch.empty(owned, dim, dtype=torch.float8_e4m3fn, **kwargs)
    layer.weight_scale_inv = torch.empty(
        owned, dim // block, dtype=torch.uint8, **kwargs
    )
    torch.manual_seed(0)
    layer.weight.copy_((torch.randn(owned, dim) * 4).to(torch.float8_e4m3fn))
    layer.weight_scale_inv.copy_(
        torch.randint(120, 134, (owned, dim // block), dtype=torch.uint8)
    )
    return layer


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
@pytest.mark.parametrize(
    "cpu_offload,delay",
    [(False, None), (True, None), (True, "producer"), (True, "lookup")],
)
@pytest.mark.parametrize("capture", ["eager", "full", "breakable"])
def test_engram_prepared_rows_survive_graph_breaks(cpu_offload, capture, delay):
    """Temporary lookup IDs survive allocator reuse and graph replay."""
    _run_engram_prepared_rows(cpu_offload, capture, delay=delay)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
@pytest.mark.parametrize(
    "missing_dependency", ["record_stream", "producer_wait", "consumer_wait"]
)
def test_engram_prefetch_detects_missing_dependency(monkeypatch, missing_dependency):
    """Negative controls must expose incorrect rows when a dependency is removed."""
    if (
        missing_dependency == "record_stream"
        and torch.cuda.memory.get_allocator_backend() != "native"
    ):
        pytest.skip(
            "The allocator reuse negative control requires the native allocator"
        )

    def start(self, hash_ids, rows, stream):
        if missing_dependency != "producer_wait":
            stream.wait_stream(torch.cuda.current_stream())
        if missing_dependency != "record_stream":
            hash_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            self.embed_tokens.lookup(hash_ids, rows, background=True)
            self._prefetch_done.record(stream)

    def finish(self, event):
        pass

    monkeypatch.setattr(Engram, "_start_prefetch", start)
    if missing_dependency == "consumer_wait":
        monkeypatch.setattr(Engram, "_finish_prefetch", finish)
    delay = "producer" if missing_dependency == "producer_wait" else "lookup"
    try:
        with pytest.raises(
            AssertionError, match=r"Tensor-likes are not (?:equal|close)!"
        ):
            _run_engram_prepared_rows(True, "eager", delay=delay)
    finally:
        torch.accelerator.synchronize()


def _run_engram_prepared_rows(
    cpu_offload, capture, tp_size=1, rank=0, use_sequence_parallel=False, delay=None
):
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )

    layer = _make_embedding(cpu_offload)
    if delay == "lookup":
        lookup = layer.lookup

        def delayed_lookup(indices, out, background=False):
            torch.cuda._sleep(2_000_000)
            lookup(indices, out, background=background)

        layer.lookup = delayed_lookup
    cols, num_tokens = (23, 65) if use_sequence_parallel else (24, 64)
    layer.n_hash_cols = cols
    layer.tp_size = tp_size
    layer.part_n_hash_cols = (cols + tp_size - 1) // tp_size
    layer.head_start = rank * layer.part_n_hash_cols
    engram = Engram.__new__(Engram)
    torch.nn.Module.__init__(engram)
    engram.embed_tokens = layer
    engram._prefetch_stream = torch.cuda.Stream() if cpu_offload else None
    engram._prefetch_done = torch.cuda.Event() if cpu_offload else None
    # Exercise the production eager boundaries even when the test process
    # imported Engram before breakable graphs were enabled.
    if capture == "breakable":
        import vllm.envs as envs

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", True)
            for name in ("_start_prefetch", "_finish_prefetch"):
                fn = eager_break_during_capture(inspect.unwrap(getattr(Engram, name)))
                setattr(engram, name, fn.__get__(engram, Engram))
    engram.use_sequence_parallel = use_sequence_parallel
    engram.staged_rows = torch.empty(
        num_tokens,
        layer.part_n_hash_cols,
        layer.dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Match the non-contiguous per-layer slice of the model hash tensor.
    hashes = torch.randint(
        0,
        layer.part_num_embeddings,
        (num_tokens, 2, cols),
        dtype=torch.int32,
        device="cuda",
    )
    src = hashes[:, 1]
    local_tokens = (
        (num_tokens + tp_size - 1) // tp_size if use_sequence_parallel else num_tokens
    )
    out = torch.empty(
        local_tokens, cols, layer.dim, dtype=torch.bfloat16, device="cuda"
    )
    embed = engram.embed

    def step(cap=None):
        if delay == "producer":
            torch.cuda._sleep(2_000_000)
        # Drop the last reference to a non-contiguous input after launching lookup.
        engram.prepare_embeddings(hashes.clone()[:, 1])
        # Exercise same-size allocator reuse before consuming the prefetched rows.
        torch.empty_like(hashes).fill_(layer.part_num_embeddings - 1)
        if cap is not None:
            cap.add_eager(lambda: None)
        out.copy_(embed(src))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)

    graph = None
    if capture == "full":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    elif capture == "breakable":
        graph = BreakableCUDAGraphCapture()
        with torch.cuda.stream(warmup), graph:
            step(graph)
        torch.cuda.current_stream().wait_stream(warmup)
        assert graph.num_eager_breaks == (3 if cpu_offload else 1)

    for _ in range(3):
        # Eager prefill can overwrite staging rows between decode replays.
        hashes.random_(0, layer.part_num_embeddings)
        hashes[0].fill_(-1)
        step()
        hashes.random_(0, layer.part_num_embeddings)
        hashes[0].fill_(-1)
        if graph is None:
            step()
        else:
            graph.replay()
        expected = _reference_lookup(
            layer.weight.cuda(),
            layer.weight_scale_inv.cuda(),
            src,
            layer.vocab_start_idx,
            layer.vocab_end_idx,
        )
        if use_sequence_parallel:
            expected = torch.nn.functional.pad(
                expected, (0, 0, 0, 0, 0, (-num_tokens) % tp_size)
            )[rank * local_tokens : (rank + 1) * local_tokens]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _engram_tp_worker(rank, tp_size, port):
    from tests.utils import init_test_distributed_environment
    from vllm.distributed import cleanup_dist_env_and_memory

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(tp_size, 1, rank, str(port), local_rank=rank)
    try:
        for cpu_offload in (False, True):
            for sp in (False, True):
                for capture in ("eager", "full", "breakable"):
                    _run_engram_prepared_rows(cpu_offload, capture, tp_size, rank, sp)
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="CUDA required",
)
def test_engram_head_collectives_survive_graph_breaks():
    """All-gather preserves head order and local SP tokens across graph replay."""
    from vllm.utils.network_utils import get_open_port

    if torch.accelerator.device_count() < 8:
        pytest.skip("Requires eight GPUs")
    torch.multiprocessing.spawn(_engram_tp_worker, args=(8, get_open_port()), nprocs=8)


@pytest.mark.cpu_test
def test_engram_consumer_completes_only_its_layer_on_shared_stream(monkeypatch):
    """A layer event releases its rows without draining later lookups."""
    work: list[Callable[[], None]] = []
    waits = []
    retained = []

    class Stream:
        def wait_stream(self, producer):
            waits.append(producer)

        def wait_event(self, event):
            for operation in work[: event.end]:
                operation()
            del work[: event.end]

    class Event:
        def record(self, stream):
            assert stream is side
            self.end = len(work)

    main, side = Stream(), Stream()
    active = main

    @contextmanager
    def use_stream(stream):
        nonlocal active
        previous, active = active, stream
        try:
            yield
        finally:
            active = previous

    def lookup(ids, rows, background=False):
        assert active is side and background
        work.append(lambda: rows.copy_(ids.unsqueeze(-1)))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: active)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: retained.append((tensor, stream)),
    )
    modules = []
    for _ in range(2):
        module = Engram.__new__(Engram)
        torch.nn.Module.__init__(module)
        module.embed_tokens = SimpleNamespace(tp_size=1, lookup=lookup)
        module._prefetch_stream, module._prefetch_done = side, Event()
        module.staged_rows = torch.full((2, 1, 1), -1.0)
        modules.append(module)
    first, second = torch.tensor([[2.0], [3.0]]), torch.tensor([[5.0], [7.0]])
    modules[0].prepare_embeddings(first)
    modules[1].prepare_embeddings(second)
    assert waits == [main, main]
    assert retained == [(first, side), (second, side)]
    assert (modules[0].staged_rows == -1).all()
    rows = modules[0].embed(first)
    torch.testing.assert_close(rows.squeeze(-1), first)
    assert (modules[1].staged_rows == -1).all()
    assert len(work) == 1
