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


@pytest.fixture
def make_engram_wkv(monkeypatch):
    from vllm import distributed
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1.common import engram

    state = SimpleNamespace(rank=0, size=1)
    for module in (distributed, linear, parameter):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_rank", lambda: state.rank
        )
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: state.size
        )
    monkeypatch.setattr(
        engram,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            engram_config=SimpleNamespace(cpu_offload=False),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        ),
    )
    monkeypatch.setattr(
        engram,
        "ParallelEngramEmbedding",
        lambda *a, **kw: SimpleNamespace(cpu_offload=False, part_n_hash_cols=2),
    )

    def make(rank, size, sp, quant_config=None):
        state.rank, state.size = rank, size
        return Engram(
            SimpleNamespace(hidden_size=256, hc_mult=4, rms_norm_eps=1e-6),
            quant_config,
            SimpleNamespace(
                num_embeddings=[64],
                head_dim=32,
                primes=[[[31, 37]]],
                max_ngram_size=2,
                n_heads=2,
            ),
            0,
            use_sequence_parallel=sp,
            prefix="model.layers.1.engram",
        ).wkv

    return make


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("tp_size,sp", [(1, False), (1, True), (8, False), (8, True)])
def test_engram_wkv_preserves_output_order(make_engram_wkv, monkeypatch, tp_size, sp):
    """Keep keys then value intact; SP owns padded token rows, not columns."""
    from vllm.model_executor.layers import linear

    weight = torch.arange(1280 * 64, dtype=torch.float32).reshape(1280, 64) % 17
    inputs = torch.arange(9 * 64, dtype=torch.float32).reshape(9, 64) % 13
    layers, local_inputs, partials = [], [], []
    for rank in range(tp_size):
        layer = make_engram_wkv(rank, tp_size, sp)
        layer.weight.weight_loader(layer.weight, weight)
        width = 1280 if sp else 1280 // tp_size
        start = 0 if sp else rank * width
        torch.testing.assert_close(layer.weight, weight[start : start + width])
        rows = inputs
        if sp:
            chunk = (len(inputs) + tp_size - 1) // tp_size
            rows = torch.nn.functional.pad(inputs, (0, 0, 0, (-len(inputs)) % tp_size))
            rows = rows[rank * chunk : (rank + 1) * chunk]
        layers.append(layer)
        local_inputs.append(rows)
        partials.append(torch.nn.functional.linear(rows, layer.weight))

    gathered = torch.cat(partials, dim=-1) if not sp else None
    calls: list[torch.Tensor] = []

    def gather(partial):
        torch.testing.assert_close(partial, partials[len(calls)])
        calls.append(partial)
        return gathered

    monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", gather)
    for layer, rows in zip(layers, local_inputs):
        result = layer(rows)
        torch.testing.assert_close(result, torch.nn.functional.linear(rows, weight))
        assert result.shape == (len(rows), 5 * 256)
    assert len(calls) == (tp_size if tp_size > 1 and not sp else 0)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("tp_size,sp", [(1, False), (1, True), (8, False), (8, True)])
@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_engram_wkv_loads_matching_weight_and_scale_shards(
    make_engram_wkv, monkeypatch, tp_size, sp, expert_dtype
):
    """Exercise real checkpoint loaders, without selecting or running a GPU kernel."""
    from vllm.model_executor.layers.quantization import fp8, modelopt
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config

    monkeypatch.setattr(
        fp8,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    monkeypatch.setattr(fp8, "cutlass_block_fp8_supported", lambda: False)
    monkeypatch.setattr(fp8, "init_fp8_linear_kernel", lambda **kw: None)
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *a, **kw: SimpleNamespace(input_quant_key=lambda: None),
    )
    config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
    )
    config.use_deep_gemm = False
    config._resolved_expert_dtype = expert_dtype
    weight = (torch.arange(1280 * 64).reshape(1280, 64) % 17).to(torch.float8_e4m3fn)
    scale_bytes = (torch.arange(40 * 2).reshape(40, 2) % 15 + 120).to(torch.uint8)
    scales = (
        scale_bytes.view(torch.float8_e8m0fnu)
        if expert_dtype == "fp4"
        else scale_bytes.float()
    )
    expected_scales = (
        scale_bytes.repeat_interleave(32, 0) if expert_dtype == "fp4" else scales
    )
    for rank in range(tp_size):
        layer = make_engram_wkv(rank, tp_size, sp, config)
        scale = layer.weight_scale if expert_dtype == "fp4" else layer.weight_scale_inv
        layer.weight.weight_loader(layer.weight, weight)
        scale.weight_loader(scale, scales)
        effective_rank, size = (0, 1) if sp else (rank, tp_size)
        width = len(weight) // size
        torch.testing.assert_close(
            layer.weight.view(torch.uint8),
            weight[effective_rank * width : (effective_rank + 1) * width].view(
                torch.uint8
            ),
        )
        scale_width = len(expected_scales) // size
        torch.testing.assert_close(
            scale,
            expected_scales[
                effective_rank * scale_width : (effective_rank + 1) * scale_width
            ],
        )


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


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_engram_meta_initialization_does_not_allocate_host_tables(
    monkeypatch, cpu_offload
):
    """Encoder-only placeholders must not allocate pinned host storage."""
    from vllm.models.deepseek_v4_1.common import engram

    monkeypatch.setattr(engram, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(engram, "is_uva_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=1),
    )
    with torch.device("meta"):
        layer = ParallelEngramEmbedding(36, 64, (17, 19), cpu_offload=cpu_offload)

    assert layer.weight.is_meta and layer.weight_scale_inv.is_meta
    assert layer.weight.shape == (36, 64)
    assert layer.weight_scale_inv.shape == (36, 2)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.dtype == torch.uint8
    assert not hasattr(layer.weight, "dummy_weight_value")
    assert not hasattr(layer.weight_scale_inv, "dummy_weight_value")


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
