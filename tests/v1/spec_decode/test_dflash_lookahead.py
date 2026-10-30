# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.platforms import current_platform
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Matches defaults from tests/v1/spec_decode/test_eagle.py
DFLASH_TARGET_DIR = "Qwen/Qwen3-8B"
DFLASH_DRAFT_DIR = "z-lab/Qwen3-8B-DFlash-b16"

BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_SPECULATIVE_TOKENS = 3


def _dflash_speculative_config(num_speculative_tokens: int) -> SpeculativeConfig:
    model_config = ModelConfig(
        model=DFLASH_TARGET_DIR,
        runner="generate",
        max_model_len=100,
        trust_remote_code=True,
    )
    return SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=DFLASH_DRAFT_DIR,
        method="dflash",
        num_speculative_tokens=num_speculative_tokens,
    )


def _create_dflash_scheduler(num_speculative_tokens: int) -> Scheduler:
    speculative_config = _dflash_speculative_config(num_speculative_tokens)
    model_config = speculative_config.target_model_config
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = NUM_BLOCKS
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def test_dflash_prefill_reserves_lookahead_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    assert scheduler.num_lookahead_tokens == NUM_SPECULATIVE_TOKENS + 1

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == BLOCK_SIZE
    # prefill block + one lookahead block
    assert len(output.scheduled_new_reqs[0].block_ids[0]) == 2


def test_dflash_first_prefill_query_window_fits_allocated_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()
    block_ids = output.scheduled_new_reqs[0].block_ids[0]
    query_positions = range(BLOCK_SIZE, BLOCK_SIZE + scheduler.num_lookahead_tokens)

    assert all(pos // BLOCK_SIZE < len(block_ids) for pos in query_positions)


def test_dflash_drafter_window_reserves_bonus_token():
    # DFlash's drafter window is num_spec + 1 (the extra slot is the bonus token),
    # so max_seq_len + num_spec + 1 must stay within the draft model's max len.
    input_fits_in_drafter = GPUModelRunner._input_fits_in_drafter
    dflash_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=_dflash_speculative_config(NUM_SPECULATIVE_TOKENS),
    )
    # window = 4, so 96 fits (96 + 4 == 100) but 97 does not (97 + 4 == 101)
    assert input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=96))
    assert not input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=97))
    assert not input_fits_in_drafter(dflash_runner, None)  # no metadata

    # Other drafters don't reserve the bonus token, so 97 fits (97 + 3 == 100).
    plain_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=SimpleNamespace(use_dflash=lambda: False),
    )
    assert input_fits_in_drafter(plain_runner, SimpleNamespace(max_seq_len=97))


@pytest.mark.cpu_test
def test_dflash_forward_exposes_padding_to_moe_router(monkeypatch):
    from vllm.config import DeviceConfig
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        _get_padding_mask,
    )
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    buffers = InputBuffers(4, 16, device=torch.device("cpu"))
    buffers.is_padding[2:] = True

    def route(input_ids, positions, inputs_embeds):
        return _get_padding_mask(input_ids.shape[0])

    draft = SimpleNamespace(
        input_buffers=buffers,
        vllm_config=VllmConfig(device_config=DeviceConfig(device="cpu")),
        model=route,
    )
    mask = DFlashSpeculator._run_model(draft, 8, None, None, None)
    assert mask is not None
    assert mask.tolist() == [False, False] + [True] * 6
    buffers.is_padding[:4] = False
    assert mask.tolist() == [False] * 4 + [True] * 4


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_dflash_query_padding_resets_reused_buffers(sample_from_anchor):
    if not current_platform.is_cuda_alike() or not torch.accelerator.is_available():
        pytest.skip("requires CUDA or ROCm")
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

    device = torch.device("cuda")
    max_reqs, max_tokens, num_spec = 4, 16, 3
    num_query = num_spec if sample_from_anchor else num_spec + 1
    target_buffers = InputBuffers(max_reqs, max_tokens, device=device)
    buffers = InputBuffers(max_reqs, max_tokens, device=device)
    buffers.is_padding.fill_(True)
    query_slots = torch.full((max_tokens,), -2, dtype=torch.int64, device=device)
    context_positions = torch.empty_like(query_slots)
    context_slots = torch.empty_like(query_slots)
    sample_indices = torch.empty(max_reqs * num_spec, dtype=torch.int64, device=device)
    sample_pos = torch.empty_like(sample_indices)
    sample_mapping = torch.empty_like(sample_indices)
    temperature = torch.empty(max_reqs, device=device)
    seeds = torch.empty(max_reqs, dtype=torch.int64, device=device)
    last_sampled = torch.full_like(seeds, 99)
    block_table = torch.arange(1, 17, dtype=torch.int32, device=device).view(4, 4)
    for num_reqs in (1, 2, 1):
        batch = InputBatch.make_dummy(num_reqs, num_reqs * 2, target_buffers)
        batch.positions.copy_(torch.arange(num_reqs * 2, device=device))
        prepare_dflash_inputs(
            buffers,
            query_slots,
            context_positions,
            context_slots,
            sample_indices,
            sample_pos,
            sample_mapping,
            temperature,
            seeds,
            batch,
            torch.ones(num_reqs, dtype=torch.int32, device=device),
            torch.zeros(num_reqs, dtype=torch.int32, device=device),
            last_sampled,
            torch.zeros_like(seeds),
            torch.ones_like(temperature),
            torch.ones_like(seeds),
            block_table,
            16,
            123,
            num_query,
            num_spec,
            max_reqs,
            max_tokens,
            64,
            sample_from_anchor,
        )
        num_tokens = num_reqs * num_query
        assert not buffers.is_padding[:num_tokens].any()
        assert buffers.is_padding[num_tokens:].all()
        assert (buffers.input_ids[num_tokens:] == 0).all()
        assert (buffers.positions[num_tokens:] == 0).all()
        assert (query_slots[num_tokens:] == PAD_SLOT_ID).all()
        assert (query_slots[:num_tokens] > 0).all()
        assert buffers.input_ids[:num_tokens:num_query].tolist() == [99] * num_reqs
        assert (sample_mapping[num_reqs * num_spec :] == -1).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize("draft_width", [4, 16])
def test_dflash_context_staging_uses_target_aux_width(monkeypatch, draft_width):
    """HC-expanded draft buffers must not widen mean-pooled target features."""
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as mod
    from vllm.config.compilation import CUDAGraphMode

    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v41", dspark_target_layer_ids=[1, 14]
            )
        )
    )

    def init_base(self, config, device):
        self.speculative_config = spec_config
        self.draft_model_config = spec_config.draft_model_config
        self.max_num_tokens, self.max_num_reqs = 10, 4
        self.num_speculative_steps, self.hidden_size = 2, draft_width
        self.dtype = torch.float32

    monkeypatch.setattr(mod.DraftModelSpeculator, "__init__", init_base)
    monkeypatch.setattr(mod, "get_parallel_drafting_token_id", lambda _: 7)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda _: False,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: 6),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
    )
    draft = mod.DFlashSpeculator(config, torch.device("cpu"))
    assert draft.hidden_states.shape == (10, draft_width)
    assert draft._aux_staging.shape == (10, 12)
    assert draft._aux_staging.dtype == torch.float32
    assert not draft._aux_staging.any()


@pytest.mark.cpu_test
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("use_aux", [False, True])
@pytest.mark.parametrize("layer_groups", [None, [1, 0, 1]])
def test_dflash_context_capture_preserves_cache_rows(
    monkeypatch, full_graph, use_aux, layer_groups
):
    """Replay inserts each live context row once and never writes padded slots."""
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as mod
    from vllm.config.compilation import CUDAGraphMode

    draft = mod.DFlashSpeculator.__new__(mod.DFlashSpeculator)
    draft.max_num_tokens, draft.max_num_reqs = 10, 4
    draft.num_speculative_steps, draft.num_query_per_req = 3, 3
    draft.dp_size, draft.dp_rank = 1, 0
    draft.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
    draft._speculator_name = "DSpark"
    draft.sample_from_anchor = True
    draft.hidden_states = torch.zeros(10, 2)
    draft._aux_staging = torch.zeros(10, 4) if use_aux else None
    draft.context_positions = torch.full((10,), 99, dtype=torch.int64)
    draft._context_slot_mappings = torch.zeros(2, 10, dtype=torch.int64)
    draft._layer_group_idx, draft._group_causal = layer_groups, True
    draft.draft_kv_cache_group_id, draft.draft_kv_cache_group_ids = 0, [0, 1]
    draft.input_buffers = SimpleNamespace()
    draft.block_tables = SimpleNamespace(
        slot_mappings=torch.zeros(2, 10, dtype=torch.int64),
        input_block_tables=[None, None],
        kernel_block_sizes=[16, 16],
    )
    draft.attn_groups, draft.kv_cache_config = [], None
    for name in (
        "sample_indices",
        "sample_pos",
        "sample_idx_mapping",
        "temperature",
        "seeds",
    ):
        setattr(draft, name, torch.zeros(4))
    draft.parallel_drafting_token_id = 7
    writes, combined, observed = [], [], []

    def combine(hidden):
        combined.append(hidden.shape[0])
        return hidden[:, :2] + hidden[:, 2:]

    def insert(hidden, positions, slots=None):
        if slots is None:
            return
        for layer, mapping in enumerate(slots if isinstance(slots, list) else [slots]):
            for row in range(len(mapping)):
                if mapping[row] >= 0:
                    writes.append((layer, int(mapping[row]), hidden[row].tolist()))

    draft.model = SimpleNamespace(
        combine_hidden_states=combine, precompute_and_store_context_kv=insert
    )
    draft._prepare_eplb_forward = lambda _: None
    draft._build_draft_attn_metadata = lambda **_: None
    draft._generate_draft = lambda *_, **__: observed.append(list(writes))
    draft.draft_tokens = torch.zeros(4, 3, dtype=torch.int64)
    captured = {}

    def capture(forward, *args, **kwargs):
        assert not draft.context_positions.any()
        assert (draft._context_slot_mappings == -1).all()
        captured.update(kwargs)
        if context := kwargs.get("prepare_context"):
            context(2)
        forward()

    def replay(desc):
        context_rows = min(desc.num_reqs * 4, 10)
        layers = len(layer_groups) if layer_groups else 1
        assert len(writes) == (0 if dummy else max(0, count - context_rows) * layers)
        if context := captured.get("prepare_context"):
            context(desc.num_reqs)
        draft._generate_draft()

    draft.query_cudagraph_manager = SimpleNamespace(
        capture=capture, run_fullgraph=replay
    )
    monkeypatch.setattr(mod, "build_slot_mappings_by_layer", lambda *args: {})
    mode = CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE
    descriptor = SimpleNamespace(cg_mode=mode, num_reqs=2, num_tokens=6)
    monkeypatch.setattr(
        mod, "dispatch_cg_and_sync_dp", lambda *a, **k: (descriptor, None)
    )

    def prepare(*args):
        n = args[9].num_tokens
        args[2][:n] = torch.arange(n)
        group = int(args[3].data_ptr() != draft._context_slot_mappings[0].data_ptr())
        args[3][:n] = torch.arange(n) + 1 + 100 * group

    monkeypatch.setattr(mod, "prepare_dflash_inputs", prepare)
    draft.capture()
    assert writes == []
    staging_ptr = draft._aux_staging.data_ptr() if use_aux else None
    # Overflow, smaller replay, maximum capacity, and dummy replay.
    for count, graph_reqs, dummy in [
        (10, 2, False),
        (3, 2, False),
        (10, 3, False),
        (10, 2, True),
    ]:
        writes.clear()
        combined.clear()
        descriptor.num_reqs = graph_reqs
        descriptor.num_tokens = graph_reqs * 3
        states = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
        strided = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4)
        aux = [strided[:, ::2], strided[:, 1::2]]
        expected = aux[0] + aux[1] if use_aux else states
        batch = SimpleNamespace(
            num_reqs=2,
            num_tokens=count,
            seq_lens_cpu_upper_bound=torch.tensor([20, 20]),
        )
        draft.propose(
            batch,
            {},
            {},
            states,
            aux if use_aux else None,
            *[torch.zeros(4) for _ in range(6)],
            dummy_run=dummy,
        )
        expected_writes = (
            []
            if dummy
            else [
                (
                    layer,
                    row + 1 + 100 * (layer_groups[layer] if layer_groups else 0),
                    expected[row].tolist(),
                )
                for layer in range(len(layer_groups) if layer_groups else 1)
                for row in range(count)
            ]
        )
        assert sorted(writes) == sorted(expected_writes)
        assert observed[-1] == writes
        if full_graph and use_aux:
            context = min(graph_reqs * 4, 10)
            assert combined == ([count - context] if count > context else []) + [
                context
            ]
            assert draft._aux_staging.data_ptr() == staging_ptr


@pytest.mark.cpu_test
def test_dflash_capture_inserts_context_before_query(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as mod
    from vllm.config.compilation import CUDAGraphMode

    manager = mod.DFlashCudaGraphManager.__new__(mod.DFlashCudaGraphManager)
    manager.max_num_reqs, manager.dp_size = 4, 1
    order = []
    desc = SimpleNamespace(num_tokens=6, num_reqs=2, cg_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(
        mod, "_prepare_dflash_inputs_to_capture", lambda *a, **k: (None, {})
    )

    def capture(self, create, progress):
        for warmup in (True, False):
            create(desc, warmup)(CUDAGraphMode.FULL)

    monkeypatch.setattr(mod.CudaGraphManager, "capture", capture)
    manager.capture(
        lambda *a: order.append(("query", a[0])),
        None,
        None,
        [],
        None,
        100,
        True,
        prepare_context=lambda n: order.append(("context", n)),
    )
    assert order == [("context", 2), ("query", 2)] * 2


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="requires CUDA",
)
def test_dflash_context_graph_replay_reads_new_aux_and_slots():
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    draft = DFlashSpeculator.__new__(DFlashSpeculator)
    draft.max_num_tokens, draft.num_speculative_steps = 10, 3
    draft._aux_staging = torch.zeros(8, 4, device="cuda")
    draft.hidden_states = torch.zeros(10, 2, device="cuda")
    draft.context_positions = torch.zeros(10, dtype=torch.int64, device="cuda")
    draft._context_slot_mappings = torch.full((1, 10), -1, device="cuda")
    draft._layer_group_idx = None
    cache = torch.zeros(11, 2, device="cuda")

    def insert(hidden, positions, slots):
        # Slot10 is scratch for padded rows; live slots are0..9.
        cache.index_copy_(0, torch.where(slots >= 0, slots, 10), hidden)

    draft.model = SimpleNamespace(
        combine_hidden_states=lambda x: x[:, :2] + x[:, 2:],
        precompute_and_store_context_kv=insert,
    )
    draft._prepare_graph_context(2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        draft._prepare_graph_context(2)
    for count in (10, 3, 8):
        cache.fill_(-1)
        states = torch.randn(count, 4, device="cuda")
        draft._context_slot_mappings[0, :count] = torch.arange(count, device="cuda")
        draft._prepare_graph_replay(count, 2, states[:, :2], [states], False)
        graph.replay()
        torch.testing.assert_close(cache[:count], states[:, :2] + states[:, 2:])
        assert (cache[count:10] == -1).all()
