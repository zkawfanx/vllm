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
