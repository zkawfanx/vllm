# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    BuildPrefillChunkMetadataKernel,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width


def _make_indexer_test_config(
    max_model_len: int, max_num_batched_tokens: int = 8
) -> SimpleNamespace:
    """Minimal builder config without model-registry/native-extension imports."""
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=8,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
        attention_config=SimpleNamespace(use_fp4_indexer_cache=False),
    )


def _mapping_metadata(starts, num_tokens):
    starts_cpu = torch.tensor(starts, dtype=torch.int32)
    num_reqs = len(starts) - 1
    return CommonAttentionMetadata(
        query_start_loc=starts_cpu.cuda(),
        query_start_loc_cpu=starts_cpu,
        seq_lens=torch.zeros(num_reqs, dtype=torch.int32, device="cuda"),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(torch.diff(starts_cpu).tolist(), default=0),
        max_seq_len=32,
        block_table_tensor=torch.empty((num_reqs, 1), device="cuda"),
        slot_mapping=torch.empty(num_tokens, dtype=torch.int64, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("starts", "num_tokens"),
    [
        ([0, 3, 5], 5),
        ([0, 129, 257, 520], 768),
        ([0, 0, 3, 3, 5, 5], 8),
        ([0, 3, 5], 3),
        ([0, 0, 0], 4),
        ([0], 4),
        ([0], 0),
    ],
)
def test_common_token_mapping_preserves_padding_and_cache(starts, num_tokens):
    common = _mapping_metadata(starts, num_tokens)
    output_size = max(starts[-1], num_tokens)
    buffer = torch.full((output_size + 2,), -1, dtype=torch.int32, device="cuda")
    unused_buffer = torch.full_like(buffer, -2)
    actual = common.token_to_req_indices(buffer)
    expected = [
        request
        for request in range(len(starts) - 1)
        for _ in range(starts[request + 1] - starts[request])
    ]
    expected += [0] * (output_size - len(expected))
    assert actual.tolist() == expected[:num_tokens]
    assert common._token_to_req_indices_cache.tolist() == expected
    assert buffer[output_size:].tolist() == [-1, -1]
    assert common.token_to_req_indices(unused_buffer).data_ptr() == actual.data_ptr()
    assert unused_buffer.tolist() == [-2] * len(buffer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_common_token_mapping_refreshes_across_batch_totals():
    buffer = torch.full((16,), -1, dtype=torch.int32, device="cuda")
    for starts in ([0, 2, 5], [0, 0, 3], [0, 0, 0], [0, 4, 6]):
        common = _mapping_metadata(starts, 8)
        actual = common.token_to_req_indices(buffer)
        expected = [
            request
            for request in range(len(starts) - 1)
            for _ in range(starts[request + 1] - starts[request])
        ]
        assert actual.tolist() == expected + [0] * (8 - starts[-1])
        assert actual.data_ptr() == buffer.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_common_token_mapping_replays_device_splits_with_stale_cpu_cache():
    """Replay refreshes the cached buffer without rereading CPU request splits."""
    buffer = torch.empty(8, dtype=torch.int32, device="cuda")
    warmup = _mapping_metadata([0, 2, 2, 6], 8)
    warmup.token_to_req_indices(buffer)
    torch.accelerator.synchronize()
    common = _mapping_metadata([0, 2, 2, 6], 8)
    unused_buffer = torch.full_like(buffer, -1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mapped = common.token_to_req_indices(buffer)
        reused = common.token_to_req_indices(unused_buffer)
    assert reused.data_ptr() == mapped.data_ptr() == buffer.data_ptr()
    for starts in ([0, 1, 5, 6], [0, 0, 6, 6], [0, 3, 3, 6]):
        common.query_start_loc.copy_(torch.tensor(starts, device="cuda"))
        graph.replay()
        expected = [
            request
            for request in range(3)
            for _ in range(starts[request + 1] - starts[request])
        ]
        assert mapped.tolist() == expected + [0, 0]
        assert common.token_to_req_indices(unused_buffer).tolist() == expected + [0, 0]
        assert common.query_start_loc_cpu.tolist() == [0, 2, 2, 6]
        assert unused_buffer.tolist() == [-1] * 8


def test_indexer_warmup_normalizes_zero_compress_ratios():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[0, 0, 4, 128, 0])
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = BuildPrefillChunkMetadataKernel().get_warmup_keys(config)

    assert {key.COMPRESS_RATIO for key in keys} == {1, 4, 128}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # storage_block_size = block_size // compress_ratio = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    vllm_config = _make_indexer_test_config(
        max_model_len=1024, max_num_batched_tokens=40
    )
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(max_num_blocks, kv_cache_spec.block_size)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )

    # Construct a single request where:
    # - num_computed = 240 (=> compressed_pos_start = 60)
    # - query_len = 40 (=> num_groups = 10)
    # => compressed positions are 60..69 which cross the storage block boundary at 64.
    query_start_loc = torch.tensor([0, 40], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([280], dtype=torch.int32, device=device)  # 240 + 40

    # Two blocks: compressed positions 0..63 map to block 5, 64..127 map to block 7.
    block_table_tensor = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    # Dummy uncompressed slot mapping (length == uncompressed num_actual_tokens).
    slot_mapping = torch.full((40,), -123, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=280,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # The compressed slot_mapping retains the original uncompressed size (40).
    # Only every compress_ratio-th position gets a valid slot; the rest are -1.
    assert md.slot_mapping.numel() == 40
    valid_slots = md.slot_mapping[md.slot_mapping >= 0]
    assert valid_slots.numel() == 10  # 40 tokens / compress_ratio 4

    storage_bs = kv_cache_spec.storage_block_size  # 64
    # Compressed positions 60..63 land in block 5, positions 64..69 in block 7.
    expected = torch.tensor(
        [
            5 * storage_bs + 60,
            5 * storage_bs + 61,
            5 * storage_bs + 62,
            5 * storage_bs + 63,
        ]
        + [
            7 * storage_bs + 0,
            7 * storage_bs + 1,
            7 * storage_bs + 2,
            7 * storage_bs + 3,
            7 * storage_bs + 4,
            7 * storage_bs + 5,
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(valid_slots, expected)


@pytest.mark.parametrize(
    (
        "compress_ratio",
        "seq_len",
        "max_seq_len",
        "expected_compressed_len",
        "expected_compressed_max",
    ),
    [
        (1, 1031, 1031, 1031, 1031),
        (4, 1031, 1031, 257, 257),
        (128, 131071, 131071, 1023, 1023),
        (4, 131072, 262400, 32768, 65600),
        (128, 131072, 262400, 1024, 2050),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_decode_max_seq_len_matches_compressed_lengths(
    compress_ratio: int,
    seq_len: int,
    max_seq_len: int,
    expected_compressed_len: int,
    expected_compressed_max: int,
):
    """Decode bounds and lengths must use compressed cache coordinates."""
    device = torch.device("cuda")
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
    )
    # Keep this metadata regression independent of model-registry subprocesses
    # and compiled attention extensions. The builder only needs these fields.
    vllm_config = _make_indexer_test_config(max_model_len=max_seq_len)
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, max_seq_len)
    block_table_width = get_block_table_width(max_num_blocks, kv_cache_spec.block_size)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )

    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    original_seq_lens = seq_lens.clone()
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=max_seq_len,
        block_table_tensor=torch.zeros(
            (1, block_table_width), dtype=torch.int32, device=device
        ),
        slot_mapping=torch.zeros(1, dtype=torch.int64, device=device),
        causal=True,
    )

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert metadata.max_seq_len == max_seq_len
    assert metadata.decode is not None
    assert metadata.decode.max_seq_len == expected_compressed_max
    assert metadata.decode.seq_lens.item() == expected_compressed_len
    torch.testing.assert_close(seq_lens, original_seq_lens)
