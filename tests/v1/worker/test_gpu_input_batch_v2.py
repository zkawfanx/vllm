# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, post_update
from vllm.v1.worker.gpu.sample.penalties import bincount

DEVICE = current_platform.device_type


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )


@pytest.mark.cpu_test
def test_dummy_request_indices_preserve_large_row_offsets():
    buffers = InputBuffers(3, 3, device=torch.device("cpu"))
    batch = InputBatch.make_dummy(3, 3, buffers)
    stride = 2**30
    expected = [0, stride, 2 * stride]
    assert (batch.idx_mapping * stride).tolist() == expected
    assert (batch.expanded_idx_mapping * stride).tolist() == expected
    assert (batch.idx_mapping_np * stride).tolist() == expected
    assert batch.expanded_local_pos.dtype == torch.int32


_ROW_STRIDE = 2**20
_HIGH_ROW = 2**11
_NUM_ROWS = _HIGH_ROW + 1


@pytest.fixture(scope="module")
def high_row_device():
    if not current_platform.is_cuda_alike():
        pytest.skip("requires CUDA or ROCm")
    required_bytes = _NUM_ROWS * _ROW_STRIDE * 4 + 2**30
    if torch.accelerator.get_memory_info(0)[0] < required_bytes:
        pytest.skip("requires 9 GiB free device memory for int32 row overflow")
    return torch.device("cuda:0")


@pytest.fixture
def high_row_inputs(high_row_device):
    tokens = torch.zeros(
        (_NUM_ROWS, _ROW_STRIDE), dtype=torch.int32, device=high_row_device
    )
    buffers = InputBuffers(_NUM_ROWS, _NUM_ROWS, device=high_row_device)
    batch = InputBatch.make_dummy(_NUM_ROWS, _NUM_ROWS, buffers)
    return tokens, batch.idx_mapping[[0, _HIGH_ROW]]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("uva_instead_of_gpu", [False, True])
def test_staged_write_chunks_preserve_ranges_and_reset(dtype, uva_instead_of_gpu):
    """Mixed-length writes must not overwrite gaps or retain old chunk counts."""
    staged = StagedWriteTensor(
        (4, 4096), dtype, torch.device("cuda"), uva_instead_of_gpu=uva_instead_of_gpu
    )
    values = list(range(2500))
    staged.stage_write(0, 3, iter(values))
    staged.stage_write(1, 2, [7])
    staged.stage_write(2, 0, iter(()))
    staged.stage_write(3, 0, [])
    staged.apply_write()
    staged.stage_write(0, 3, [9])
    staged.apply_write()
    # The next GPU consumer must observe writes without a host synchronization.
    observed = staged.gpu.clone()
    expected = torch.zeros_like(staged.gpu)
    expected[0, 3:2503] = torch.tensor(values, dtype=dtype, device="cuda")
    expected[0, 3] = 9
    expected[1, 2] = 7
    assert torch.equal(observed, expected)
    assert staged._staged_write_max_len == 0

    staged.stage_write_elem(0, 9)
    staged.apply_write()
    staged.stage_write(2, 0, iter(()))
    staged.apply_write()
    staged.apply_write()
    observed = staged.gpu.clone()
    expected[0, 0] = 9
    assert torch.equal(observed, expected)


def test_staged_write_preserves_large_row_offsets(high_row_device):
    """An int32 staged index must not wrap when multiplied by its row stride."""
    staged = StagedWriteTensor((_NUM_ROWS, _ROW_STRIDE), torch.int32, high_row_device)
    staged.stage_write(0, 1, [11, 12])
    staged.stage_write(_HIGH_ROW, 1, [21, 22])
    staged.apply_write()
    assert staged.gpu[0, :4].tolist() == [0, 11, 12, 0]
    assert staged.gpu[_HIGH_ROW, :4].tolist() == [0, 21, 22, 0]


def test_post_update_preserves_large_row_offsets_and_skips_padding(high_row_inputs):
    tokens, idx_mapping = high_row_inputs
    device = tokens.device
    idx_mapping = torch.cat((idx_mapping, idx_mapping.new_tensor([-1])))
    num_computed = torch.zeros(_NUM_ROWS, dtype=torch.int32, device=device)
    last_sampled = torch.full((_NUM_ROWS,), -1, dtype=torch.int64, device=device)
    total_len = torch.zeros(_NUM_ROWS, dtype=torch.int32, device=device)
    total_len[[0, _HIGH_ROW]] = torch.tensor(
        [50, 100], dtype=torch.int32, device=device
    )
    post_update(
        idx_mapping,
        num_computed,
        last_sampled,
        None,
        torch.tensor([[111], [222], [333]], dtype=torch.int64, device=device),
        torch.ones(3, dtype=torch.int32, device=device),
        torch.zeros(3, dtype=torch.int32, device=device),
        torch.arange(4, dtype=torch.int32, device=device),
        tokens,
        total_len,
    )
    assert tokens[0, 50].item() == 111
    assert tokens[_HIGH_ROW, 100].item() == 222
    assert last_sampled[[0, _HIGH_ROW]].tolist() == [111, 222]
    assert total_len[[0, _HIGH_ROW]].tolist() == [51, 101]
    assert num_computed[[0, _HIGH_ROW]].tolist() == [1, 1]
    assert tokens[1].count_nonzero().item() == 0


def test_penalty_counts_preserve_large_row_offsets(high_row_inputs):
    tokens, idx_mapping = high_row_inputs
    device = tokens.device
    tokens[0, :4] = tokens.new_tensor([1, 2, 3, 3])
    tokens[_HIGH_ROW, :4] = tokens.new_tensor([1, 2, 3, 3])
    prompt_len = torch.full((_NUM_ROWS,), 2, dtype=torch.int32, device=device)
    prefill_len = torch.full((_NUM_ROWS,), 4, dtype=torch.int32, device=device)
    prompt_mask = torch.zeros((_NUM_ROWS, 1), dtype=torch.int32, device=device)
    output_counts = torch.zeros((_NUM_ROWS, 32), dtype=torch.int32, device=device)
    bincount(
        idx_mapping,
        tokens,
        prompt_len,
        prefill_len,
        prompt_mask,
        output_counts,
        max_prefill_len=4,
    )
    assert prompt_mask[[0, _HIGH_ROW], 0].tolist() == [6, 6]
    assert output_counts[[0, _HIGH_ROW], 3].tolist() == [2, 2]
    assert output_counts.sum().item() == 4
