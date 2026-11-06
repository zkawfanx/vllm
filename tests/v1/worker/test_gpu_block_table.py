# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)


def test_block_tables_apply_staged_writes_fuses_kv_groups(monkeypatch):
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16, 32, 8],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8, 8, 8],
        device=device,
        kernel_block_sizes=[16, 16, 8],
    )

    def fail_if_apply_write_called():
        pytest.fail("multi-group writes should use the fused apply kernel")

    for block_table in block_tables.block_tables:
        monkeypatch.setattr(block_table, "apply_write", fail_if_apply_write_called)

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2], [10, 11], []),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([3], [12], [5, 6]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )
    # Group 1 has blocks_per_kv_block == 2, so each KV block expands to two
    # kernel block IDs.
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :4],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[0].gpu[1, :1],
        torch.tensor([3], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[1, :2],
        torch.tensor([24, 25], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[1, :2],
        torch.tensor([5, 6], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 2
    assert block_tables.num_blocks.np[1, 0] == 4
    assert block_tables.num_blocks.np[2, 0] == 0
    assert block_tables.num_blocks.np[0, 1] == 1
    assert block_tables.num_blocks.np[1, 1] == 2
    assert block_tables.num_blocks.np[2, 1] == 2
    assert torch.equal(
        block_tables.num_blocks.gpu[:, :2],
        torch.tensor([[2, 1], [4, 2], [0, 2]], dtype=torch.int32, device=device),
    )

    for block_table in block_tables.block_tables:
        assert not block_table._staged_write_indices
        assert not block_table._staged_write_starts
        assert not block_table._staged_write_contents
        assert not block_table._staged_write_cu_lens

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([7], [13], [8]),
        overwrite=False,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :3],
        torch.tensor([1, 2, 7], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :6],
        torch.tensor([20, 21, 22, 23, 26, 27], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[0, :1],
        torch.tensor([8], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 3
    assert block_tables.num_blocks.np[1, 0] == 6
    assert block_tables.num_blocks.np[2, 0] == 1


def test_block_tables_apply_staged_writes_single_group():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[16],
    )

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2],),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )


def test_v1_block_table_move_row_clears_vacated_row():
    """condense() moves the last row into a freed slot; the vacated row must
    not keep stale block ids. Padded dummy-run batches dereference stale rows
    as mamba state slots (bypassing the NULL_BLOCK_ID fill of real decode
    padding) and write state in place there — corrupting the blocks' new
    owner once they are reallocated, e.g. to an in-flight NIXL load."""
    from vllm.v1.worker.block_table import BlockTable

    block_table = BlockTable(
        block_size=16,
        max_num_reqs=4,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cuda"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
    )
    block_table.add_row([7, 8, 9], row_idx=0)
    block_table.add_row([4, 5], row_idx=1)

    block_table.move_row(1, 0)

    assert block_table.block_table.np[0, :2].tolist() == [4, 5]
    assert block_table.num_blocks_per_row[0] == 2
    # The vacated source row routes to the reserved null block.
    assert block_table.num_blocks_per_row[1] == 0
    assert (block_table.block_table.np[1] == 0).all()


def test_get_dummy_block_tables_returns_zeroed_rows():
    """Dummy runs bypass the gather, so the persistent input_block_tables
    hold the previous real step's rows. Mamba/GDN metadata routes in-place
    state writes through block_table[:, 0] (dummy slot mappings are
    PAD-filled, state indices are not), so stale rows would direct dummy
    state writes at freed — possibly reallocated — blocks.
    get_dummy_block_tables must hand out zeroed (null block) rows while
    preserving the persistent storage address for CUDA graphs."""
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8],
        device=device,
        kernel_block_sizes=[16],
    )
    # Simulate a real step: stage a request's blocks and gather them into
    # the persistent input block tables.
    block_tables.append_block_ids(req_index=0, new_block_ids=([1, 2],), overwrite=True)
    block_tables.apply_staged_writes()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
    block_tables.gather_block_tables(idx_mapping, num_reqs_padded=1)
    torch.accelerator.synchronize()
    assert block_tables.input_block_tables[0][0, 0].item() == 1

    dummy = block_tables.get_dummy_block_tables(num_reqs=1)
    torch.accelerator.synchronize()
    assert (dummy[0] == 0).all()
    # CUDA graph invariant: same persistent tensor, not a fresh allocation.
    assert dummy[0].data_ptr() == block_tables.input_block_tables[0].data_ptr()


@pytest.mark.parametrize("uva_target", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_staged_write_typed_inputs_and_ownership(uva_target, dtype):
    """Copied iterables and borrowed arrays interleave without changing row order."""
    import array
    import gc

    import numpy as np

    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    state = StagedWriteTensor(
        (8, 8), dtype, torch.device("cuda"), uva_instead_of_gpu=uva_target
    )
    for base in (0, 100):
        values = [base + 1, base + 2]
        state.stage_write(0, 1, values)
        values[:] = [-1, -1]  # Lists are snapshots, not borrowed buffers.
        owner = torch.arange(base + 3, base + 7, dtype=torch.int64)
        state.stage_write(1, 0, owner.numpy()[::2])
        del owner
        gc.collect()  # A NumPy view must retain its tensor storage until apply.
        state.stage_write(2, 2, (v for v in range(base + 7, base + 9)))
        state.stage_write(3, 1, array.array("i", [base + 9, base + 10]))
        state.stage_write_elem(4, base + 11)
        state.stage_write(4, 1, {0: base + 12, 1: base + 13}.values())
        state.stage_write(5, 0, np.array([], dtype=np.int64))
        state.stage_write(5, 0, iter(()))
        state.stage_write(6, 0, np.array([base + 14], dtype=np.int64))
        if dtype == torch.float32:
            state.stage_write(7, 0, {0: 0.25, 1: -0.5}.values())
        state.apply_write()
        # The consumer is ordered on the same stream, before host synchronization.
        result = state.gpu.clone()
        torch.accelerator.synchronize()
        expected = torch.zeros((8, 8), dtype=dtype)
        expected[0, 1:3] = torch.tensor([base + 1, base + 2])
        expected[1, :2] = torch.tensor([base + 3, base + 5])
        expected[2, 2:4] = torch.tensor([base + 7, base + 8])
        expected[3, 1:3] = torch.tensor([base + 9, base + 10])
        expected[4, :3] = torch.tensor([base + 11, base + 12, base + 13])
        expected[6, 0] = base + 14
        if dtype == torch.float32:
            expected[7, :2] = torch.tensor([0.25, -0.5])
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    state.stage_write(0, 0, np.arange(6, dtype=np.int64)[::2])
    state.apply_write()
    torch.accelerator.synchronize()
    assert state.gpu[0, :3].tolist() == [0, 2, 4]


def test_fused_staged_write_typed_chunks():
    import array

    import numpy as np

    from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor

    device = torch.device("cuda")
    tables = [StagedWriteTensor((4, 8), torch.int32, device) for _ in range(2)]
    writer = FusedStagedWriter(device, max_writes=8)
    ptrs = torch.tensor(
        [t.gpu.data_ptr() for t in tables], dtype=torch.uint64, device=device
    )
    strides = torch.tensor([t.gpu.stride(0) for t in tables], device=device)
    for base in (0, 100):
        for i, t in enumerate(tables):
            b = base + 10 * i
            t.stage_write(0, 0, [b + 1])
            t.stage_write(1, 2, np.array([b + 2, b + 3], dtype=np.int64))
            t.stage_write(2, 1, array.array("i", [b + 4]))
            t.stage_write_elem(3, b + 5)
        writer.apply(tables, ptrs, strides)
        torch.accelerator.synchronize()
        for i, t in enumerate(tables):
            b = base + 10 * i
            expected = torch.zeros((4, 8), dtype=torch.int32)
            expected[0, 0] = b + 1
            expected[1, 2:4] = torch.tensor([b + 2, b + 3])
            expected[2, 1] = b + 4
            expected[3, 0] = b + 5
            torch.testing.assert_close(t.gpu.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("uva_target", [False, True])
def test_staged_write_snapshots_pinned_source_before_return(uva_target):
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    state = StagedWriteTensor(
        (1, 4),
        torch.int32,
        torch.device("cuda"),
        uva_instead_of_gpu=uva_target,
    )
    source = torch.tensor([1, 2, 3, 4], dtype=torch.int32).pin_memory()
    state.stage_write(0, 0, source.numpy())
    # Keep H2D pending while the caller reuses its pinned storage after apply.
    torch.cuda._sleep(10_000_000)
    state.apply_write()
    source.fill_(99)
    torch.accelerator.synchronize()
    assert state.gpu[0].tolist() == [1, 2, 3, 4]
