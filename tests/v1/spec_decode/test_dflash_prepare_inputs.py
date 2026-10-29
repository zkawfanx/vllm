# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires GPU"
)


def _prepare(query_len, sample_from_anchor, num_reqs=3):
    device = current_platform.device_type
    max_reqs, max_tokens, steps = 8, 16384, 7
    query_per_req = steps if sample_from_anchor else steps + 1

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    def output(size):
        return torch.full((size,), -99, dtype=torch.int64, device=device)

    lengths = [query_len] + [1] * (num_reqs - 1)
    offsets = np.cumsum([0] + lengths)
    positions = torch.cat([torch.arange(n, device=device) for n in lengths])
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.array(lengths),
        positions=positions,
        query_start_loc=tensor(offsets),
        idx_mapping=tensor(list(range(num_reqs))),
    )
    buffers = InputBuffers(max_reqs, max_tokens, device)
    for value in vars(buffers).values():
        if isinstance(value, torch.Tensor):
            value.fill_(1)
    slots, ctx_pos, ctx_slots = (
        output(max_tokens),
        output(len(positions)),
        output(len(positions)),
    )
    samples = [output(max_reqs * steps) for _ in range(3)]
    temperature = torch.full((max_reqs,), -99.0, device=device)
    seeds = output(max_reqs)
    table = torch.arange(1, 1025, device=device).repeat(num_reqs, 1)
    prepare_dflash_inputs(
        buffers,
        slots,
        ctx_pos,
        ctx_slots,
        *samples,
        temperature,
        seeds,
        batch,
        tensor([1] * num_reqs),
        tensor([0] * num_reqs),
        tensor([42] * max_reqs),
        tensor([43] * max_reqs),
        torch.ones(max_reqs, device=device),
        tensor([17] * max_reqs),
        table,
        16,
        99,
        query_per_req,
        steps,
        max_reqs,
        max_tokens,
        16384,
        sample_from_anchor,
    )
    active = num_reqs * query_per_req
    expected_pos = torch.cat(
        [torch.arange(n, n + query_per_req, device=device) for n in lengths]
    )
    torch.testing.assert_close(buffers.positions[:active], expected_pos)
    torch.testing.assert_close(ctx_pos, positions)
    torch.testing.assert_close(ctx_slots, positions + 16)
    torch.testing.assert_close(slots[:active], expected_pos + 16)
    assert buffers.input_ids[:active].reshape(num_reqs, -1)[:, 0].eq(42).all()
    assert buffers.input_ids[:active].reshape(num_reqs, -1)[:, 1:].eq(99).all()
    assert not buffers.is_padding[:active].any()
    assert buffers.is_padding[active:].all()
    assert buffers.input_ids[active:].eq(0).all()
    assert buffers.positions[active:].eq(0).all()
    assert slots[active:].eq(PAD_SLOT_ID).all()
    assert buffers.query_start_loc[:num_reqs].tolist() == [
        i * query_per_req for i in range(num_reqs)
    ]
    assert buffers.seq_lens[:num_reqs].tolist() == [n + query_per_req for n in lengths]
    assert buffers.query_start_loc[num_reqs:].eq(active).all()
    assert buffers.seq_lens[num_reqs:].eq(0).all()
    sample_start = 0 if sample_from_anchor else 1
    expected_indices = torch.arange(active, device=device).reshape(num_reqs, -1)
    torch.testing.assert_close(
        samples[0][: num_reqs * steps], expected_indices[:, sample_start:].flatten()
    )
    expected_sample_pos = expected_pos.reshape(num_reqs, -1)[:, sample_start:]
    if sample_from_anchor:
        expected_sample_pos = expected_sample_pos + 1
    torch.testing.assert_close(
        samples[1][: num_reqs * steps], expected_sample_pos.flatten()
    )
    assert samples[2][: num_reqs * steps].tolist() == [
        i for i in range(num_reqs) for _ in range(steps)
    ]
    for value, expected in zip(samples, (0, 0, -1)):
        assert value[num_reqs * steps :].eq(expected).all()


@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("query_len", [1, 121, 128, 249, 300, 8192])
def test_prepare_inputs_covers_tile_boundaries_and_long_padding(
    query_len, sample_from_anchor
):
    _prepare(query_len, sample_from_anchor)


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_prepare_inputs_keeps_launch_tile_independent_of_query_length(
    monkeypatch, sample_from_anchor
):
    kernel = speculator._prepare_dflash_inputs_kernel
    launches = []

    class RecordLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs["BLOCK_SIZE"]))
                return kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(speculator, "_prepare_dflash_inputs_kernel", RecordLaunch())
    query_per_req = 7 if sample_from_anchor else 8
    for query_len in (1, 121, 128, 249, 300, 8192):
        _prepare(query_len, sample_from_anchor, num_reqs=1)
        assert launches[-1] == ((1, (query_len + query_per_req + 127) // 128), 128)
