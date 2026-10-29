# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


@pytest.mark.cpu_test
def test_prefill_sampling_keys_follow_last_target_positions(monkeypatch):
    """Request compaction retains predecessor positions for the first draft."""
    speculator = _make_speculator(monkeypatch, torch.ones(4, 3))
    speculator.input_buffers.positions = torch.tensor([3, 4, 8, 9])
    speculator.last_token_indices = torch.tensor([1, 3])
    speculator.sample_src_positions = torch.zeros(2, dtype=torch.int64)
    speculator.idx_mapping = torch.tensor([1, 0])
    speculator.draft_tokens = torch.zeros(2, 3, dtype=torch.int64)
    speculator.temperature = speculator.seeds = None
    speculator.current_draft_step = torch.tensor(0)
    speculator.draft_logits = None
    speculator.sample_draft = lambda hidden, positions, *args: positions

    speculator._prefill(2, 4, None, None, None)

    assert speculator.draft_tokens[:, 0].tolist() == [5, 10]
    assert speculator.sample_src_positions.tolist() == [5, 10]
    assert speculator.input_buffers.positions[:2].tolist() == [4, 9]


@pytest.mark.cpu_test
def test_decode_sampling_uses_unclamped_sampling_positions(monkeypatch):
    """Clamped model positions must not reuse a previous draft's random draw."""
    speculator = _make_speculator(monkeypatch, torch.ones(4, 3))
    speculator.input_buffers.positions = torch.tensor([7, 7, 0, 0])
    speculator.sample_src_positions = torch.tensor([9, 11])
    speculator.idx_mapping = torch.tensor([1, 0])
    speculator.temperature = speculator.seeds = None
    speculator.current_draft_step = torch.tensor(1)
    speculator.draft_logits = None
    speculator.draft_tokens = torch.zeros(2, 3, dtype=torch.int64)
    speculator.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=8)
    )
    speculator.num_speculative_steps = 3
    speculator._prepare_eplb_forward = lambda *args: None
    speculator.sample_draft = lambda hidden, positions, *args: positions
    sampled = []
    monkeypatch.setattr(
        spec_module,
        "update_draft_inputs",
        lambda tokens, *args, **kwargs: sampled.append(tokens.clone()),
    )

    speculator._generate_draft(2, 2, None, None, None)

    assert sampled[0].tolist() == [9, 11]


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not torch.accelerator.is_available(),
    reason="Requires CUDA or ROCm",
)
@pytest.mark.parametrize("advance_draft_positions", [False, True])
def test_sampling_keys_advance_when_forward_positions_are_fixed_or_clamped(
    advance_draft_positions,
):
    device = "cuda"
    buffers = SimpleNamespace(
        input_ids=torch.zeros(2, dtype=torch.int32, device=device),
        positions=torch.tensor([6, 7], device=device),
        query_start_loc=torch.zeros(3, dtype=torch.int32, device=device),
        seq_lens=torch.zeros(2, dtype=torch.int32, device=device),
    )
    sample_positions = torch.tensor([7, 8], device=device)
    draft_tokens = torch.tensor([10, 11], device=device)
    spec_module.prepare_decode_inputs(
        draft_tokens,
        torch.tensor([7, 8], device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
        buffers,
        sample_positions,
        max_model_len=8,
        max_num_reqs=2,
        advance_draft_positions=advance_draft_positions,
    )
    assert sample_positions.tolist() == [8, 9]
    expected_forward_positions = [7, 7] if advance_draft_positions else [6, 7]
    assert buffers.positions.tolist() == expected_forward_positions

    output = torch.zeros(2, 3, dtype=torch.int64, device=device)
    hidden_states = torch.ones(2, 3, device=device)
    for step in (1, 2):
        spec_module.update_draft_inputs(
            draft_tokens,
            torch.tensor(step, device=device),
            hidden_states,
            output,
            torch.zeros_like(hidden_states),
            buffers,
            sample_positions,
            num_reqs=2,
            max_model_len=8,
            num_speculative_steps=3,
            advance_draft_positions=advance_draft_positions,
        )
        # The final step writes the output but does not advance sampling again.
        assert sample_positions.tolist() == [9, 10]
        assert buffers.positions.tolist() == expected_forward_positions
        assert output[:, step].tolist() == [10, 11]
