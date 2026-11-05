# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

pytestmark = pytest.mark.cpu_test


def _speculator(monkeypatch):
    spec = object.__new__(DFlashSpeculator)
    spec.num_query_per_req = 5
    spec.num_speculative_steps = 5
    spec.max_model_len = 128
    spec.max_num_reqs = 2
    spec.max_num_tokens = 16
    spec.draft_kv_cache_group_id = 0
    spec.draft_kv_cache_group_ids = [0]
    spec._layer_group_idx = None
    spec._aux_staging = None
    spec._context_slot_mappings = torch.zeros(1, 16, dtype=torch.int64)
    spec.hidden_states = torch.zeros(16, 2)
    spec.context_positions = torch.zeros(16, dtype=torch.int64)
    spec.block_tables = SimpleNamespace(
        slot_mappings=torch.zeros(1, 16, dtype=torch.int64),
        input_block_tables=[None],
        kernel_block_sizes=[16],
    )
    for name in (
        "input_buffers",
        "sample_indices",
        "sample_pos",
        "sample_idx_mapping",
        "temperature",
        "seeds",
    ):
        setattr(spec, name, None)
    spec.parallel_drafting_token_id = 0
    spec.sample_from_anchor = True
    spec.draft_tokens = torch.full((2, 5), 99)
    spec.model = Mock()
    spec.query_cudagraph_manager = Mock()
    spec.dp_size = 1
    spec.dp_rank = 0
    spec._group_causal = None
    spec.kv_cache_config = None
    spec._build_draft_attn_metadata = Mock(return_value={})
    spec._prepare_eplb_forward = Mock()
    spec._prepare_graph_replay = Mock()
    spec._generate_draft = Mock()
    monkeypatch.setattr(
        spec_module, "build_slot_mappings_by_layer", Mock(return_value={})
    )
    return spec


def _propose(spec, k):
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=3,
        seq_lens_cpu_upper_bound=torch.tensor([3]),
    )
    hidden = torch.ones(3, 2)
    return spec.propose(
        batch,
        {},
        {},
        hidden,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        num_speculative_tokens=k,
    )


def test_zero_k_refreshes_context_slots_without_query_or_graph(monkeypatch):
    spec = _speculator(monkeypatch)
    slots = [4, 8]

    def prepare(*args):
        # Real preparation rebuilds mappings when requests reuse batch slots.
        args[3][:3].fill_(slots.pop(0))

    monkeypatch.setattr(spec_module, "prepare_dflash_inputs", prepare)
    dispatch = Mock(side_effect=AssertionError("query dispatch must be skipped"))
    monkeypatch.setattr(spec_module, "dispatch_cg_and_sync_dp", dispatch)
    for slot in (4, 8):
        result = _propose(spec, 0)
        assert result.shape == (1, 5)  # persistent fixed-width buffer
        args = spec.model.precompute_and_store_context_kv.call_args.args
        torch.testing.assert_close(args[0], torch.ones(3, 2))
        assert args[2].tolist() == [slot] * 3
    spec._generate_draft.assert_not_called()
    spec.query_cudagraph_manager.run_fullgraph.assert_not_called()


@pytest.mark.parametrize("k", [None, 5])
def test_fixed_k_keeps_full_graph_replay(monkeypatch, k):
    spec = _speculator(monkeypatch)
    monkeypatch.setattr(spec_module, "prepare_dflash_inputs", Mock())
    desc = SimpleNamespace(num_reqs=1, num_tokens=5, cg_mode=CUDAGraphMode.FULL)
    dispatch = Mock(return_value=(desc, None))
    monkeypatch.setattr(spec_module, "dispatch_cg_and_sync_dp", dispatch)
    result = _propose(spec, k)
    assert result.shape == (1, 5)
    spec._prepare_graph_replay.assert_called_once()
    spec.query_cudagraph_manager.run_fullgraph.assert_called_once_with(desc)
    spec.model.precompute_and_store_context_kv.assert_not_called()
    assert dispatch.call_args.kwargs["uniform_token_count"] == 5


@pytest.mark.parametrize("skip_drafts", [True, False])
def test_runner_publishes_only_current_drafts_for_reused_slot(monkeypatch, skip_drafts):
    from vllm.v1.worker.gpu import model_runner as mrv2

    runner = object.__new__(mrv2.GPUModelRunner)
    batch = SimpleNamespace(
        req_ids=["new-request"], idx_mapping=torch.tensor([2]), query_start_loc=None
    )
    runner.execute_model_state = mrv2.ExecuteModelState(
        input_batch=batch,
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=torch.ones(1, 2),
        aux_hidden_states=None,
        finished_req_ids=set(),
        skip_drafts=skip_drafts,
    )
    runner.num_speculative_steps = 5
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    runner.pp_handler = None
    runner.eplb = Mock()
    runner.model = SimpleNamespace(compute_logits=Mock())
    runner.prompt_logprobs_worker = Mock()
    runner.prompt_logprobs_worker.compute_prompt_logprobs.return_value = {}
    runner.sample = Mock(
        return_value=(
            SimpleNamespace(sampled_token_ids=torch.tensor([[1]])),
            torch.tensor([1]),
            torch.tensor([5]),
        )
    )
    runner.req_states = SimpleNamespace(
        draft_tokens=torch.full((4, 5), 77),
        last_sampled_tokens=None,
        next_prefill_tokens=None,
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
    )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=None),
            seeds=SimpleNamespace(gpu=None),
        )
    )
    runner.speculator = Mock(supports_mm_inputs=False)
    runner.speculator.propose.return_value = torch.full((1, 5), 99)
    runner.draft_tokens_handler = Mock()
    runner.postprocess_sampled = Mock()
    runner.kv_connector = Mock()
    runner.main_stream = runner.output_copy_stream = None
    runner.check_ep_fault = Mock()
    monkeypatch.setattr(mrv2, "AsyncOutput", Mock())
    monkeypatch.setattr(
        mrv2.pcp,
        "maybe_restore_pcp_for_sampling",
        lambda manager, hidden, input_batch: (hidden, input_batch),
    )

    runner.sample_tokens(None)

    expected_k = 0 if skip_drafts else 5
    assert (
        runner.speculator.propose.call_args.kwargs["num_speculative_tokens"]
        == expected_k
    )
    assert (
        runner.req_states.draft_tokens[2].tolist() == [(-1 if skip_drafts else 99)] * 5
    )
    assert runner.req_states.draft_tokens[1].tolist() == [77] * 5
    published = runner.draft_tokens_handler.set_draft_tokens.call_args.args[1]
    assert published.shape == (1, expected_k)
    if not skip_drafts:
        assert published.tolist() == [[99] * 5]
    assert runner.execute_model_state is None
