# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu import model_runner as mrv2


@pytest.mark.parametrize("is_cpu", [False, True])
def test_dummy_sampler_profiles_warmup_params_after_default_pass(monkeypatch, is_cpu):
    """GPU profiling must flush warmup params before accounting for their peak."""
    logits = torch.zeros(3, 64)
    batch = object()
    sampler = Mock()
    runner = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda _: logits),
        input_buffers=object(),
        sampler=sampler,
    )
    monkeypatch.setattr(mrv2.InputBatch, "make_dummy", lambda *args: batch)
    monkeypatch.setattr(mrv2.current_platform, "is_cpu", lambda: is_cpu)

    mrv2.GPUModelRunner._dummy_sampler_run(runner, torch.zeros(3, 4))

    expected = [call(logits, batch)]
    if not is_cpu:
        params = SamplingParams.for_sampler_warmup()
        # Dummy batches have no prefilled tokens; retain the target sampler's
        # prompt-length contract when setting up minimum-token processing.
        expected.extend(call.add_request(i, 0, params) for i in range(3))
        expected.extend([call.apply_staged_writes(), call(logits, batch)])
    assert sampler.mock_calls == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_new_reqs", [0, 1, 2])
@pytest.mark.parametrize("has_sampler", [False, True])
def test_sampler_flushes_only_on_admission(num_new_reqs, has_sampler):
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner._remove_request = Mock()
    runner.req_states = Mock(req_id_to_index={str(i): i for i in range(2)})
    runner.model_state = Mock()
    runner.block_tables = Mock()
    runner.lora_state = Mock()
    runner.pooling_runner = None
    runner.encoder_cache = None
    runner.is_last_pp_rank = has_sampler
    runner.sampler = Mock() if has_sampler else None
    runner.prompt_logprobs_worker = Mock()
    flushes = Mock()
    flushes.attach_mock(runner.req_states.apply_staged_writes, "requests")
    flushes.attach_mock(runner.model_state.apply_staged_writes, "model")
    if has_sampler:
        flushes.attach_mock(runner.sampler.apply_staged_writes, "sampler")

    requests = [
        SimpleNamespace(
            req_id=str(i),
            prompt_token_ids=[1],
            prefill_token_ids=[1],
            num_computed_tokens=0,
            sampling_params=SimpleNamespace(max_tokens=8),
            block_ids=([i],),
            lora_request=None,
        )
        for i in range(num_new_reqs)
    ]
    # Re-admission must flush reused slots; intervening decode must not copy them.
    for new_requests in (requests, [], requests):
        flushes.reset_mock()
        if has_sampler:
            runner.sampler.add_request.reset_mock()
        runner.add_requests(SimpleNamespace(scheduled_new_reqs=new_requests))
        expected = []
        if new_requests:
            expected = [call.requests(), call.model()]
            if has_sampler:
                expected.append(call.sampler())
        assert flushes.mock_calls == expected
        if has_sampler:
            assert runner.sampler.add_request.call_args_list == [
                call(i, 1, request.sampling_params)
                for i, request in enumerate(new_requests)
            ]
