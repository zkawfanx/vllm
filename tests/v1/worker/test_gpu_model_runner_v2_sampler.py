# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_new_reqs", [0, 1, 2])
@pytest.mark.parametrize("has_sampler", [False, True])
def test_sampler_flushes_only_on_admission(num_new_reqs, has_sampler):
    runner = GPUModelRunner.__new__(GPUModelRunner)
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
