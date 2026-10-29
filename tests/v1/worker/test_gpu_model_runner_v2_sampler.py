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
