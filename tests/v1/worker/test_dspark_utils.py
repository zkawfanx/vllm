# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import pytest

from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_dspark_parallel_config


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    decode_context_parallel_size: int = 2
    enable_expert_parallel: bool = True
    enable_eplb: bool = False
    eplb_config: _FakeEPLBConfig = field(default_factory=_FakeEPLBConfig)
    enable_elastic_ep: bool = False

    def __post_init__(self):
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


@pytest.mark.cpu_test
@pytest.mark.parametrize("enable_eplb", [False, True])
def test_dspark_parallel_config_isolates_draft_without_mutating_target(enable_eplb):
    """Disable interdependent EPLB fields atomically while retaining EP/DCP."""
    redundant = 32 if enable_eplb else 0
    target = _FakeParallelConfig(
        enable_eplb=enable_eplb,
        eplb_config=_FakeEPLBConfig(redundant),
        enable_elastic_ep=enable_eplb,
    )
    draft = _get_dspark_parallel_config(target, tensor_parallel_size=4)

    assert (target.pipeline_parallel_size, target.tensor_parallel_size) == (2, 8)
    assert target.enable_eplb == enable_eplb
    assert target.eplb_config.num_redundant_experts == redundant
    assert target.enable_elastic_ep == enable_eplb
    assert draft is not target
    assert draft.eplb_config is not target.eplb_config
    assert (draft.pipeline_parallel_size, draft.tensor_parallel_size) == (1, 4)
    assert not draft.enable_eplb
    assert draft.eplb_config.num_redundant_experts == 0
    assert not draft.enable_elastic_ep
    assert draft.enable_expert_parallel
    assert draft.decode_context_parallel_size == 2
