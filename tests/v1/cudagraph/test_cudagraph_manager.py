# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.device_communicators import pynccl_allocator
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_graph_pool_id():
    pynccl_allocator._graph_pool_id = None
    yield
    pynccl_allocator._graph_pool_id = None


def _create_vllm_config() -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=[4],
    )
    compilation_config.max_cudagraph_capture_size = 4
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=4)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0
    return vllm_config


def test_full_capture_sets_graph_pool_id_before_cuda_graph(monkeypatch):
    """FULL capture must set graph_pool_id before entering torch.cuda.graph().

    NCCL symmetric memory checks this global during graph capture; without
    it, capture fails with:
    AssertionError: graph_pool_id is not set under graph capture
    """
    graph_pool = object()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: graph_pool,
    )

    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config(),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=1,
    )

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    manager._capture_descs[CUDAGraphMode.FULL] = [desc]

    def create_forward_fn(desc, warmup):
        return lambda _mode: None

    @contextmanager
    def fake_graph_capture(*args, **kwargs):
        yield SimpleNamespace(stream=MagicMock())

    fake_offloader = MagicMock()

    def cuda_graph_enter(*args, **kwargs):
        assert pynccl_allocator._graph_pool_id is graph_pool

    mock_cuda_graph_ctx = MagicMock()
    mock_cuda_graph_ctx.__enter__ = cuda_graph_enter
    mock_cuda_graph_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch.object(gpu_cudagraph_utils, "graph_capture", fake_graph_capture),
        patch.object(gpu_cudagraph_utils, "get_offloader", lambda: fake_offloader),
        patch.object(gpu_cudagraph_utils.torch.cuda, "CUDAGraph"),
        patch.object(
            gpu_cudagraph_utils.torch.cuda,
            "graph",
            return_value=mock_cuda_graph_ctx,
        ) as mock_cuda_graph,
    ):
        manager.capture(create_forward_fn)

    mock_cuda_graph.assert_called_once()


@pytest.mark.parametrize("first_rank", [False, True])
@pytest.mark.parametrize("raw_tokens", [False, True])
@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE])
def test_pipeline_capture_preserves_required_token_ids(
    monkeypatch, first_rank, raw_tokens, mode
):
    """Capture must use the same token-ID contract as eager pipeline execution."""
    from contextlib import nullcontext

    from vllm.sequence import IntermediateTensors

    class ForwardReached(Exception):
        pass

    ids = torch.tensor([101, 102, 103, 104])
    intermediates = IntermediateTensors({"hidden_states": torch.ones(4, 2)})
    buffers = SimpleNamespace(
        input_ids=ids,
        positions=torch.arange(4),
        is_padding=torch.zeros(4, dtype=torch.bool),
    )

    def forward(**kwargs):
        if first_rank or raw_tokens:
            torch.testing.assert_close(kwargs["input_ids"], ids[:3])
            assert kwargs["input_ids"].data_ptr() == ids.data_ptr()
        else:
            assert kwargs["input_ids"] is None
        if not first_rank:
            assert kwargs["inputs_embeds"] is None
            torch.testing.assert_close(
                kwargs["intermediate_tensors"]["hidden_states"],
                intermediates["hidden_states"][:3],
            )
        raise ForwardReached

    model = MagicMock(side_effect=forward)
    model.requires_raw_input_tokens = raw_tokens
    manager = gpu_cudagraph_utils.ModelCudaGraphManager.__new__(
        gpu_cudagraph_utils.ModelCudaGraphManager
    )
    manager.is_first_pp_rank = first_rank
    manager.use_breakable_cg = False
    manager.max_num_reqs = 4
    manager.dp_size = 1
    manager.vllm_config = None
    manager.run_pw_graph = lambda model, inputs: model(**inputs)
    state = SimpleNamespace(prepare_dummy_inputs=lambda *a: {})
    monkeypatch.setattr(
        gpu_cudagraph_utils, "prepare_inputs_to_capture", lambda *a, **k: (None, None)
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils, "set_forward_context", lambda *a, **k: nullcontext()
    )

    def capture(self, create_forward_fn, *args):
        desc = BatchExecutionDescriptor(mode, num_tokens=3, num_reqs=1)
        create_forward_fn(desc, warmup=False)(mode)

    monkeypatch.setattr(gpu_cudagraph_utils.CudaGraphManager, "capture", capture)
    with pytest.raises(ForwardReached):
        manager.capture(model, state, buffers, intermediates, None, [], None)
