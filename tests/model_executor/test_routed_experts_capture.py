# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    ForwardContext,
    get_forward_context,
    set_forward_context,
)
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _forward_context(**kwargs):
    return ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={}, **kwargs
    )


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class _DummyRouter:
        _routing_replay_out: torch.Tensor | None = None

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self._quant_method = self.routed_experts.quant_method

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module])
    )

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self._quant_method = self.routed_experts.quant_method

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module])
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


def test_gpu_model_runner_does_not_bind_draft_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self.routed_experts = _make_modular_routed_experts()
            self._quant_method = self.routed_experts.quant_method

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [target_module]),
        compilation_config=types.SimpleNamespace(
            static_forward_context={
                "model.layers.7.mlp.experts": target_module,
                "mtp.layers.0.mlp.experts": draft_module,
            }
        ),
    )

    capturer = types.SimpleNamespace(capture=lambda *_: None)
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None


def test_gpu_model_runner_rejects_monolithic_without_replay_support(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            # Use a concrete monolithic expert and override its capability
            # instead of instantiating the abstract base class directly.
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsFp8,
            )

            fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
            self.routed_experts = types.SimpleNamespace(
                quant_method=types.SimpleNamespace(
                    is_monolithic=True,
                    moe_kernel=types.SimpleNamespace(
                        impl=types.SimpleNamespace(fused_experts=fused_experts)
                    ),
                )
            )
            self._quant_method = self.routed_experts.quant_method
            self._quant_method.moe_kernel.impl.fused_experts = fused_experts
            fused_experts.supports_routing_replay_capture = lambda: False

    class DummyCapturer:
        def capture(self, layer_id, topk_ids):
            pass

    dummy_module = DummyFusedMoE()
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module])
    )

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, DummyCapturer())


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = _forward_context()
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=DPMetadata(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=DPMetadata(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=DPMetadata(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


@pytest.mark.parametrize("tp_size, num_tokens", [(2, 6), (2, 3), (2, 1), (4, 2)])
@pytest.mark.parametrize(
    "cg_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE]
)
def test_single_dp_ep_capture_restores_tp_shards(tp_size, num_tokens, cg_mode):
    """Capture all current rows after model-level SP, including buffer reuse."""
    capturer = _capturer_with_buffer(tp_size=tp_size)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=tp_size,
            enable_expert_parallel=True,
            use_sequence_parallel_moe=False,
            is_moe_model=True,
        ),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        ),
    )
    tp_group = Mock()
    shard_n = (num_tokens + tp_size - 1) // tp_size
    with (
        set_forward_context(
            None,
            config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cg_mode,
        ),
        patch(f"{_REC_MODULE}.get_tp_group", return_value=tp_group),
    ):
        assert get_forward_context().dp_metadata is None
        for offset in (20, 1):
            expected = torch.arange(num_tokens * 2, dtype=torch.int32).reshape(-1, 2)
            expected += offset
            padded = torch.full((shard_n * tp_size, 2), 99, dtype=torch.int32)
            padded[:num_tokens] = expected
            tp_group.all_gather.return_value = padded
            capturer.capture(0, padded[:shard_n])
            assert torch.equal(capturer.device_buffer[:num_tokens, 0], expected)
            assert (capturer.device_buffer[num_tokens:, 0] == -1).all()
    if num_tokens == 1:
        tp_group.all_gather.assert_not_called()
    else:
        assert tp_group.all_gather.call_count == 2


@pytest.mark.parametrize("num_tokens, num_rows", [(6, 6), (3, 4), (0, 0)])
def test_single_dp_tp_capture_does_not_regather_full_batch(num_tokens, num_rows):
    capturer = _capturer_with_buffer(tp_size=2)
    topk = torch.arange(num_rows * 2, dtype=torch.int32).reshape(-1, 2)
    ctx = _forward_context(batch_descriptor=BatchDescriptor(num_tokens=num_tokens))
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        patch(f"{_REC_MODULE}.get_tp_group") as get_tp_group,
    ):
        capturer.capture(0, topk)
    get_tp_group.assert_not_called()
    assert torch.equal(capturer.device_buffer[:num_tokens, 0], topk[:num_tokens])
    assert (capturer.device_buffer[num_tokens:, 0] == -1).all()


@pytest.mark.parametrize("num_tokens, num_rows", [(6, 2), (0, 1)])
def test_single_dp_tp_capture_rejects_unexpected_batch_without_writing(
    num_tokens, num_rows
):
    capturer = _capturer_with_buffer(tp_size=2)
    topk = torch.ones(num_rows, 2, dtype=torch.int32)
    ctx = _forward_context(batch_descriptor=BatchDescriptor(num_tokens=num_tokens))
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(0, topk)
    assert (capturer.device_buffer == -1).all()


def test_routed_experts_capturer_dp_ep_gathered_shards():
    capturer = _capturer_with_buffer(dp_rank=1, tp_size=2)
    num_tokens_dp = torch.tensor([3, 2], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=DPMetadata(
            num_tokens_across_dp_cpu=num_tokens_dp,
            local_sizes=[2, 2, 1, 1],
        )
    )
    topk = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [-1, -1], [10, 11], [12, 13]],
        dtype=torch.int32,
    )

    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)

    assert torch.equal(capturer.device_buffer[:2, 0], topk[4:])


@pytest.mark.parametrize("single_dp", [False, True])
def test_routed_experts_capturer_sp_modular_gathers_tp_shards(single_dp):
    capturer = _capturer_with_buffer(dp_rank=0 if single_dp else 1, tp_size=2)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=(
            None if single_dp else DPMetadata(num_tokens_across_dp_cpu=num_tokens_dp)
        ),
        batch_descriptor=BatchDescriptor(num_tokens=3),
    )
    local_shard = torch.tensor([[10, 12], [11, 13]], dtype=torch.int32).t()
    gathered = torch.tensor([[10, 11], [12, 13], [14, 15], [-1, -1]], dtype=torch.int32)
    tp_group = Mock()
    tp_group.all_gather.return_value = gathered

    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        patch(f"{_REC_MODULE}.get_tp_group", return_value=tp_group),
    ):
        capturer.capture(layer_id=0, topk_ids=local_shard)

    tp_group.all_gather.assert_called_once()
    gather_input = tp_group.all_gather.call_args.args[0]
    assert gather_input.is_contiguous()
    torch.testing.assert_close(gather_input, local_shard)
    assert tp_group.all_gather.call_args.kwargs == {"dim": 0}
    assert torch.equal(capturer.device_buffer[:3, 0], gathered[:3])


def test_routed_experts_capturer_rejects_incomplete_dp_shard_groups():
    """Invalid gather metadata must not silently select another rank's rows."""
    capturer = _capturer_with_buffer(dp_rank=1, tp_size=2)
    ctx = _forward_context(
        dp_metadata=DPMetadata(
            num_tokens_across_dp_cpu=torch.tensor([3, 2]),
            local_sizes=[2, 2, 2],
        )
    )
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(ValueError, match="not divisible by DP rank count"),
    ):
        capturer.capture(0, torch.ones(6, 2, dtype=torch.int32))
    assert (capturer.device_buffer == -1).all()


@pytest.mark.parametrize(
    ("counts", "local_sizes", "start", "stop"),
    [
        ([3, 0], [2, 2, 0, 0], 3, 3),
        ([0, 3], None, 0, 3),
        ([0, 0], [0, 0, 0, 0], 0, 0),
        ([2, 2], [1, 1, 1, 1], 2, 4),
    ],
    ids=["idle-rank", "only-active-rank", "all-idle", "unpadded-gather-tie"],
)
def test_routed_experts_capturer_dp_preserves_empty_and_unpadded_spans(
    counts, local_sizes, start, stop
):
    capturer = _capturer_with_buffer(dp_rank=1, tp_size=2)
    topk = torch.arange(sum(counts) * 2, dtype=torch.int32).reshape(-1, 2)
    ctx = _forward_context(
        dp_metadata=DPMetadata(
            num_tokens_across_dp_cpu=torch.tensor(counts), local_sizes=local_sizes
        )
    )
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        patch(f"{_REC_MODULE}.get_tp_group") as get_tp_group,
    ):
        capturer.capture(0, topk)
    get_tp_group.assert_not_called()
    torch.testing.assert_close(
        capturer.device_buffer[: stop - start, 0], topk[start:stop]
    )
    assert (capturer.device_buffer[stop - start :, 0] == -1).all()


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_capture_uses_pre_sp_size_after_model_shards_padding_mask(tp_rank):
    """Model-level SP shortens the mask; capture still needs all three rows."""
    from vllm.models.common.ops import sequence_parallel as sp

    capturer = _capturer_with_buffer(tp_size=2)
    expected = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    ctx = _forward_context(
        is_padding=torch.zeros(3, dtype=torch.bool),
        batch_descriptor=BatchDescriptor(num_tokens=3),
    )
    with (
        patch.object(sp, "get_tensor_model_parallel_world_size", return_value=2),
        patch.object(sp, "get_tensor_model_parallel_rank", return_value=tp_rank),
    ):
        ctx.is_padding = sp.sp_padding_mask(ctx.is_padding, torch.empty(3, 1))
    assert len(ctx.is_padding) == 2
    tp_group = Mock()
    tp_group.all_gather.return_value = expected
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        patch(f"{_REC_MODULE}.get_tp_group", return_value=tp_group),
    ):
        capturer.capture(0, expected[2 * tp_rank : 2 * (tp_rank + 1)])
    tp_group.all_gather.assert_called_once()
    torch.testing.assert_close(capturer.device_buffer[:3, 0], expected[:3])
    assert (capturer.device_buffer[3:, 0] == -1).all()


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_routed_experts_capturer_microbatches_preserve_step_rows(
    tp_size, reverse_order
):
    """DBO contexts must retain destination offsets across TP reconstruction."""
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    wrapper = UBatchWrapper.__new__(UBatchWrapper)
    wrapper.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        )
    )
    wrapper.comm_stream = wrapper.ready_barrier = None
    slices = [
        UBatchSlice(slice(0, 1), slice(0, 3)),
        UBatchSlice(slice(1, 2), slice(3, 5)),
    ]
    dp_metadata = [
        DPMetadata(num_tokens_across_dp_cpu=torch.tensor([n, n])) for n in (3, 2)
    ]
    with patch(
        "vllm.v1.worker.gpu_ubatch_wrapper.make_ubatch_contexts",
        side_effect=lambda **kwargs: kwargs["forward_contexts"],
    ):
        metadata = wrapper._make_ubatch_metadata(
            slices,
            None,
            None,
            torch.arange(5),
            torch.arange(5),
            None,
            None,
            None,
            dp_metadata,
            BatchDescriptor(num_tokens=5),
            CUDAGraphMode.NONE,
        )

    capturer = _capturer_with_buffer(dp_rank=1, tp_size=tp_size)
    expected = torch.arange(10, dtype=torch.int32).view(5, 2)
    order = [1, 0] if reverse_order else [0, 1]
    for i in order:
        rows = expected[slices[i].token_slice]
        padded = torch.cat([rows, torch.full((len(rows) % tp_size, 2), -9)])
        with (
            patch(
                f"{_REC_MODULE}.get_forward_context", return_value=metadata[i].context
            ),
            patch(f"{_REC_MODULE}.get_tp_group") as tp_group,
        ):
            tp_group.return_value.all_gather.return_value = padded
            capturer.capture(0, padded[: len(padded) // tp_size])
    torch.testing.assert_close(capturer.device_buffer[:5, 0], expected)
    assert (capturer.device_buffer[5:] == -1).all()
    assert (capturer.device_buffer[:, 1:] == -1).all()
