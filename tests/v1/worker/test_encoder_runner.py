# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for model runner V2 encoder routing, capture, and embedding gathering.

Covers the speculative-drafter encoder-cache handling: the drafter reads one
position ahead of the target model (``draft_lookahead``). The +1 look-ahead
feature past the processed boundary is used when its encoder output is present
and tolerated (token-embedding fallback) when it is not, while a miss within
the processed range still fails loudly.
"""

import weakref
from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

pytestmark = pytest.mark.cpu_test

HIDDEN = 4


def _embeds_item(embeds: torch.Tensor) -> MultiModalKwargsItem:
    """A `prompt_embeds` kwargs item, as the HF renderer builds it."""
    return MultiModalKwargsItem(
        {
            "embedding": MultiModalFieldElem(
                data=embeds, field=MultiModalSharedField(batch_size=1)
            )
        }
    )


def _feature(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _make_runner(
    features: list[MultiModalFeatureSpec],
    cached: list[MultiModalFeatureSpec],
) -> EncoderRunner:
    cache = EncoderCache()
    cache.mm_features["req0"] = features
    for f in cached:
        length = f.mm_position.length
        cache.encoder_outputs[f.identifier] = torch.arange(
            length * HIDDEN, dtype=torch.float32
        ).reshape(length, HIDDEN)
    return EncoderRunner(
        model=None,  # unused by gather_mm_embeddings
        max_num_tokens=64,
        hidden_size=HIDDEN,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _gather(runner: EncoderRunner, *, num_scheduled: int, draft_lookahead: int):
    # Single prefilling request, num_computed_tokens=0, prefill_len large.
    return runner.gather_mm_embeddings(
        req_ids=["req0"],
        total_num_scheduled_tokens=num_scheduled,
        num_scheduled_tokens=np.array([num_scheduled]),
        query_start_loc=np.array([0]),
        prefill_lens=np.array([1000]),
        num_computed_tokens=np.array([0]),
        draft_lookahead=draft_lookahead,
    )


def test_draft_lookahead_uses_boundary_feature_when_cached():
    """The drafter's +1 look-ahead can reach the feature at offset ==
    processed_end (the next chunk). When its encoder output is already cached
    (the scheduler encoded it ahead), it is used for the look-ahead position
    rather than ignored."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # starts exactly at processed_end
    runner = _make_runner([f0, f1], cached=[f0, f1])

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled=8, draft_lookahead=1)

    # f0 covers positions 0..6 (+1 skew); f1's first embed covers position 7.
    assert len(mm_embeds) == 2
    assert [e.modality for e in mm_embeds] == ["image", "image"]
    assert bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 8


def test_draft_lookahead_tolerates_missing_boundary_feature():
    """When the +1 look-ahead feature past the processed boundary is not yet
    encoded, fall back to the token embedding (the draft token is verified by
    the target) instead of raising."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # boundary feature, not cached
    runner = _make_runner([f0, f1], cached=[f0])

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled=8, draft_lookahead=1)

    # Only f0 is gathered; f1's boundary position falls back silently.
    assert len(mm_embeds) == 1
    assert [e.modality for e in mm_embeds] == ["image"]
    assert not bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 7


def test_draft_lookahead_raises_on_interior_miss():
    """A miss for a feature within the processed range (not the look-ahead
    boundary) is a real invariant violation and must fail loudly, even on the
    drafter path."""
    f0 = _feature("h0", offset=0, length=8)  # interior, within processed range
    runner = _make_runner([f0], cached=[])

    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather(runner, num_scheduled=8, draft_lookahead=1)


def test_target_path_raises_on_encoder_cache_miss():
    """On the target path (no look-ahead) a miss is a real invariant
    violation and must fail loudly."""
    f0 = _feature("h0", offset=0, length=8)
    runner = _make_runner([f0], cached=[])

    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather(runner, num_scheduled=8, draft_lookahead=0)


@pytest.mark.parametrize("draft_lookahead", [0, 1])
def test_multi_request_batch_gathers_per_request(draft_lookahead):
    """Two prefilling requests in one batch: per-request query bounds must be
    indexed by request, not applied as whole arrays."""
    a0 = _feature("a0", offset=0, length=8)
    b0 = _feature("b0", offset=0, length=8)
    cache = EncoderCache()
    cache.mm_features["req0"] = [a0]
    cache.mm_features["req1"] = [b0]
    for f in (a0, b0):
        cache.encoder_outputs[f.identifier] = torch.arange(
            f.mm_position.length * HIDDEN, dtype=torch.float32
        ).reshape(f.mm_position.length, HIDDEN)
    runner = EncoderRunner(
        model=None,
        max_num_tokens=64,
        hidden_size=HIDDEN,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    mm_embeds, is_mm_embed = runner.gather_mm_embeddings(
        req_ids=["req0", "req1"],
        total_num_scheduled_tokens=16,
        num_scheduled_tokens=np.array([8, 8]),
        query_start_loc=np.array([0, 8]),
        prefill_lens=np.array([1000, 1000]),
        num_computed_tokens=np.array([0, 0]),
        draft_lookahead=draft_lookahead,
    )

    # Both requests contribute a feature; with the +1 skew each marks 7 of its
    # 8 positions (the skew drops one), otherwise all 8.
    assert len(mm_embeds) == 2
    assert [e.modality for e in mm_embeds] == ["image", "image"]
    assert int(is_mm_embed.sum()) == (14 if draft_lookahead else 16)


def test_gather_preserves_mixed_modalities():
    """Modalities must be attached on tensors in gather order."""
    video = MultiModalFeatureSpec(
        data=None,
        modality="video",
        identifier="v0",
        mm_position=PlaceholderRange(offset=0, length=4),
    )
    audio = MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier="a0",
        mm_position=PlaceholderRange(offset=4, length=4),
    )
    runner = _make_runner([video, audio], cached=[video, audio])

    mm_embeds, is_mm_embed = _gather(runner, num_scheduled=8, draft_lookahead=0)

    assert len(mm_embeds) == 2
    assert [e.modality for e in mm_embeds] == ["video", "audio"]
    assert int(is_mm_embed.sum()) == 8


@pytest.fixture
def graph_runner(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.mm import encoder_runner as module

    events = []
    managers = []
    state = SimpleNamespace(fail_capture=False)

    @contextmanager
    def capture_context(**kwargs):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    class Manager:
        def __init__(self, **kwargs):
            managers.append(self)
            events.append("create")

        def capture(self, graph_pool):
            assert graph_pool == "encoder-pool"
            events.append("capture")
            if state.fail_capture:
                raise RuntimeError("capture failed")

        def clear(self):
            events.append("clear")

        def supports_modality(self, modality):
            return modality == "image"

        def execute(self, inputs):
            events.append("graph")
            return [inputs["value"].clone() + 20]

    def eager(**inputs):
        events.append("eager")
        return [inputs["value"].clone() + 10]

    monkeypatch.setattr(module, "EncoderCudaGraphManager", Manager)
    monkeypatch.setattr(
        module, "supports_encoder_cudagraph", lambda model: model.supported
    )
    monkeypatch.setattr(module, "graph_capture", capture_context)
    monkeypatch.setattr(
        module.current_platform, "graph_pool_handle", lambda: "encoder-pool"
    )
    monkeypatch.setattr(module.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: events.append("empty")
    )
    memory = iter([1000, 750])

    def get_memory():
        events.append("memory")
        return next(memory), 2000

    monkeypatch.setattr(torch.accelerator, "get_memory_info", get_memory)
    monkeypatch.setattr(module, "group_and_batch_mm_kwargs", lambda inputs, **_: inputs)

    def make(enabled=True, supported=True, enforce_eager=False):
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mm_encoder=enabled),
            model_config=SimpleNamespace(enforce_eager=enforce_eager),
        )
        return EncoderRunner(
            model=SimpleNamespace(supported=supported, embed_multimodal=eager),
            max_num_tokens=8,
            hidden_size=HIDDEN,
            encoder_cache=EncoderCache(),
            dtype=torch.float32,
            device=torch.device("cpu"),
            vllm_config=config,
        )

    return SimpleNamespace(make=make, events=events, managers=managers, state=state)


@pytest.mark.parametrize(
    "enabled,supported,enforce_eager",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_encoder_graph_disabled_or_unsupported_keeps_eager(
    graph_runner, enabled, supported, enforce_eager
):
    runner = graph_runner.make(enabled, supported, enforce_eager)
    assert runner.profile_cudagraph_memory() == 0
    runner.capture_model()
    value = torch.ones(2, HIDDEN)
    output = runner.execute_mm_encoder([("image", 1, {"value": value})])
    torch.testing.assert_close(output[0], value + 10)
    assert graph_runner.events == ["eager"]
    assert runner.encoder_cudagraph_manager is None


def test_encoder_graph_routes_only_after_capture_and_preserves_other_modalities(
    graph_runner,
):
    runner = graph_runner.make()
    value = torch.ones(2, HIDDEN)
    image = ("image", 1, {"value": value})
    runner.execute_mm_encoder([image])
    assert graph_runner.events == ["eager"]
    runner.capture_model()
    assert runner.encoder_cudagraph_manager is graph_runner.managers[0]
    runner.capture_model()
    assert len(graph_runner.managers) == 1
    graph_runner.events.clear()
    output = runner.execute_mm_encoder([image, ("audio", 1, {"value": value})])
    runner.encoder_cache.encoder_outputs["saved"] = output[0]
    runner.execute_mm_encoder([("image", 1, {"value": value * 3})])
    torch.testing.assert_close(
        runner.encoder_cache.encoder_outputs["saved"], value + 20
    )
    torch.testing.assert_close(output[1], value + 10)
    assert graph_runner.events == ["graph", "eager", "graph"]
    assert runner.execute_mm_encoder([]) == []
    assert graph_runner.events == ["graph", "eager", "graph"]


def test_encoder_graph_profile_uses_temporary_pool_and_keeps_serving_eager(
    graph_runner,
):
    runner = graph_runner.make()
    assert runner.profile_cudagraph_memory() == 250
    assert runner.encoder_cudagraph_manager is None
    assert graph_runner.events == [
        "create",
        "enter",
        "sync",
        "empty",
        "memory",
        "capture",
        "sync",
        "memory",
        "exit",
        "clear",
        "collect",
        "empty",
    ]
    runner.capture_model()
    assert len(graph_runner.managers) == 2
    assert runner.encoder_cudagraph_manager is graph_runner.managers[1]


@pytest.mark.parametrize("method", ["profile_cudagraph_memory", "capture_model"])
def test_encoder_graph_capture_failure_cleans_up_without_enabling_replay(
    graph_runner, method
):
    runner = graph_runner.make()
    graph_runner.state.fail_capture = True
    with pytest.raises(RuntimeError, match="capture failed"):
        getattr(runner, method)()
    assert "clear" in graph_runner.events
    assert runner.encoder_cudagraph_manager is None


@pytest.mark.parametrize("capture_decoder", [False, True])
def test_v2_runner_captures_encoder_after_decoder_and_counts_memory(
    monkeypatch, capture_decoder
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm.v1.worker.gpu import model_runner as module

    events = []
    memory = iter([1000, 700])
    monkeypatch.setattr(module.gc, "collect", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (next(memory), 2000)
    )
    monkeypatch.setattr(module, "create_lora_capture_hook", lambda *_: None)
    runner = module.GPUModelRunner.__new__(module.GPUModelRunner)
    runner.supports_mm_inputs = runner.is_first_pp_rank = True
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: capture_decoder,
        capture=lambda *args, **kwargs: events.append("decoder"),
    )
    runner.model_state = SimpleNamespace(
        encoder_runner=SimpleNamespace(
            cudagraph_enabled=True,
            encoder_cudagraph_manager=None,
            capture_model=lambda: events.append("encoder"),
            profile_cudagraph_memory=lambda: 250,
        )
    )
    runner.speculator = SimpleNamespace(capture=lambda: events.append("speculator"))
    runner.maybe_setup_dummy_loras = lambda _: nullcontext()
    runner.lora_config = None
    runner.use_aux_hidden_state_outputs = False
    for name in (
        "model",
        "input_buffers",
        "intermediate_tensors",
        "block_tables",
        "attn_groups",
        "kv_cache_config",
    ):
        setattr(runner, name, None)
    assert runner.profile_cudagraph_memory() == 250
    assert runner.capture_model() == 300
    assert events == (["decoder", "speculator"] if capture_decoder else []) + [
        "encoder"
    ]
    runner.is_first_pp_rank = False
    assert runner.profile_cudagraph_memory() == 0


def test_consumed_cpu_input_updates_are_request_local_and_replayable():
    """Release one occurrence, keep layout/embeddings, then restore encode input."""
    pixels = torch.ones(8, HIDDEN)
    feature = replace(_feature("same", 0, 8), data=_embeds_item(pixels))
    later = replace(feature, mm_position=PlaceholderRange(offset=16, length=8))
    source = [feature, later]
    cache = EncoderCache()
    cache.add_request("a", source)
    cache.add_request("b", source)
    cache.encoder_outputs["same"] = torch.ones(8, HIDDEN)
    cache.update_encoder_inputs("a", {0: None})
    assert cache.mm_features["a"][0].data is None
    assert cache.mm_features["a"][0].mm_position is feature.mm_position
    assert cache.mm_features["a"][1].data is feature.data
    assert cache.mm_features["b"][0].data is feature.data
    assert source[0].data is not None
    assert "same" in cache.encoder_outputs
    cache.free_encoder_cache("same")
    cache.update_encoder_inputs("a", {0: source[0].data})
    runner = EncoderRunner(
        model=torch.nn.Module(),
        max_num_tokens=32,
        hidden_size=HIDDEN,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    hashes, inputs = runner.prepare_mm_inputs({"a": [0]})
    assert hashes == ["same"]
    assert inputs == [(feature.modality, feature.data)]


def test_consumed_cpu_input_releases_storage_before_request_finishes():
    """The request may stay alive after its worker-owned storage is freed."""
    cache = EncoderCache()
    pixels = torch.ones(8, HIDDEN)
    ref = weakref.ref(pixels)
    cache.add_request("a", [replace(_feature("h", 0, 8), data=_embeds_item(pixels))])
    del pixels
    assert ref() is not None
    cache.update_encoder_inputs("a", {0: None})
    assert ref() is None
    assert "a" in cache.mm_features
