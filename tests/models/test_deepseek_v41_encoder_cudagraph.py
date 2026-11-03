# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    build_packed_merge_metadata,
    build_packed_vit_metadata,
    get_vision_cos_sin,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
)
from vllm.models.deepseek_v4_1.common.vl_cudagraph import (
    DeepseekV4VLEncoderCudaGraphMixin,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture
def model():
    model = DeepseekV4VLEncoderCudaGraphMixin()
    model.config = SimpleNamespace(hidden_size=3, vision_downsample_ratio=2)
    model.multimodal_config = None
    attention = SimpleNamespace(attn_backend=AttentionBackendEnum.FLASH_ATTN)
    model.vision = SimpleNamespace(
        blocks=[SimpleNamespace(attn=SimpleNamespace(attn=attention))]
    )
    model.image_start = torch.full((3,), -1.0)
    model.image_newline = torch.full((3,), -2.0)
    model.image_end = torch.full((3,), -3.0)
    return model


@pytest.mark.parametrize("grids", [[(3, 5)], [(2, 4), (3, 5), (1, 1)]])
def test_packed_merge_matches_dense_padding_and_channel_order(grids):
    # Tuple-returning identity projections isolate the spatial merge itself.
    aligner = SimpleNamespace(
        downsample_ratio=2, w1=lambda x: (x, None), w2=lambda x: (x, None)
    )
    sizes = [h * w for h, w in grids]
    x = torch.arange(sum(sizes) * 3, dtype=torch.float32).view(-1, 3) / 10
    metadata = build_packed_merge_metadata(
        grids, 2, device=torch.device("cpu"), dtype=x.dtype
    )
    expected = torch.cat(
        [
            DeepseekV4Aligner.forward(aligner, item, h, w)
            for item, (h, w) in zip(x.split(sizes), grids)
        ]
    )
    actual = DeepseekV4Aligner.forward_packed(aligner, x, **metadata)
    torch.testing.assert_close(actual, expected)


def test_packed_rope_resets_per_image_and_keeps_max_length_on_host():
    grids = [(2, 3), (1, 2)]
    metadata = build_packed_vit_metadata(
        grids,
        rope_dim=4,
        rope_theta=10000,
        device=torch.device("cpu"),
        max_seqlen_override=64,
    )
    tables = [get_vision_cos_sin(h, w, 4, 10000) for h, w in grids]
    torch.testing.assert_close(metadata["vit_cos"], torch.cat([t[0] for t in tables]))
    torch.testing.assert_close(metadata["vit_sin"], torch.cat([t[1] for t in tables]))
    assert metadata["cu_seqlens"].tolist() == [0, 6, 8]
    assert metadata["max_seqlen"].device.type == "cpu"
    assert metadata["max_seqlen"].item() == 64


def test_padding_replaces_stale_offsets_with_zero_length_sequences(model):
    pad = model.get_encoder_cudagraph_config().padding_logics["cu_seqlens"]
    dst = torch.full((6,), 99, dtype=torch.int32)
    for src, expected in [
        ([0, 6, 8], [0, 6, 8, 8, 8, 8]),
        ([0, 3], [0, 3, 3, 3, 3, 3]),
        ([0], [0, 0, 0, 0, 0, 0]),
        ([], [0, 0, 0, 0, 0, 0]),
    ]:
        pad(dst, torch.tensor(src, dtype=torch.int32))
        assert dst.tolist() == expected


def test_selection_and_span_assembly_preserve_order_and_own_output(model):
    types = torch.tensor(
        [
            IMAGE_START,
            IMAGE,
            IMAGE_NEW_LINE,
            IMAGE_END,
            IMAGE_START,
            IMAGE,
            IMAGE,
            IMAGE_NEW_LINE,
            IMAGE_END,
        ]
    )
    mm_kwargs = {
        "vit_grid": torch.tensor([[2, 2], [2, 4]]),
        "llm_grid": torch.tensor([[1, 1], [1, 2]]),
        "patches": torch.arange(12).view(12, 1),
        "types": types,
    }
    selected = model.select_encoder_cudagraph_items(mm_kwargs, [1, 0])
    assert selected["patches"].flatten().tolist() == list(range(4, 12)) + list(range(4))
    specs = model.get_encoder_cudagraph_item_specs(selected)
    assert [(s.input_size, s.output_tokens) for s in specs] == [(8, 5), (4, 4)]
    aligner_out = torch.arange(12, dtype=torch.float32).view(4, 3)
    dest = {}
    model.postprocess_encoder_output(
        aligner_out,
        [1, 0],
        [4, 5],
        dest,
        batch_mm_kwargs=selected,
    )
    expected = [
        torch.stack(
            [model.image_start, aligner_out[2], model.image_newline, model.image_end]
        ),
        torch.stack(
            [
                model.image_start,
                aligner_out[0],
                aligner_out[1],
                model.image_newline,
                model.image_end,
            ]
        ),
    ]
    aligner_out.fill_(999)  # A subsequent graph replay must not alter delivered spans.
    for idx in range(2):
        torch.testing.assert_close(dest[idx], expected[idx])
    empty = model.select_encoder_cudagraph_items(mm_kwargs, [])
    assert empty["patches"].numel() == empty["types"].numel() == 0
    assert empty["vit_grid"].shape == empty["llm_grid"].shape == (0, 2)


@pytest.mark.parametrize("backend", ["FLASHINFER", "TORCH_SDPA"])
def test_uncapturable_attention_backend_is_rejected(model, backend):
    model.vision.blocks[0].attn.attn.attn_backend = AttentionBackendEnum[backend]
    with pytest.raises(ValueError, match=backend):
        model.get_encoder_cudagraph_config()


def test_dynamic_fp8_is_rejected_but_static_scales_are_allowed(model):
    model.multimodal_config = SimpleNamespace(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=None,
        get_limit_per_prompt=lambda _: 1,
    )
    with pytest.raises(ValueError, match="dynamic FP8"):
        model.get_encoder_cudagraph_config()
    model.multimodal_config.mm_encoder_fp8_scale_path = "scales.json"
    assert model.get_encoder_cudagraph_config().modalities == ["image"]


def test_eager_fallback_returns_complete_spans_for_direct_manager_scatter(model):
    model.aligner = SimpleNamespace(w1=SimpleNamespace(weight=torch.zeros(1)))
    model._encode_image = lambda patches, h, w: patches[: -(-h // 2) * -(-w // 2)]
    mm_kwargs = {
        "vit_grid": [[2, 2], [2, 4]],
        "llm_grid": [[1, 1], [1, 2]],
        "patches": torch.arange(36, dtype=torch.float32).view(12, 3),
        "types": torch.tensor(
            [
                IMAGE_START,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_END,
                IMAGE_START,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_END,
            ]
        ),
    }
    result = model.encoder_eager_forward(mm_kwargs)
    patches = mm_kwargs["patches"]
    expected = torch.stack(
        [
            model.image_start,
            patches[0],
            model.image_newline,
            model.image_end,
            model.image_start,
            patches[4],
            patches[5],
            model.image_newline,
            model.image_end,
        ]
    )
    torch.testing.assert_close(result, expected)
    assert [part.shape[0] for part in result.split([4, 5])] == [4, 5]


def test_zero_image_limit_is_rejected_before_accessing_stubbed_vision(model):
    model.multimodal_config = SimpleNamespace(get_limit_per_prompt=lambda _: 0)
    model.vision = None
    with pytest.raises(RuntimeError, match="image limit is 0"):
        model.get_encoder_cudagraph_config()


def test_manager_restores_order_across_graph_and_oversized_eager_fallback(
    model, monkeypatch
):
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    model.aligner = SimpleNamespace(w1=SimpleNamespace(weight=torch.zeros(1)))
    model._encode_image = lambda patches, h, w: patches[: -(-h // 2) * -(-w // 2)]
    patches = torch.arange(36, dtype=torch.float32).view(12, 3)
    mm_kwargs = {
        "vit_grid": [[2, 4], [2, 2]],
        "llm_grid": [[1, 2], [1, 1]],
        "patches": patches,
        "types": torch.tensor(
            [
                IMAGE_START,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_END,
                IMAGE_START,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_END,
            ]
        ),
    }
    manager = EncoderCudaGraphManager.__new__(EncoderCudaGraphManager)
    manager.model = model
    manager.token_budgets = [4]
    manager.max_batch_size = 2
    manager.graph_misses = 0
    graph_calls = []
    graph_output = patches[8:9].clone()

    def replay(batch, budget):
        graph_calls.append((batch["patches"].clone(), budget))
        return graph_output

    monkeypatch.setattr(manager, "_run_budget_graph", replay)
    outputs = manager._execute_local_single_path(mm_kwargs)
    assert manager.graph_misses == 1
    assert len(graph_calls) == 1
    torch.testing.assert_close(graph_calls[0][0], patches[8:])
    assert graph_calls[0][1] == 4
    graph_output.fill_(999)
    expected = [
        torch.stack(
            [
                model.image_start,
                patches[0],
                patches[1],
                model.image_newline,
                model.image_end,
            ]
        ),
        torch.stack(
            [model.image_start, patches[8], model.image_newline, model.image_end]
        ),
    ]
    assert [output.shape[0] for output in outputs] == [5, 4]
    for actual, wanted in zip(outputs, expected):
        torch.testing.assert_close(actual, wanted)
