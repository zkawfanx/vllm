# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config


def test_processor_tokenizes_text_without_applying_image_updates():
    prompt = "before<｜deepseek_image｜>after"
    image = object()
    prompt_ids = [1, 129264, 2]
    tokenizer = MagicMock()
    tokenizer.encode.return_value = prompt_ids
    hf_processor = MagicMock()
    ctx = MagicMock()
    ctx.call_hf_processor.return_value = BatchFeature(
        {"patches": torch.ones(1, dtype=torch.bfloat16)}
    )

    info = MagicMock(spec=DeepseekV4VLProcessingInfo)
    info.ctx = ctx
    info.get_hf_processor.return_value = hf_processor
    info.get_tokenizer.return_value = tokenizer
    processor = DeepseekV4VLMultiModalProcessor(info, MagicMock())
    tok_kwargs = {"truncation": False, "add_special_tokens": False}

    actual_prompt_ids, processed, is_update_applied = (
        processor._apply_hf_processor_text_mm(
            prompt,
            MultiModalDataItems({"image": ImageProcessorItems([image])}),
            {},
            tok_kwargs,
        )
    )

    assert actual_prompt_ids == prompt_ids
    assert "input_ids" not in processed
    assert torch.equal(processed["patches"], torch.ones(1, dtype=torch.bfloat16))
    assert not is_update_applied
    tokenizer.encode.assert_called_once_with(prompt, **tok_kwargs)
    ctx.call_hf_processor.assert_called_once_with(
        hf_processor,
        {"text": prompt, "images": [image]},
        tok_kwargs,
    )


@pytest.mark.parametrize(
    "budget,max_wh_ratio,min_pixels",
    [
        (64, None, 0),
        (64, None, 295936),
        (64, 1.0, 295936),
        (64, 1.25, 295936),
        (64, 1.4, 0),
        (64, 1.4, 295936),
        (64, 1.4, 1000000),
        (64, 2.0, 0),
        (64, 2.0, 295936),
        (1024, None, 295936),
    ],
)
def test_profile_image_covers_processed_features(budget, max_wh_ratio, min_pixels):
    """Rank actual preprocessing, not a grid lost to clamping or downscaling."""
    config = DeepseekV41Config(
        vision_config={
            "max_image_tokens": budget,
            "max_wh_ratio": max_wh_ratio,
            "min_pixels": min_pixels,
        }
    )
    ctx = MagicMock()
    ctx.get_hf_config.return_value = config
    info = DeepseekV4VLProcessingInfo(ctx)
    size = info.get_image_size_with_most_features()
    processor = info.get_hf_processor()
    processed = processor(images=[Image.new("RGB", (size.width, size.height))])
    profile_patches = processed["patches"].shape[0]

    # Include the square and portrait that exceed the former 408-patch dummy
    # at budget=64, max_wh_ratio=1.4, min_pixels=295936.
    for dimensions in [(512, 512), (388, 776), (640, 480), (728, 504), (2814, 630)]:
        reference = processor(images=[Image.new("RGB", dimensions)])
        assert profile_patches >= reference["patches"].shape[0], dimensions
    assert processed["types"].numel() <= budget
    if budget == 1024:
        assert profile_patches == 9189


@pytest.mark.parametrize(
    "vision_config,dimensions",
    [
        (
            {
                "max_image_tokens": 41,
                "patch_size": 10,
                "downsample_ratio": 4,
                "max_wh_ratio": 3.9,
                "min_pixels": 460000,
            },
            (1760, 480),
        ),
        ({"max_image_tokens": 5, "max_wh_ratio": 0.75}, (512, 1024)),
        (
            {
                "max_image_tokens": 5,
                "max_wh_ratio": 0.75,
                "downsample_ratio": 1,
            },
            (512, 1024),
        ),
        (
            {
                "max_image_tokens": 4,
                "downsample_ratio": 2,
                "max_wh_ratio": 0.5,
                "min_pixels": 0,
            },
            (1284, 2646),
        ),
    ],
)
def test_profile_image_covers_downscaled_grids(vision_config, dimensions):
    """Downscaling can favor full aligner cells over the rounded clamp edge."""
    ctx = MagicMock()
    ctx.get_hf_config.return_value = DeepseekV41Config(vision_config=vision_config)
    info = DeepseekV4VLProcessingInfo(ctx)
    size = info.get_image_size_with_most_features()
    processor = info.get_hf_processor()
    profile = processor(images=[Image.new("RGB", (size.width, size.height))])
    reference = processor(images=[Image.new("RGB", dimensions)])
    assert profile["patches"].shape[0] >= reference["patches"].shape[0]
    assert profile["types"].numel() <= vision_config["max_image_tokens"]


def test_profile_image_rejects_budget_without_room_for_patches():
    ctx = MagicMock()
    ctx.get_hf_config.return_value = DeepseekV41Config(
        vision_config={"max_image_tokens": 3}
    )
    with pytest.raises(ValueError, match="allow no image patches"):
        DeepseekV4VLProcessingInfo(ctx).get_image_size_with_most_features()
