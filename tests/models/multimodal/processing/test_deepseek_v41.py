# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
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
    "max_wh_ratio,min_pixels",
    [(None, 0), (2.0, 0), (1.25, 295936), (None, 295936)],
)
def test_profile_image_covers_maximum_processed_features(max_wh_ratio, min_pixels):
    """Profiling must cover every legal patch grid, including newline overhead."""
    budget = 64
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
    processed = info.get_hf_processor()(
        images=[Image.new("RGB", (size.width, size.height))]
    )

    downsample = config.vision_downsample_ratio
    # Enumerate ViT grids directly, independently of the profiling search.
    max_patches = max(
        h * w
        for h in range(1, downsample * (budget - 2) // 2 + 1)
        for w in range(1, downsample * (budget - 3) + 1)
        if math.ceil(h / downsample) * (math.ceil(w / downsample) + 1) + 2 <= budget
        and (max_wh_ratio is None or w <= math.ceil(h * max_wh_ratio))
    )
    assert processed["patches"].shape[0] == max_patches
    assert processed["types"].numel() <= budget
    assert size.height * size.width >= min_pixels


def test_profile_image_rejects_budget_without_room_for_patches():
    ctx = MagicMock()
    ctx.get_hf_config.return_value = DeepseekV41Config(
        vision_config={"max_image_tokens": 3}
    )
    with pytest.raises(ValueError, match="allow no image patches"):
        DeepseekV4VLProcessingInfo(ctx).get_image_size_with_most_features()
