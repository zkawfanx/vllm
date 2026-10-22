# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import torch
from transformers import BatchFeature

from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems


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
