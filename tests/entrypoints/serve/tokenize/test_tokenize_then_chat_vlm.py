# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test: calling ``/tokenize`` with multimodal data followed by
``/v1/chat/completions`` with the same data must not cause an error.

Ensures that the ``/tokenize`` endpoint does not pollute internal caches
(e.g. multimodal feature caches) and that a subsequent
``/v1/chat/completions`` request with the same multimodal payload
completes successfully.
"""

import io
import json
from typing import Any

import openai
import pybase64 as base64
import pytest
import pytest_asyncio
import requests
from PIL import Image

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_tokenize_then_chat_completion_with_image(
    client: openai.AsyncOpenAI,
    server: RemoteOpenAIServer,
    local_asset_server,
):
    """Tokenize a multimodal message, then send the same message to chat
    completions.  The chat completion must succeed (not 500)."""

    image_url = local_asset_server.url_for("stop_sign.jpg")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]

    tok_resp = requests.post(
        server.url_for("tokenize"),
        json={"model": MODEL_NAME, "messages": messages},
    )
    tok_resp.raise_for_status()
    tok_data = tok_resp.json()
    assert tok_data["count"] > 0, "Tokenization must return tokens"

    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
    )

    assert chat_completion.choices[0].message.content, (
        "Chat completion must produce non-empty content after tokenize"
    )


@pytest.mark.parametrize(
    "anthropic_generation", [False, True], ids=["chat", "messages"]
)
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.asyncio
async def test_count_tokens_then_generate_cold_image(
    server: RemoteOpenAIServer, anthropic_generation: bool, stream: bool
):
    """Repeated count-only calls must not break the first decode of an image."""
    # Different pixels keep every case cold despite the module-scoped server.
    image = Image.new(
        "RGB", (56, 56), (32 + 64 * anthropic_generation, 64 * stream, 96)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode()
    content = [
        {"type": "text", "text": "Describe this image briefly."},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data,
            },
        },
    ]
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
    }
    counts = []
    for _ in range(2):
        response = requests.post(
            server.url_for("v1/messages/count_tokens"), json=payload, timeout=120
        )
        response.raise_for_status()
        counts.append(response.json()["input_tokens"])
    assert counts[0] == counts[1] > 0

    endpoint = "v1/messages" if anthropic_generation else "v1/chat/completions"
    if not anthropic_generation:
        payload["messages"][0]["content"][1] = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"},
        }
    response = requests.post(
        server.url_for(endpoint),
        json={**payload, "max_tokens": 16, "temperature": 0.0, "stream": stream},
        timeout=120,
    )
    response.raise_for_status()
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert not any(
            event.get("error") or event.get("type") == "error" for event in events
        )
        if anthropic_generation:
            assert any(event.get("type") == "message_stop" for event in events)
            assert any(event.get("delta", {}).get("text") for event in events)
        else:
            choices = [
                choice for event in events for choice in event.get("choices", [])
            ]
            assert any(choice["delta"].get("content") for choice in choices)
            assert any(choice.get("finish_reason") for choice in choices)
    else:
        result = response.json()
        if anthropic_generation:
            assert any(block.get("text") for block in result["content"])
        else:
            assert result["choices"][0]["message"]["content"]
