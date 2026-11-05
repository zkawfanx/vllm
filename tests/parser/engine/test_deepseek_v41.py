# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace

import pytest

from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.deepseek_v4 import deepseek_v4_config
from vllm.parser.deepseek_v41 import DeepSeekV41Parser, deepseek_v41_config
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.parser_manager import ParserManager

CALLS = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="count" string="false">42</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="add">\n'
    '<｜DSML｜ parameter name="x" string="false">1.5</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="y" string="false">2.25</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


def tokenizer_for(text, special_calls):
    vocab = {"<think>": 50, "</think>": 51, "<｜User｜>": 54, "<｜Assistant｜>": 55}
    if special_calls:
        vocab |= {"<｜DSML｜ calls>": 52, "</｜DSML｜ calls>": 53}
    tokens: list[tuple[int, str]] = []
    while text:
        special = next((marker for marker in vocab if text.startswith(marker)), None)
        piece = special or text[0]
        tokens.append((vocab[piece] if special else 100 + len(tokens), piece))
        text = text[len(piece) :]
    return MockTokenizer(vocab, tokens), tokens


def parser_for(tokenizer, controls):
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
    )
    return cls(tokenizer, chat_template_kwargs=controls)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("special_calls", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_registered_adapters_parse_parallel_calls_across_chunks(
    thinking,
    special_calls,
    chunk_size,
):
    text = ("Plan.</think>" if thinking else "") + "Checking." + CALLS
    tokenizer, tokens = tokenizer_for(text, special_calls)
    parser = parser_for(tokenizer, {"thinking": thinking})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    assert output.reasoning == ("Plan." if thinking else "")
    assert output.content.strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [json.loads(call["arguments"]) for call in output.tool_calls] == [
        {"city": "杭州", "count": 42},
        {"x": 1.5, "y": 2.25},
    ]


@pytest.mark.parametrize("thinking", [False, True])
def test_registered_adapters_parse_complete_output(thinking):
    text = ("Plan.</think>" if thinking else "") + CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": thinking})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert (reasoning or "") == ("Plan." if thinking else "")
    assert not (content or "").strip()
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert json.loads(calls[0].arguments) == {"city": "杭州", "count": 42}


@pytest.mark.parametrize(
    ("controls", "text", "expected_reasoning", "reasoning_tokens"),
    [
        ({"thinking": False}, "12", "", 0),
        ({"enable_thinking": False}, "12", "", 0),
        ({"thinking": True, "reasoning_effort": "none"}, "12", "", 0),
        ({}, "</think>12", "", 0),
        ({}, "Plan.</think>12", "Plan.", 5),
    ],
)
def test_reasoning_adapter_controls_and_usage(
    controls,
    text,
    expected_reasoning,
    reasoning_tokens,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, controls)
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
        )
    )
    assert output.reasoning == expected_reasoning
    assert output.content == "12"
    assert (
        parser.reasoning_parser.count_reasoning_tokens([tid for tid, _ in tokens])
        == reasoning_tokens
    )
    assert parser.is_reasoning_end([50, 100, 51])
    assert not parser.is_reasoning_end([51, 100, 50])


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("native_engine", [False, True])
@pytest.mark.parametrize(
    ("prompt_ids", "text", "reasoning", "content", "reasoning_tokens"),
    [
        ([54, 100, 55, 50, 101, 51, 102], "12", "", "12", 0),
        ([55, 50, 101, 51, 102], "12</think>34", "", "1234", 0),
        ([54, 100, 55, 50, 101], "Plan.</think>12", "Plan.", "12", 5),
        # A prior turn's close must not end even implicitly opened reasoning.
        ([55, 50, 101, 51, 54, 102, 55], "Plan.</think>12", "Plan.", "12", 5),
        ([55, 50, 101, 51, 54, 102, 55, 50], "Plan.</think>12", "Plan.", "12", 5),
    ],
)
def test_prompt_reasoning_state_and_counts(
    streaming,
    native_engine,
    prompt_ids,
    text,
    reasoning,
    content,
    reasoning_tokens,
    chunk_size,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = (
        DeepSeekV41Parser(tokenizer, chat_template_kwargs={"thinking": True})
        if native_engine
        else parser_for(tokenizer, {"thinking": True})
    )
    output_ids = [tid for tid, _ in tokens]
    counter = parser if native_engine else parser.reasoning_parser
    if streaming:
        output = collect_output(
            replay_streaming(
                parser,
                tokens,
                chunk_size=chunk_size,
                finished_on_last=True,
                prompt_token_ids=prompt_ids,
            )
        )
        actual_reasoning, actual_content = output.reasoning, output.content
    else:
        parser.set_prompt_token_ids(prompt_ids)
        assert counter.count_reasoning_tokens(output_ids) == reasoning_tokens
        actual_reasoning, actual_content, _ = parser.parse(
            text, _test_request(), model_output_token_ids=output_ids
        )
    assert (actual_reasoning or "") == reasoning
    assert actual_content == content
    assert counter.count_reasoning_tokens(output_ids) == reasoning_tokens


def test_prompt_state_is_replaced_and_explicit_reasoning_still_counts():
    tokenizer, _ = tokenizer_for("", False)
    parser = parser_for(tokenizer, {"thinking": True})
    parser.set_prompt_token_ids([55, 50, 51])
    assert parser.reasoning_parser.count_reasoning_tokens([100, 50, 101, 51, 102]) == 1
    assert parser.reasoning_parser.count_reasoning_tokens([100, 101]) == 0
    parser.set_prompt_token_ids([55])
    assert parser.reasoning_parser.count_reasoning_tokens([100, 101]) == 2


@pytest.mark.parametrize("missing_vocab", [False, True])
def test_prompt_state_requires_resolved_turn_boundaries(missing_vocab):
    tokenizer, _ = tokenizer_for("", False)
    config = deepseek_v41_config(thinking=True)
    if missing_vocab:
        tokenizer.set_vocab({"<think>": 50, "</think>": 51})
    else:
        config = replace(config, turn_boundary_tokens=frozenset())
    parser = ParserEngine(tokenizer, parser_engine_config=config)
    parser.set_prompt_token_ids([55, 50, 51])
    assert parser.extract_reasoning("Plan.", _test_request()) == ("Plan.", None)
    assert not deepseek_v4_config(thinking=True).turn_boundary_tokens


def test_explicit_content_initial_state_overrides_prompt():
    tokenizer, tokens = tokenizer_for("12", False)
    parser = DeepSeekV41Parser(tokenizer)
    parser.set_prompt_token_ids([55, 50])
    reasoning, content, _ = parser._single_pass_parse(
        "12", [tid for tid, _ in tokens], initial_state=ParserState.CONTENT
    )
    assert reasoning is None
    assert content == "12"


def test_python_argument_conversion_and_partial_values():
    converter = deepseek_v41_config().arg_converter
    raw = (
        '<｜DSML｜ parameter name="object" string="false">'
        '{"a": [true, null]}</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bad" string="false">[broken</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bare">7</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="text" string="true">  a<b'
    )
    assert json.loads(converter(raw, True)) == {
        "object": {"a": [True, None]},
        "bad": "[broken",
        "bare": 7,
        "text": "  a<b",
    }


@pytest.mark.parametrize("streaming", [False, True])
def test_registered_adapters_keep_parameters_without_string_attribute(streaming):
    text = (
        'Plan.</think><｜DSML｜ calls><｜DSML｜ invoke name="get_weather">'
        '<｜DSML｜ parameter name="city">杭州</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="count">7</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="options">{"k": [true, null]}</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="quoted" string="true">42</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": True})
    if streaming:
        output = collect_output(
            replay_streaming(
                parser,
                tokens,
                chunk_size=3,
                finished_on_last=True,
                tools=DUMMY_TOOLS,
                prompt_token_ids=[50],
            )
        )
        assert output.reasoning == "Plan."
        assert len(output.tool_calls) == 1
        arguments = output.tool_calls[0]["arguments"]
    else:
        reasoning, _, calls = parser.parse(
            text,
            _test_request(tools=DUMMY_TOOLS),
            enable_auto_tools=True,
            model_output_token_ids=[tid for tid, _ in tokens],
        )
        assert reasoning == "Plan."
        assert len(calls) == 1
        arguments = calls[0].arguments
    assert json.loads(arguments) == {
        "city": "杭州",
        "count": 7,
        "options": {"k": [True, None]},
        "quoted": "42",
    }


def test_adjacent_parameters_preserve_implicit_close():
    converter = deepseek_v41_config().arg_converter
    raw = (
        '<｜DSML｜ parameter name="first" string="true">text'
        '<｜DSML｜ parameter name="second" string="false">42</｜DSML｜ parameter>'
    )
    assert json.loads(converter(raw, False)) == {"first": "text", "second": 42}
