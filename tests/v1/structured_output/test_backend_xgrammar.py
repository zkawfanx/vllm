# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


def test_rejected_sequence_rolls_back_matcher_and_counter():
    matcher = Mock()
    matcher.accept_token.side_effect = [True, False, True]
    matcher.is_terminated.return_value = False
    grammar = XgrammarGrammar(vocab_size=4, matcher=matcher, ctx=Mock())
    assert not grammar.accept_tokens("r", [0, 2])
    matcher.rollback.assert_called_once_with(1)
    assert grammar.num_processed_tokens == 0
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("r", [0])
    assert grammar.num_processed_tokens == 1
