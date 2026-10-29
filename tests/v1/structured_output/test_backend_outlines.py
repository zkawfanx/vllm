# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import outlines_core as oc
import pytest

from vllm.v1.structured_output.backend_outlines import OutlinesGrammar


@pytest.fixture
def grammar():
    vocab = oc.Vocabulary(3, {b"a": [0], b"b": [1], b"z": [2]})
    return OutlinesGrammar(4, 3, oc.Guide(oc.Index("ab", vocab), max_rollback=8))


def test_eos_termination_is_pure_and_reversible(grammar):
    assert grammar.accept_tokens("r", [0, 1])
    assert not grammar.is_terminated()
    assert not grammar.is_terminated()
    assert grammar.validate_tokens([3, 2]) == [3]
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("r", [3])
    assert grammar.is_terminated()
    grammar.rollback(1)
    assert not grammar.is_terminated()
    assert grammar.validate_tokens([3]) == [3]
    grammar.rollback(2)
    assert grammar.num_processed_tokens == 0
    assert grammar.validate_tokens([0, 1, 3]) == [0, 1, 3]


@pytest.mark.parametrize("tokens", [[0, 2], [0, 3]])
def test_rejected_sequence_preserves_grammar(grammar, tokens):
    assert not grammar.accept_tokens("r", tokens)
    assert grammar.num_processed_tokens == 0
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("r", [0, 1, 3])
    assert grammar.is_terminated()
    grammar.reset()
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0
