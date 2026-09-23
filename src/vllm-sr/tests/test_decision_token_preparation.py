"""Framework-free token preparation tests for both Decision families."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.contracts import ChoiceQuestion, ScoreQuestion  # noqa: E402
from decision_runtime.model_inputs import (  # noqa: E402
    QWEN_DEFAULT_NO,
    QWEN_DEFAULT_YES,
    VELA_DEFAULT_NO,
    VELA_DEFAULT_YES,
    build_model_input,
)
from decision_runtime.qwen35_inputs import (  # noqa: E402
    encode_qwen_rows,
    token_ids_sha256,
)
from decision_runtime.vela_inputs import encode_vela_rows  # noqa: E402

QWEN_POLICY = {
    "choice_null_description": "preserve_json_null",
    "noul_default_false": QWEN_DEFAULT_NO,
    "noul_default_true": QWEN_DEFAULT_YES,
    "noul_explicit_null": "preserve_json_null",
}
VELA_POLICY = {
    "choice_null_description": "render_key",
    "noul_default_false": VELA_DEFAULT_NO,
    "noul_default_true": VELA_DEFAULT_YES,
    "noul_explicit_null": "use_default",
}


class CharacterTokenizer:
    """Stable fake tokenizer that makes segment boundaries observable."""

    cls_token_id = 101
    bos_token_id = None
    sep_token_id = 102
    eos_token_id = None
    pad_token_id = 0
    mask_token_id = 103

    def __init__(self) -> None:
        self.batch_calls = 0
        self.encode_calls = 0
        self.single_calls = 0

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [200 + ord(character) for character in text]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encode_calls += 1
        return self._ids(text)

    def __call__(self, value, **kwargs):
        assert kwargs["add_special_tokens"] is False
        if isinstance(value, list):
            self.batch_calls += 1
            return {"input_ids": [self._ids(text) for text in value]}
        assert kwargs["truncation"] is False
        self.single_calls += 1
        return {"input_ids": self._ids(value)}


def _choice(question_id: str, state: str):
    question = ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {"left": None, "right": "right"},
        }
    )
    return build_model_input(
        question_id=question_id,
        state=state,
        question=question,
        **QWEN_POLICY,
    )


def test_qwen_preparation_batches_unique_segments_and_preserves_identity() -> None:
    tokenizer = CharacterTokenizer()
    rows = (_choice("first", "A"), _choice("second", "B"))

    encoded = encode_qwen_rows(rows, tokenizer, max_length=20_000)

    assert tokenizer.batch_calls == 1
    assert tokenizer.encode_calls == 0
    assert [row.question_id for row in encoded] == ["first", "second"]
    assert all(len(row.candidate_positions) == 2 for row in encoded)
    assert all(row.candidate_positions[-1] < row.query_position for row in encoded)
    assert encoded[0].prompt_sha256 != encoded[1].prompt_sha256
    assert len(token_ids_sha256(encoded[0])) == 64


def test_qwen_preparation_falls_back_without_cross_request_cache() -> None:
    tokenizer = CharacterTokenizer()
    encoded = encode_qwen_rows(
        (_choice("move", "A"),),
        tokenizer,
        max_length=20_000,
        max_cached_characters=0,
    )

    assert tokenizer.batch_calls == 0
    assert tokenizer.encode_calls == 4
    assert encoded[0].input_tokens > 0


def test_qwen_preparation_never_truncates() -> None:
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="no truncation allowed"):
        encode_qwen_rows((_choice("move", "A"),), tokenizer, max_length=10)


def test_vela_preparation_uses_markers_and_keeps_question_ids_out_of_tokens() -> None:
    tokenizer = CharacterTokenizer()
    score = ScoreQuestion.model_validate(
        {
            "type": "score",
            "instructions": "Rate.",
            "criteria": ["low", "high"],
        }
    )
    row = build_model_input(
        question_id="§opaque-id",
        state="board",
        question=score,
        **VELA_POLICY,
    )

    encoded = encode_vela_rows((row,), tokenizer, max_length=1000)[0]

    assert encoded.input_ids[0] == tokenizer.cls_token_id
    assert encoded.input_ids[-1] == tokenizer.sep_token_id
    assert len(encoded.marker_positions) == 2
    assert all(
        encoded.input_ids[position] == tokenizer.mask_token_id
        for position in encoded.marker_positions
    )
    assert 200 + ord("§") not in encoded.input_ids


def test_vela_preparation_rejects_complete_input_overflow() -> None:
    tokenizer = CharacterTokenizer()
    question = ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {"left": None, "right": "right"},
        }
    )
    row = build_model_input(
        question_id="move",
        state="A",
        question=question,
        **VELA_POLICY,
    )
    with pytest.raises(ValueError, match="no room|no truncation allowed"):
        encode_vela_rows((row,), tokenizer, max_length=8)


def test_vela_preparation_reuses_exact_spans_for_many_questions_and_states() -> None:
    question = ChoiceQuestion.model_validate(
        {
            "type": "choice",
            "instructions": "Choose.",
            "criteria": {"left": "move left", "right": "move right"},
        }
    )
    rows = tuple(
        build_model_input(
            question_id=f"q{index}",
            state="shared state" if index < 3 else "other state",
            question=question,
            **VELA_POLICY,
        )
        for index in range(4)
    )
    cached_tokenizer = CharacterTokenizer()
    cached = encode_vela_rows(rows, cached_tokenizer, max_length=1000)
    uncached_tokenizer = CharacterTokenizer()
    uncached = encode_vela_rows(
        rows, uncached_tokenizer, max_length=1000, max_cached_characters=0
    )

    assert cached == uncached
    assert cached_tokenizer.single_calls == 5  # question, 2 candidates, 2 states
    assert uncached_tokenizer.single_calls == 16
