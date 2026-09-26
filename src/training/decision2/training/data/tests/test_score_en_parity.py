"""Small contract checks for the gold-free source parity gate."""

from __future__ import annotations

import pytest

from training.data.score_en_parity import _probabilities, choose_roster


class _Tokenizer:
    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(char) % 256 for char in value]


def _row(index: int, kind: str) -> dict:
    options = (
        [
            {"key": "false", "description": "no"},
            {"key": "true", "description": "yes"},
        ]
        if kind == "noul"
        else [
            {"key": "a", "description": "left"},
            {"key": "b", "description": "right"},
        ]
    )
    return {
        "id": f"{kind}-{index}",
        "language": "en",
        "task_type": kind,
        "state": "A short context.",
        "instructions": "Choose the supported option.",
        "options": options,
        "label": 0,
    }


def test_gold_free_roster_is_balanced_and_deterministic() -> None:
    rows = [_row(i, kind) for kind in ("choice", "noul") for i in range(20)]
    prompts, encoded = choose_roster(rows, _Tokenizer())
    again, _ = choose_roster(list(reversed(rows)), _Tokenizer())
    assert prompts == again
    assert len(prompts) == len(encoded) == 32
    assert all(set(prompt) == {"id", "state", "questions"} for prompt in prompts)
    assert [prompt["questions"]["decision"]["type"] for prompt in prompts].count(
        "noul"
    ) == 16
    assert all("label" not in str(prompt) for prompt in prompts)


def test_parity_probabilities_require_valid_native_answer() -> None:
    probs, decision = _probabilities(
        {
            "adapter_status": "ok",
            "adapter_errors": {},
            "answers": {"decision": {"type": "noul", "noul": 0.7}},
        },
        "noul",
    )
    assert probs == pytest.approx({"false": 0.3, "true": 0.7})
    assert decision == "true"
    with pytest.raises(ValueError, match="invalid model response"):
        _probabilities({"adapter_status": "invalid", "adapter_errors": {}}, "noul")
