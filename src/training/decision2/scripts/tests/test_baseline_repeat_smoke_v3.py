"""Synthetic, gold-free checks for the v3 baseline repeatability audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference.run import digest
from scripts.baseline_repeat_smoke_v3 import compare


def _rows(path: Path, *, delta: float = 0, changed: bool = False) -> None:
    prompts = [
        json.loads(line)
        for line in (path.parent / "prompts.jsonl").read_text().splitlines()
    ]
    with path.open("w") as output:
        for row in prompts:
            question = next(iter(row["questions"]))
            kind = row["questions"][question]["type"]
            if kind == "choice":
                answer = {
                    "type": kind,
                    "choice": "b" if changed else "a",
                    "probabilities": {"a": 0.7 - delta, "b": 0.3 + delta},
                }
            elif kind == "noul":
                answer = {"type": kind, "noul": 0.7 + delta}
            else:
                answer = {
                    "type": kind,
                    "score": 0.3,
                    "probabilities": {"0": 0.7 - delta, "1": 0.3 + delta},
                }
            output.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "answers": {question: answer},
                        "source_input_sha256": digest(
                            {"state": row["state"], "questions": row["questions"]}
                        ),
                        "model_id": "example/nox",
                        "model_revision": "revision",
                        "backend": "nox",
                        "adapter_version": "native-published-v2",
                        "revision_attested": True,
                        "runtime_matches_validated": True,
                        "usage": {"input_tokens": 100},
                    }
                )
                + "\n"
            )


def test_exact_two_process_probabilities_and_categories(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps(
                {
                    "id": str(index),
                    "state": "state",
                    "questions": {"q": {"type": kind}},
                }
            )
            + "\n"
            for index, kind in enumerate(
                ("choice", "noul", "score") * 10 + ("choice", "noul")
            )
        )
    )
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _rows(first)
    _rows(second)
    result = compare(
        prompts,
        first,
        second,
        model_id="example/nox",
        revision="revision",
        backend="nox",
        adapter_version="native-published-v2",
    )
    assert result["gate_pass"] is True
    assert result["max_option_probability_drift"] == 0
    _rows(second, delta=0.01)
    assert (
        compare(
            prompts,
            first,
            second,
            model_id="example/nox",
            revision="revision",
            backend="nox",
            adapter_version="native-published-v2",
        )["gate_pass"]
        is False
    )
    _rows(second, changed=True)
    result = compare(
        prompts,
        first,
        second,
        model_id="example/nox",
        revision="revision",
        backend="nox",
        adapter_version="native-published-v2",
    )
    assert result["categorical_mismatch_n"] > 0
    assert result["gate_pass"] is False


def test_input_or_runtime_tamper_fails(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps(
                {"id": str(i), "state": "state", "questions": {"q": {"type": "noul"}}}
            )
            + "\n"
            for i in range(32)
        )
    )
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _rows(first)
    _rows(second)
    records = [json.loads(line) for line in second.read_text().splitlines()]
    records[0]["runtime_matches_validated"] = False
    second.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="runtime"):
        compare(
            prompts,
            first,
            second,
            model_id="example/nox",
            revision="revision",
            backend="nox",
            adapter_version="native-published-v2",
        )


def test_same_native_overflow_is_repeatable_but_mismatch_fails(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps(
                {"id": str(i), "state": "state", "questions": {"q": {"type": "noul"}}}
            )
            + "\n"
            for i in range(32)
        )
    )
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    for path in (first, second):
        _rows(path)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["answers"]["q"] = {
            "type": "noul",
            "error": "context_overflow",
        }
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    result = compare(
        prompts,
        first,
        second,
        model_id="example/nox",
        revision="revision",
        backend="nox",
        adapter_version="native-published-v2",
    )
    assert result["gate_pass"] is True
    assert result["stable_invalid_n"] == 1
    records = [json.loads(line) for line in second.read_text().splitlines()]
    records[0]["answers"]["q"]["error"] = "candidate_limit"
    second.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="invalid answer differs"):
        compare(
            prompts,
            first,
            second,
            model_id="example/nox",
            revision="revision",
            backend="nox",
            adapter_version="native-published-v2",
        )
