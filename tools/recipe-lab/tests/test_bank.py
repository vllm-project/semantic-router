"""Answer bank tests: build/grade/apply and the evidence-first join."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from recipe_lab.bank import BankEntry, apply_bank, grade, load_bank, write_bank
from recipe_lab.contract import ContractError


def test_grade_substring_case_insensitive():
    # I: grade(answer, case) dispatches on the case's grader; contains is the
    # default when the case carries expected_contains.
    assert (
        grade("The capital of France is Paris.", {"expected_contains": "paris"}) is True
    )
    assert grade("Bonjour!", {"expected_contains": "paris"}) is False
    assert grade("anything", {"expected_contains": None}) is None
    assert grade("anything", {}) is None
    assert grade("ok", {"grader": "contains", "expected_contains": "OK"}) is True


def test_grade_mcq_uses_extracted_letter(monkeypatch):
    monkeypatch.setattr(
        "recipe_lab.bank._MCQ_LETTER", lambda text: "A" if "Option A" in text else None
    )
    case = {"grader": "mcq", "expected": "a", "case_id": "c1"}
    assert grade("Option A is correct", case) is True  # expected letter is normalized
    assert grade("Option B is correct", case) is False
    # no extractable letter is a wrong answer, not unevaluated (mcq_judge
    # counts no_answer as wrong)
    assert grade("I don't know", case) is False


def test_grade_mcq_requires_judge_module(monkeypatch):
    monkeypatch.setattr("recipe_lab.bank._MCQ_LETTER", None)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("training"):
            raise ImportError("no training package on sys.path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ContractError, match=r"mcq grader needs src/ on sys\.path"):
        grade("Option A", {"grader": "mcq", "expected": "A", "case_id": "c1"})


def test_grade_unknown_grader_is_contract_error():
    with pytest.raises(ContractError, match="unknown grader"):
        grade("x", {"grader": "llm-judge", "expected": "y", "case_id": "c1"})


def test_write_load_bank_roundtrip(tmp_path):
    entries = [
        BankEntry(case_id="c1", model="m-a", answer="Paris.", answer_ok=True),
        BankEntry(case_id="c1", model="m-b", answer="London.", answer_ok=False),
    ]
    path = write_bank(entries, tmp_path / "bank.jsonl")
    bank = load_bank(path)
    assert bank[("c1", "m-a")].answer_ok is True
    assert bank[("c1", "m-b")].answer_ok is False


def test_load_bank_rejects_duplicate_key(tmp_path):
    path = tmp_path / "bank.jsonl"
    entry = {"case_id": "c1", "model": "m-a", "answer": "x", "answer_ok": True}
    path.write_text(
        "\n".join([json.dumps(entry), json.dumps(entry)]),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="duplicate"):
        load_bank(path)


def test_apply_bank_joins_by_routed_model():
    bank = {
        ("c1", "m-a"): BankEntry(
            case_id="c1", model="m-a", answer="ok", answer_ok=True
        ),
        ("c1", "m-b"): BankEntry(
            case_id="c1", model="m-b", answer="bad", answer_ok=False
        ),
    }
    rows = [
        {"case_id": "c1", "routed": "m-a"},
        {"case_id": "c1", "routed": "m-b"},
        {"case_id": "c2", "routed": "m-a"},  # no bank entry
    ]
    merged, stats = apply_bank(rows, bank)
    assert merged[0]["answer_ok"] is True
    assert merged[1]["answer_ok"] is False
    assert (
        merged[2]["answer_ok"] is None
    )  # evidence-first: missing entry stays no-evidence
    assert stats == {"joined": 2, "no_bank_entry": 1}


def _real_mcq_available() -> bool:
    # the repo's src/ carries training.model_eval; make it importable for the
    # skipif probe and the test itself (no-op when the layout differs)
    repo_src = Path(__file__).resolve().parents[3] / "src"
    if repo_src.is_dir() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    try:
        import training.model_eval.mcq_judge  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _real_mcq_available(),
    reason="src/training not importable from this environment",
)
def test_grade_mcq_with_real_judge():
    # verified (I): mcq_judge is importable as a plain function module
    from training.model_eval.mcq_judge import extract_letter  # noqa: F401

    case = {"grader": "mcq", "expected": "B", "case_id": "c2"}
    assert grade("Answer: B", case) is True
    assert grade("Answer: A", case) is False
