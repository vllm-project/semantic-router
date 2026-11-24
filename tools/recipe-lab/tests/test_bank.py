"""Answer bank tests: build/grade/apply and the evidence-first join."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from recipe_lab.bank import BankEntry, apply_bank, grade, load_bank, write_bank
from recipe_lab.contract import API_VERSION, ContractError


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


# ---- C6: judged.jsonl import into the bank ----


def test_import_judged_converts_rows():
    from recipe_lab.bank import import_judged

    judged = [
        {
            "task_id": "q1",
            "is_correct": True,
            "completion": "Answer: B.",
            "completion_tokens": 42,
        },
        {
            "task_id": "q2",
            "is_correct": False,
            "completion": "Answer: A.",
            "completion_tokens": None,
        },
    ]
    entries = import_judged("mock/small", judged)
    assert [(e.case_id, e.model) for e in entries] == [
        ("q1", "mock/small"),
        ("q2", "mock/small"),
    ]
    assert entries[0].answer_ok is True and entries[0].tokens == 42
    assert entries[1].answer_ok is False and entries[1].tokens is None
    # entries round-trip through the bank format with the token evidence
    assert entries[0].to_dict() == {
        "case_id": "q1",
        "model": "mock/small",
        "answer": "Answer: B.",
        "answer_ok": True,
        "tokens": 42,
    }
    assert "tokens" not in entries[1].to_dict()


def test_import_judged_rejects_duplicate_task_id():
    from recipe_lab.bank import import_judged

    judged = [
        {"task_id": "q1", "is_correct": True, "completion": "x"},
        {"task_id": "q1", "is_correct": False, "completion": "y"},
    ]
    with pytest.raises(ContractError, match="duplicate task_id 'q1'"):
        import_judged("m", judged)


def test_import_judged_rejects_missing_or_unjudged_rows():
    from recipe_lab.bank import import_judged

    with pytest.raises(ContractError, match="without a task_id"):
        import_judged("m", [{"is_correct": True}])
    with pytest.raises(ContractError, match="is_correct must be a boolean"):
        import_judged("m", [{"task_id": "q1", "is_correct": None}])


def test_bank_tokens_roundtrip_and_legacy_rows(tmp_path):
    entries = [
        BankEntry(case_id="c1", model="m-a", answer="ok", answer_ok=True, tokens=17),
        BankEntry(case_id="c1", model="m-b", answer="bad", answer_ok=False),
    ]
    path = write_bank(entries, tmp_path / "bank.jsonl")
    bank = load_bank(path)
    assert bank[("c1", "m-a")].tokens == 17
    assert bank[("c1", "m-b")].tokens is None
    # legacy bank rows (no tokens field) still load
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps({"case_id": "c9", "model": "m-a", "answer": "x", "answer_ok": True}),
        encoding="utf-8",
    )
    assert load_bank(legacy)[("c9", "m-a")].tokens is None
    # malformed tokens are rejected, never silently dropped
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "case_id": "c1",
                "model": "m-a",
                "answer": "x",
                "answer_ok": True,
                "tokens": -3,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="tokens"):
        load_bank(bad)


def test_apply_bank_carries_tokens():
    bank = {
        ("c1", "m-a"): BankEntry(
            case_id="c1", model="m-a", answer="ok", answer_ok=True, tokens=33
        ),
        ("c1", "m-b"): BankEntry(
            case_id="c1", model="m-b", answer="bad", answer_ok=False
        ),
    }
    merged, _stats = apply_bank(
        [{"case_id": "c1", "routed": "m-a"}, {"case_id": "c1", "routed": "m-b"}], bank
    )
    assert merged[0]["tokens"] == 33
    assert "tokens" not in merged[1]


# ---- C6: `bank import` CLI ----


def _write_import_spec(tmp_path: Path) -> str:
    spec = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "capability.score",
            "direction": "maximize",
            "min_improvement_pct": 2.0,
        },
        "workload": {
            "manifest": "dev.jsonl",
            "content_hash": "sha256:" + "0" * 64,
            "profile": "quick",
        },
        "attribution": {"oracle_policy": {"math": "mock/large"}},
        "models": [
            {"name": "mock/small", "endpoint": "http://127.0.0.1:8001/v1"},
            {"name": "mock/large", "endpoint": "http://127.0.0.1:8002/v1"},
        ],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def _write_profile_dev(tmp_path: Path) -> Path:
    from recipe_lab.profile import results_dirname

    profile_dir = tmp_path / "profile"
    fixtures = {
        "mock/small": [
            {
                "task_id": "q1",
                "is_correct": True,
                "completion": "Answer: B.",
                "completion_tokens": 42,
            },
            {
                "task_id": "q2",
                "is_correct": False,
                "completion": "Answer: A.",
                "completion_tokens": 40,
            },
        ],
        "mock/large": [
            {
                "task_id": "q1",
                "is_correct": True,
                "completion": "Answer: B.",
                "completion_tokens": 400,
            },
            {
                "task_id": "q2",
                "is_correct": True,
                "completion": "Answer: B.",
                "completion_tokens": 410,
            },
        ],
    }
    for alias, rows in fixtures.items():
        rdir = profile_dir / "dev" / "results" / results_dirname(alias)
        rdir.mkdir(parents=True)
        (rdir / "judged.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
    return profile_dir


def test_bank_import_cli_end_to_end(tmp_path, capsys):
    from recipe_lab.cli import main

    bank_path = tmp_path / "bank.jsonl"
    rc = main(
        [
            "bank",
            "import",
            "--spec",
            _write_import_spec(tmp_path),
            "--profile-dir",
            str(_write_profile_dev(tmp_path)),
            "--output",
            str(bank_path),
        ]
    )
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["entries"] == 4
    assert parsed["models"]["mock/small"] == {"cases": 2, "correct": 1}
    assert parsed["models"]["mock/large"] == {"cases": 2, "correct": 2}
    assert parsed["split"] == "dev"
    bank = load_bank(bank_path)
    assert bank[("q1", "mock/small")].answer_ok is True
    assert bank[("q1", "mock/small")].tokens == 42
    assert bank[("q2", "mock/large")].answer_ok is True


def test_bank_import_cli_refuses_missing_judged(tmp_path, capsys):
    from recipe_lab.cli import main

    profile_dir = _write_profile_dev(tmp_path)
    # drop one model's judged file: the import must fail loudly, not skip
    (profile_dir / "dev" / "results" / "mock_large_direct" / "judged.jsonl").unlink()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "bank",
                "import",
                "--spec",
                _write_import_spec(tmp_path),
                "--profile-dir",
                str(profile_dir),
                "--output",
                str(tmp_path / "bank.jsonl"),
            ]
        )
    assert exc.value.code == 2
    assert "mock/large" in capsys.readouterr().err
    assert not (tmp_path / "bank.jsonl").exists()
