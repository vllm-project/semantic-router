"""CLI tests: stage-2 contract gates (A4/A1/A5/B) and the Sub-issue 5 CLI audit.

Audit contract (see tools/agent/skills/recipe-lab-loop/references/command-reference.md):
stdout of every command is a single JSON document (allow_nan=False, UTF-8);
progress goes to stderr; exit codes are 0 success, 1 FAIL verdict
(live-verdict only), 2 usage/contract/tool error.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml
from recipe_lab.cli import build_parser, main, verdict_exit_code
from tests.fixtures import VALID_SPEC, VALID_WORKLOADS_SPEC

EXIT_CONTRACT_ERROR = 2
EXPECTED_TOTAL_CASES = 24


def _write_frozen_spec(tmp_path):
    """Spec YAML whose workload hash covers tmp_path/cases.jsonl bytes."""
    cases = tmp_path / "cases.jsonl"
    cases.write_bytes(b'{"case_id":"c1","prompt":"p","truth":"math"}\n')
    spec = dict(VALID_SPEC)
    spec["workload"] = {
        "manifest": "cases.jsonl",
        "content_hash": "sha256:" + hashlib.sha256(cases.read_bytes()).hexdigest(),
        "profile": "quick",
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path), cases


def _write_workloads_spec(tmp_path):
    """M5 spec with two tune workloads and one transfer workload; the split
    manifests are real files whose hashes cover their bytes."""
    spec = copy.deepcopy(VALID_WORKLOADS_SPEC)
    datasets = {
        "calib.jsonl": b'{"case_id":"c1","prompt":"pc"}\n',
        "dev.jsonl": b'{"case_id":"d1","prompt":"pd"}\n',
        "frozen_test.jsonl": b'{"case_id":"f1","prompt":"pf"}\n',
        "office.jsonl": b'{"case_id":"o1","prompt":"po"}\n',
        "gpqa.jsonl": b'{"case_id":"g1","prompt":"pg"}\n',
    }
    for name, payload in datasets.items():
        (tmp_path / name).write_bytes(payload)
        for entry in spec["workloads"]:
            for split in entry["splits"].values():
                if split["manifest"] == name:
                    split["content_hash"] = (
                        "sha256:" + hashlib.sha256(payload).hexdigest()
                    )
    spec_path = tmp_path / "spec-workloads.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def test_plan_accepts_the_v3_multi_workload_contract(tmp_path, capsys):
    # M5 completion criterion: the V3-shaped contract loads and passes plan
    spec_path = _write_workloads_spec(tmp_path)
    assert main(["plan", "--spec", spec_path]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert [w["name"] for w in parsed["workloads"]] == [
        "mmlu-pro",
        "office",
        "gpqa-transfer",
    ]


def test_attribute_honors_workload_selection(tmp_path, capsys):
    spec_path = _write_workloads_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    assert (
        main(
            [
                "attribute",
                "--spec",
                spec_path,
                "--records",
                str(records),
                "--split",
                "dev",
                "--workload",
                "office",
            ]
        )
        == 0
    )


def test_replay_and_propose_reject_transfer_workloads(tmp_path, capsys):
    spec_path = _write_workloads_spec(tmp_path)
    assert (
        main(
            [
                "run",
                "--spec",
                spec_path,
                "--workload",
                "gpqa-transfer",
                "--cases",
                str(tmp_path / "gpqa.jsonl"),
                "--bank",
                str(tmp_path / "bank.jsonl"),
                "--candidates",
                str(tmp_path),
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--output-dir",
                str(tmp_path / "out"),
                "--router-bin",
                "/bin/true",
                "--mock-backend",
                str(tmp_path / "mock.py"),
            ]
        )
        == EXIT_CONTRACT_ERROR
    )
    assert "role 'transfer'" in capsys.readouterr().err
    assert (
        main(
            [
                "propose-thresholds",
                "--spec",
                spec_path,
                "--workload",
                "gpqa-transfer",
                "--base-config",
                str(tmp_path / "base.yaml"),
                "--signal",
                "decision:difficulty",
                "--output-dir",
                str(tmp_path / "cands"),
            ]
        )
        == EXIT_CONTRACT_ERROR
    )
    assert "role 'transfer'" in capsys.readouterr().err


def test_verify_and_live_verdict_reject_transfer_without_final(tmp_path, capsys):
    spec_path = _write_workloads_spec(tmp_path)
    assert (
        main(
            [
                "verify",
                "--spec",
                spec_path,
                "--workload",
                "gpqa-transfer",
                "--baseline-run-id",
                "run-baseline",
                "--candidate",
                str(tmp_path / "candidate.json"),
                "--candidate-targets",
                "model-a",
            ]
        )
        == EXIT_CONTRACT_ERROR
    )
    assert "role 'transfer'" in capsys.readouterr().err
    assert (
        main(
            [
                "live-verdict",
                "--spec",
                spec_path,
                "--workload",
                "gpqa-transfer",
                "--baseline-run-id",
                "run-baseline",
                "--candidate-run-id",
                "run-candidate",
            ]
        )
        == EXIT_CONTRACT_ERROR
    )
    assert "role 'transfer'" in capsys.readouterr().err


def test_ledger_records_carry_the_workload_field(tmp_path):
    from recipe_lab.ledger import Ledger, LedgerRecord

    ledger = Ledger(tmp_path / "ledger.jsonl")
    stored = ledger.append(
        LedgerRecord(
            round=1,
            ts="2026-01-01T00:00:00Z",
            candidate={"kind": "config", "ref": "c1"},
            tier="replay",
            baseline_ref="base",
            metrics={"routed_accuracy": 0.7},
            verdict="RANKED",
            dataset_hash="sha256:" + "0" * 64,
            split="dev",
            workload="office",
        )
    )
    assert stored["workload"] == "office"
    reloaded = Ledger(tmp_path / "ledger.jsonl").read()[0]
    assert reloaded["workload"] == "office"


def test_hash_command_prints_content_hash(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_bytes(b'{"case_id":"c1"}\n')
    assert main(["hash", "--file", str(cases)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert (
        parsed["content_hash"]
        == "sha256:" + hashlib.sha256(b'{"case_id":"c1"}\n').hexdigest()
    )


def test_hash_command_missing_file_exits_2(tmp_path, capsys):
    assert main(["hash", "--file", str(tmp_path / "nope.jsonl")]) == EXIT_CONTRACT_ERROR
    assert "cannot read dataset file" in capsys.readouterr().err


def test_attribute_cases_hash_mismatch_exits_2(tmp_path, capsys):
    spec_path, _ = _write_frozen_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "case_id": "c1",
                "truth": "math",
                "matched_decision": "math",
                "routed": "local/model-math",
                "answer_ok": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    drifted = tmp_path / "drifted.jsonl"
    drifted.write_bytes(b'{"case_id":"c9","prompt":"p","truth":"math"}\n')
    assert (
        main(
            [
                "attribute",
                "--spec",
                spec_path,
                "--records",
                str(records),
                "--cases",
                str(drifted),
            ]
        )
        == EXIT_CONTRACT_ERROR
    )
    assert "cases hash mismatch" in capsys.readouterr().err


def test_attribute_missing_manifest_exits_2(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    cases.unlink()  # the workload manifest itself is gone
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    assert (
        main(["attribute", "--spec", spec_path, "--records", str(records)])
        == EXIT_CONTRACT_ERROR
    )
    assert "dataset manifest not found" in capsys.readouterr().err


def test_attribute_cases_hash_match_passes(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "case_id": "c1",
                "truth": "math",
                "matched_decision": "math",
                "routed": "local/model-math",
                "answer_ok": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "attribute",
                "--spec",
                spec_path,
                "--records",
                str(records),
                "--cases",
                str(cases),
            ]
        )
        == 0
    )


# ---- A1: replay_metric contract + ledger top --spec ----


def _replay_record(ref: str, metric: str, value: float) -> dict:
    return {
        "round": 1,
        "ts": "2026-10-29T00:00:00Z",
        "candidate": {
            "kind": "config",
            "ref": ref,
            "change_summary": "s",
            "provenance": {"proposer": "p", "insight_refs": []},
        },
        "tier": "replay",
        "baseline_ref": "base@main",
        "metrics": {metric: value},
        "verdict": "RANKED",
        "dataset_hash": "sha256:" + "0" * 64,
        "reasons": [],
        "gates": None,
        "bench_refs": [],
        "attribution_delta": None,
        "attribution_ref": None,
        "coverage_warning": False,
        "notes": "",
    }


def _write_a1_spec(tmp_path) -> str:
    spec = dict(VALID_SPEC)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def test_ledger_top_spec_uses_replay_metric(tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(_replay_record("a.yaml", "routed_accuracy", 0.80))
        + "\n"
        + json.dumps(_replay_record("b.yaml", "routed_accuracy", 0.91))
        + "\n",
        encoding="utf-8",
    )
    spec_path = _write_a1_spec(tmp_path)
    assert main(["ledger", "top", "--file", str(ledger), "--spec", spec_path]) == 0
    assert json.loads(capsys.readouterr().out)["candidate"]["ref"] == "b.yaml"


def test_ledger_top_metric_key_and_spec_are_exclusive(tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "ledger",
                "top",
                "--file",
                str(ledger),
                "--metric-key",
                "accuracy",
                "--spec",
                "s.yaml",
            ]
        )
    assert exc.value.code == EXIT_CONTRACT_ERROR  # argparse usage error
    with pytest.raises(SystemExit) as exc2:  # neither given
        main(["ledger", "top", "--file", str(ledger)])
    assert exc2.value.code == EXIT_CONTRACT_ERROR
    capsys.readouterr()


def test_run_round_beyond_budget_exits_2(tmp_path, capsys):
    # A5: budget.rounds is a hard cap; the check fires before any stack work
    spec_path, cases = _write_frozen_spec(tmp_path)
    rc = main(
        [
            "run",
            "--spec",
            spec_path,
            "--round",
            "6",
            "--cases",
            str(cases),
            "--bank",
            str(tmp_path / "bank.json"),
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
            "--candidates",
            str(tmp_path / "cands"),
            "--router-bin",
            "/bin/true",
            "--mock-backend",
            str(tmp_path / "mock.py"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == EXIT_CONTRACT_ERROR
    assert "budget exhausted" in capsys.readouterr().err


def test_run_baseline_incomplete_capture_surfaces_the_reason(monkeypatch, tmp_path):
    # review follow-up: the baseline path read result["summary"] directly,
    # so a partial baseline capture died as a bare KeyError. The run must
    # surface the capture reason instead, and write no ledger records -
    # the candidates' deltas would be silently null.
    from recipe_lab import cli

    spec_path, cases = _write_frozen_spec(tmp_path)
    bank = tmp_path / "bank.jsonl"
    bank.write_text("", encoding="utf-8")
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "cand-1.yaml").write_text("kind: config\n", encoding="utf-8")

    def fake_evaluate(**kwargs):
        return {
            "statuses": {"c1": 200},
            "join_stats": {"matched": 1},
            "timings": {},
            "reload": "restart",
            "incomplete": True,
            "incomplete_reason": (
                "replay records for the restart run: 1 of 2 cases joined"
            ),
        }

    monkeypatch.setattr(cli, "evaluate_candidate", fake_evaluate)
    ledger_path = tmp_path / "ledger.jsonl"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--spec",
                spec_path,
                "--round",
                "1",
                "--cases",
                str(cases),
                "--bank",
                str(bank),
                "--baseline-config",
                str(tmp_path / "baseline.yaml"),
                "--ledger",
                str(ledger_path),
                "--candidates",
                str(cands),
                "--router-bin",
                "/bin/true",
                "--mock-backend",
                str(tmp_path / "mock.py"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert "baseline replay capture incomplete" in str(exc.value)
    assert "1 of 2 cases joined" in str(exc.value)
    assert not ledger_path.exists()


def test_run_records_declared_gates_as_not_evaluated(monkeypatch, tmp_path):
    # A3: replay never evaluates guards; every declared gate lands in the
    # replay record as explicitly not evaluated (never null).
    from recipe_lab import cli

    spec_path, cases = _write_frozen_spec(tmp_path)
    bank = tmp_path / "bank.jsonl"
    bank.write_text("", encoding="utf-8")
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "cand-1.yaml").write_text("kind: config\n", encoding="utf-8")

    def fake_evaluate(**kwargs):
        # the runner's summary key for routed_accuracy is "accuracy" (A1)
        return {
            "summary": {"accuracy": 0.9},
            "statuses": {"ok": 1},
            "bank_stats": {"no_bank_entry": 0},
        }

    monkeypatch.setattr(cli, "evaluate_candidate", fake_evaluate)
    ledger_path = tmp_path / "ledger.jsonl"
    assert (
        main(
            [
                "run",
                "--spec",
                spec_path,
                "--round",
                "1",
                "--cases",
                str(cases),
                "--bank",
                str(bank),
                "--ledger",
                str(ledger_path),
                "--candidates",
                str(cands),
                "--router-bin",
                "/bin/true",
                "--mock-backend",
                str(tmp_path / "mock.py"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["gates"] == {"pii-recall-floor": "not_evaluated"}
    assert rows[-1]["metrics"] == {"routed_accuracy": 0.9}


# ---- S5: run --proposer is a closed vocabulary plus agent:<id> ----


@pytest.mark.parametrize(
    "tag",
    [
        "threshold-search",
        "rule-threshold",
        "distribution-calibration",
        "bootstrap",
        "agent:claude-1",
        "agent:a.b_c-d",
    ],
)
def test_run_proposer_parses_valid_tags(tag):
    args = build_parser().parse_args(
        [
            "run",
            "--spec",
            "s.yaml",
            "--cases",
            "c.jsonl",
            "--bank",
            "b.jsonl",
            "--candidates",
            "x.yaml",
            "--ledger",
            "l.jsonl",
            "--output-dir",
            "o",
            "--router-bin",
            "r",
            "--mock-backend",
            "m",
            "--proposer",
            tag,
        ]
    )
    assert args.proposer == tag


@pytest.mark.parametrize(
    "tag", ["Agent1", "agent:", "agent:UPPER", "external", "agent:sp ace"]
)
def test_run_proposer_rejects_invalid_tags(tag, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--proposer", tag, "--spec", "s.yaml"])
    assert exc.value.code == EXIT_CONTRACT_ERROR
    assert "invalid --proposer" in capsys.readouterr().err


def test_run_agent_proposer_lands_in_ledger_provenance(monkeypatch, tmp_path):
    # S5: a self-authored candidate tagged agent:<id> keeps the tag verbatim.
    from recipe_lab import cli

    spec_path, cases = _write_frozen_spec(tmp_path)
    bank = tmp_path / "bank.jsonl"
    bank.write_text("", encoding="utf-8")
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "cand-1.yaml").write_text("kind: config\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "evaluate_candidate",
        lambda **kwargs: {
            "summary": {"accuracy": 0.9},
            "statuses": {"ok": 1},
            "bank_stats": {"no_bank_entry": 0},
        },
    )
    ledger_path = tmp_path / "ledger.jsonl"
    assert (
        main(
            [
                "run",
                "--spec",
                spec_path,
                "--round",
                "1",
                "--cases",
                str(cases),
                "--bank",
                str(bank),
                "--ledger",
                str(ledger_path),
                "--candidates",
                str(cands),
                "--proposer",
                "agent:claude-loop",
                "--router-bin",
                "/bin/true",
                "--mock-backend",
                str(tmp_path / "mock.py"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["candidate"]["provenance"]["proposer"] == "agent:claude-loop"


# ---- B: --split handling + the split command ----


def test_attribute_rejects_frozen_test_split(tmp_path, capsys):
    spec_path, _cases = _write_frozen_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    assert (
        main(
            [
                "attribute",
                "--spec",
                spec_path,
                "--records",
                str(records),
                "--split",
                "frozen_test",
            ]
        )
        == EXIT_CONTRACT_ERROR
    )
    assert "frozen_test is quarantined" in capsys.readouterr().err


def test_attribute_rejects_calib_on_legacy_workload(tmp_path, capsys):
    spec_path, _ = _write_frozen_spec(tmp_path)  # legacy single-dataset form
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    assert (
        main(
            [
                "attribute",
                "--spec",
                spec_path,
                "--records",
                str(records),
                "--split",
                "calib",
            ]
        )
        == EXIT_CONTRACT_ERROR
    )
    assert "unknown workload split 'calib'" in capsys.readouterr().err


def test_attribute_dev_split_prints_legacy_warning(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "case_id": "c1",
                "truth": "math",
                "matched_decision": "math",
                "routed": "local/model-math",
                "answer_ok": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "attribute",
                "--spec",
                spec_path,
                "--records",
                str(records),
                "--cases",
                str(cases),
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert "legacy single-dataset form" in capsys.readouterr().err


def test_split_command_outputs_spec_ready_block(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    rows = []
    for i in range(24):
        rows.append(
            json.dumps(
                {
                    "case_id": f"c{i}",
                    "prompt": f"prompt {i}",
                    "truth": "math",
                    "source_id": f"src{i % 6}",
                }
            )
        )
    cases.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out_dir = tmp_path / "slices"
    assert main(["split", "--cases", str(cases), "--output-dir", str(out_dir)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["split_rule"] == "group by source_id; seed 42; 40/30/30"
    assert sorted(parsed["splits"]) == ["calib", "dev", "frozen_test"]
    for name, spec_block in parsed["splits"].items():
        assert spec_block["manifest"] == f"{name}.jsonl"
        assert spec_block["content_hash"].startswith("sha256:")
        assert (out_dir / f"{name}.jsonl").is_file()
    # totals match: no case lost or duplicated
    total = sum(
        sum(1 for _ in (out_dir / f"{name}.jsonl").open(encoding="utf-8") if _.strip())
        for name in parsed["splits"]
    )
    assert total == EXPECTED_TOTAL_CASES


# ---- X3: cases to-srbench (deterministic sr-bench rendering) ----


def test_cases_to_srbench_command_is_deterministic(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"case_id":"c1","prompt":"p1","truth":"math"}\n'
        '{"case_id":"c2","prompt":"p2","truth":"code"}\n',
        encoding="utf-8",
    )

    def argv(out):
        return ["cases", "to-srbench", "--cases", str(cases), "--output", str(out)]

    assert main(argv(tmp_path / "a.jsonl")) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["cases_count"] == 2
    assert first["srbench_sha256"].startswith("sha256:")
    # same cases => byte-identical output (the hash is freezable in a contract)
    assert main(argv(tmp_path / "b.jsonl")) == 0
    capsys.readouterr()
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert (
        first["srbench_sha256"]
        == "sha256:" + hashlib.sha256((tmp_path / "a.jsonl").read_bytes()).hexdigest()
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "answer": "math",
            "benchmark": "recipe-lab",
            "id": "c1",
            "messages": [{"content": "p1", "role": "user"}],
        },
        {
            "answer": "code",
            "benchmark": "recipe-lab",
            "id": "c2",
            "messages": [{"content": "p2", "role": "user"}],
        },
    ]


def test_cases_to_srbench_rejects_signal_fact_truth(tmp_path):
    # signal-fact truths have no sr-bench answer to grade against
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"case_id":"c1","prompt":"p1","truth":{"signals":{"domain":"code"}}}\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "cases",
                "to-srbench",
                "--cases",
                str(cases),
                "--output",
                str(tmp_path / "d.jsonl"),
            ]
        )
        == 2
    )


# ---- Sub-issue 5 CLI audit: exit-code contract and structured stdout ----


def _write_spec(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(VALID_SPEC), encoding="utf-8")
    return str(spec)


def test_verdict_exit_code_mapping():
    assert verdict_exit_code("PASS") == 0
    assert verdict_exit_code("FAIL") == 1
    assert verdict_exit_code("INCONCLUSIVE") == EXIT_CONTRACT_ERROR


@pytest.mark.parametrize(
    "verdict,expected", [("PASS", 0), ("FAIL", 1), ("INCONCLUSIVE", 2)]
)
def test_verify_exit_code_follows_verdict(
    tmp_path, capsys, monkeypatch, verdict, expected
):
    # I: verify shares the live-verdict exit-code contract (0/1/2).
    import recipe_lab.cli as cli_module

    spec_path, _ = _write_frozen_spec(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"kind": "config", "ref": "cand-1", "srbench_targets": ["tgt-1"]}),
        encoding="utf-8",
    )
    fake = {
        "verdict": verdict,
        "metrics": {"candidate": 0.7, "baseline": 0.8},
        "reasons": [],
        "gates": [],
        "constraints": [],
        "bench_refs": [],
    }
    monkeypatch.setattr(cli_module, "verify_live", lambda **kwargs: fake)
    monkeypatch.setattr(cli_module, "SrbenchCli", lambda **kwargs: object())
    assert (
        main(
            [
                "verify",
                "--spec",
                spec_path,
                "--baseline-run-id",
                "run-b",
                "--candidate",
                str(candidate),
                "--work-dir",
                str(tmp_path / "work"),
            ]
        )
        == expected
    )
    assert json.loads(capsys.readouterr().out)["verdict"] == verdict


def test_plan_prints_single_json_document(tmp_path, capsys):
    spec = _write_spec(tmp_path)
    assert main(["plan", "--spec", spec]) == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)  # exactly one JSON document, no extra text
    assert parsed["api_version"] == "recipe-lab/v1alpha1"
    assert parsed["objective"]["metric_key"] == "capability.score"


def test_missing_spec_exits_2_with_stderr_message(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["plan", "--spec", str(tmp_path / "missing.yaml")])
    assert exc.value.code == EXIT_CONTRACT_ERROR
    assert "contract error" in capsys.readouterr().err


def test_argparse_usage_error_exits_2():
    with pytest.raises(SystemExit) as exc:
        main(["no-such-command"])
    assert exc.value.code == EXIT_CONTRACT_ERROR


def _top_level_parser_map() -> dict:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("build_parser() registers no subcommands")


def _documented_invocations() -> list:
    """(doc, args) for every recipe-lab invocation the docs teach."""
    root = Path(__file__).resolve().parents[3]
    docs = {
        "README": (root / "tools/recipe-lab/README.md").read_text(encoding="utf-8"),
        "command-reference.md": (
            root / "tools/agent/skills/recipe-lab-loop/references/command-reference.md"
        ).read_text(encoding="utf-8"),
    }
    found = []
    for doc, text in docs.items():
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("recipe-lab "):
                found.append((doc, stripped[len("recipe-lab ") :]))
        for m in re.finditer(r"`recipe-lab\s+([^`\n]+)`", text):
            found.append((doc, m.group(1).strip()))
    return found
