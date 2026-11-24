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


def test_observe_lanes_keeps_separate_configs_for_splits_and_retries(
    tmp_path, monkeypatch
):
    from recipe_lab import cli, observe_lanes

    spec_path = Path(_write_workloads_spec(tmp_path))
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["template"] = {"bundle": "mom-v1", "recipe": "balance"}
    spec["models"] = [{"name": "test-model", "endpoint": "http://127.0.0.1:8001/v1"}]
    spec["providers_config"] = "providers.yaml"
    (tmp_path / "providers.yaml").write_text("version: v0.3\n", encoding="utf-8")
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    template = {
        "bundle": "mom-v1",
        "recipe": "balance",
        "sha256": "abc",
        "catalog_version": "latest",
        "lanes": {"fast": {"immediate": False, "minimum_candidates": 1}},
    }
    monkeypatch.setattr(cli, "template_info", lambda **kwargs: template)

    def exclusive_init(argv, **kwargs):
        output = Path(argv[argv.index("--output") + 1])
        with output.open("x", encoding="utf-8") as f:
            f.write("version: v0.3\n")
        return {}

    monkeypatch.setattr(
        cli,
        "build_observation_config",
        lambda **kwargs: observe_lanes.build_observation_config(
            **kwargs, run_json=exclusive_init
        ),
    )
    monkeypatch.setattr(
        observe_lanes, "probe_records", lambda *args, **kwargs: ([], {}, {})
    )
    out = tmp_path / "lanes"
    configs = []
    for split in ("calib", "dev", "dev"):
        assert (
            main(
                [
                    "observe-lanes",
                    "--spec",
                    str(spec_path),
                    "--workload",
                    "mmlu-pro",
                    "--split",
                    split,
                    "--out",
                    str(out),
                    "--router-bin",
                    "stub",
                    "--mock-backend",
                    "stub",
                ]
            )
            == 0
        )
        summary = json.loads(
            (out / f"lanes-{split}-summary.json").read_text(encoding="utf-8")
        )
        configs.append(Path(summary["observation_config"]))
    assert len(set(configs)) == len(configs)
    assert all(config.is_file() for config in configs)
    for name in (
        "observe-config.yaml",
        "bindings-observe.yaml",
        "providers-observe.yaml",
    ):
        assert (out / name).read_bytes() == (configs[0].parent / name).read_bytes()


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
                "--output-dir",
                str(tmp_path / "attr"),
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
                "--output-dir",
                str(tmp_path / "attr"),
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
        "bindings",
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
    # The tag is validated by the argument parser, so the refusal is a receipt.
    assert "invalid --proposer" in json.loads(capsys.readouterr().out)["reason"]


@pytest.mark.parametrize(
    ("proposer", "override", "expected_model"),
    [
        ("agent:claude-loop", None, "vllm-sr/auto"),
        ("bindings", None, "vllm-sr/recipe-lab"),
        ("bindings", "custom", "custom"),
    ],
)
def test_run_targets_proposer_entrypoint_and_records_provenance(
    monkeypatch, tmp_path, proposer, override, expected_model
):
    from recipe_lab import cli

    spec_path, cases = _write_frozen_spec(tmp_path)
    bank = tmp_path / "bank.jsonl"
    bank.write_text("", encoding="utf-8")
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "cand-1.yaml").write_text("kind: config\n", encoding="utf-8")

    def fake_evaluate(**kwargs):
        assert kwargs["router_model"] == expected_model
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
                "--proposer",
                proposer,
                *(["--router-model", override] if override is not None else []),
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
    assert rows[-1]["candidate"]["provenance"]["proposer"] == proposer


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


def test_argparse_usage_error_exits_2(capsys):
    # A usage error is a recipe-lab-cli-error-v1 receipt on stdout: the agent
    # always has a document to parse and nothing lands on stderr.
    with pytest.raises(SystemExit) as exc:
        main(["no-such-command"])
    assert exc.value.code == EXIT_CONTRACT_ERROR
    captured = capsys.readouterr()
    assert captured.err == ""
    receipt = json.loads(captured.out)
    assert receipt["schema"] == "recipe-lab-cli-error-v1"
    assert receipt["verdict"] == "INCONCLUSIVE"
    assert "invalid choice: 'no-such-command'" in receipt["reason"]


def test_missing_subcommand_is_a_json_receipt_too(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == EXIT_CONTRACT_ERROR
    assert "command" in json.loads(capsys.readouterr().out)["reason"]


def test_record_draft_without_pass_exits_2(tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "record-draft",
                "--ledger",
                str(ledger),
                "--output",
                str(tmp_path / "d.md"),
            ]
        )
    assert exc.value.code == EXIT_CONTRACT_ERROR
    assert "no PASS live record" in capsys.readouterr().err


# ---- A6: every subcommand is documented in README and the skill reference ----


def _top_level_subcommands() -> list[str]:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    raise AssertionError("build_parser() registers no subcommands")


def _doc_texts() -> dict:
    """Docs whose CLI surface must mirror build_parser() (S4, both ways)."""
    root = Path(__file__).resolve().parents[3]
    skill = root / "tools/agent/skills/recipe-lab-loop"
    texts = {
        "README": (root / "tools/recipe-lab/README.md").read_text(encoding="utf-8"),
        "SKILL.md": (skill / "SKILL.md").read_text(encoding="utf-8"),
    }
    for ref in sorted((skill / "references").glob("*.md")):
        texts[f"references/{ref.name}"] = ref.read_text(encoding="utf-8")
    return texts


def test_every_subcommand_is_documented():
    # the command-surface docs must name every subcommand; prose references
    # (SKILL.md steps, evidence/red-line notes) legitimately mention only the
    # commands their flow uses, so they join the reverse check below only
    docs = {
        k: v
        for k, v in _doc_texts().items()
        if k in ("README", "references/command-reference.md")
    }
    missing = []
    for name in _top_level_subcommands():
        pattern = rf"(?<![\w-]){re.escape(name)}(?![\w-])"
        for doc, text in docs.items():
            if not re.search(pattern, text):
                missing.append(f"{name}: {doc}")
    assert not missing, "undocumented subcommands: " + ", ".join(missing)


# ---- A6 reverse: every documented invocation exists in the CLI ----


def _top_level_parser_map() -> dict:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("build_parser() registers no subcommands")


def _documented_invocations() -> list:
    """(doc, args) for every recipe-lab invocation the docs teach."""
    docs = _doc_texts()
    found = []
    for doc, text in docs.items():
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("recipe-lab "):
                found.append((doc, stripped[len("recipe-lab ") :]))
        for m in re.finditer(r"`recipe-lab\s+([^`\n]+)`", text):
            found.append((doc, m.group(1).strip()))
    return found


def test_every_documented_invocation_exists():
    parsers = _top_level_parser_map()
    unknown = []
    for doc, invocation in _documented_invocations():
        tokens = invocation.split()
        if not tokens:
            continue
        name = tokens[0].rstrip(".")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            continue  # flags, placeholders and prose are not subcommands
        if name not in parsers:
            unknown.append(f"{doc}: recipe-lab {name}")
            continue
        sub_names = set()
        for action in parsers[name]._actions:
            if isinstance(action, argparse._SubParsersAction):
                sub_names = set(action.choices)
        if (
            len(tokens) > 1
            and re.fullmatch(r"[a-z][a-z0-9|]*", tokens[1])
            and sub_names
        ):
            for action in tokens[1].split("|"):
                if action not in sub_names:
                    unknown.append(f"{doc}: recipe-lab {name} {action}")
    assert not unknown, "documented but nonexistent invocations: " + ", ".join(unknown)


# ---- C7: the verify model_eval driver ----


def _write_model_eval_spec(tmp_path):
    """Spec with one model_eval-driver workload whose dev split is a real
    cases manifest with its declared model_eval tasks rendering (Y3) and a
    C2 labels file."""
    dev = tmp_path / "dev.jsonl"
    dev.write_bytes(b'{"case_id":"t1","prompt":"q","truth":"A"}\n')
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_bytes(b'{"task_id":"t1","prompt_user":"q","answer":"A"}\n')
    (tmp_path / "labels.jsonl").write_bytes(b'{"case_id":"t1","router_label":"math"}\n')
    spec = copy.deepcopy(VALID_SPEC)
    spec.pop("workload", None)
    spec["workloads"] = [
        {
            "name": "cold",
            "profile": "quick",
            "driver": "model_eval",
            "splits": {
                "dev": {
                    "manifest": "dev.jsonl",
                    "content_hash": "sha256:"
                    + hashlib.sha256(dev.read_bytes()).hexdigest(),
                    "model_eval_tasks": "tasks.jsonl",
                    "model_eval_tasks_sha256": "sha256:"
                    + hashlib.sha256(tasks.read_bytes()).hexdigest(),
                }
            },
        }
    ]
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


_FAKE_MODEL_EVAL_RESULT = {
    "verdict": "PASS",
    "reasons": [],
    "improvement_pct": 100.0,
    "metrics": {"baseline": 0.5, "candidate": 1.0},
    "gates": {},
    "constraints": [],
    "undeclared_measurements": [],
    "bench_refs": ["meval:tasks:sha256:ab", "meval:baseline-rep0"],
    "baseline_run_id": None,
    "trials": {},
    "abba_summary": {},
    "preview": None,
    "paired": None,
    "evidence_quality": {},
    "references": {},
    "pareto": [],
    "distribution": {"STRONG": 0.6, "WEAK": 0.4},
    "driver": "model_eval",
    "tasks_hash": "sha256:ab",
    "config_sha256": "sha256:cd",
    "planned_cases": 1,
}


def _model_eval_verify_args(spec_path, tmp_path, candidate):
    return [
        "verify",
        "--spec",
        spec_path,
        "--candidate",
        str(candidate),
        "--tasks",
        str(tmp_path / "tasks.jsonl"),
        "--labels",
        str(tmp_path / "labels.jsonl"),
        "--candidate-config",
        str(tmp_path / "candidate-config.yaml"),
        "--baseline-router-endpoint",
        "http://127.0.0.1:9001/v1",
        "--candidate-router-endpoint",
        "http://127.0.0.1:9002/v1",
        "--model-eval-dir",
        str(tmp_path / "scripts"),
        "--repetitions",
        "1",
        "--work-dir",
        str(tmp_path / "work"),
    ]


@pytest.mark.parametrize(
    ("provenance", "override", "expected_model"),
    [
        ({}, None, "vllm-sr/auto"),
        ({"proposer": "bindings"}, None, "vllm-sr/recipe-lab"),
        ({"proposer": "bindings"}, "custom", "custom"),
    ],
)
def test_verify_model_eval_driver_runs_without_srbench(
    tmp_path, capsys, monkeypatch, provenance, override, expected_model
):
    # C7: the driver branch skips the sr-bench targets entirely - the
    # candidate JSON needs no srbench_targets and SrbenchCli is never built.
    # Y3: --tasks is hash-checked against the contract's declared rendering
    # and --labels is passed through for the in-driver category rewrite.
    import recipe_lab.cli as cli_module

    spec_path = _write_model_eval_spec(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"kind": "config", "ref": "cand-1", "provenance": provenance}),
        encoding="utf-8",
    )
    captured: dict = {}

    def fake_verify(**kwargs):
        captured.update(kwargs)
        return dict(_FAKE_MODEL_EVAL_RESULT)

    monkeypatch.setattr(cli_module, "verify_live_model_eval", fake_verify)
    assert (
        main(
            [
                *_model_eval_verify_args(spec_path, tmp_path, candidate),
                *(["--router-model", override] if override is not None else []),
                "--candidate-management-endpoint",
                "http://127.0.0.1:8080",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "PASS"
    assert captured["tasks_path"] == str(tmp_path / "tasks.jsonl")
    assert captured["labels_path"] == str(tmp_path / "labels.jsonl")
    assert captured["baseline_endpoint"] == "http://127.0.0.1:9001/v1"
    assert captured["candidate_endpoint"] == "http://127.0.0.1:9002/v1"
    assert captured["management_endpoint"] == "http://127.0.0.1:8080"
    assert captured["router_model"] == expected_model


def test_verify_model_eval_driver_appends_ledger(tmp_path, capsys, monkeypatch):
    import recipe_lab.cli as cli_module

    spec_path = _write_model_eval_spec(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"kind": "config", "ref": "cand-1"}), encoding="utf-8"
    )
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(
        cli_module,
        "verify_live_model_eval",
        lambda **kwargs: dict(_FAKE_MODEL_EVAL_RESULT),
    )
    assert (
        main(
            [
                *_model_eval_verify_args(spec_path, tmp_path, candidate),
                "--ledger",
                str(ledger),
                "--round",
                "2",
            ]
        )
        == 0
    )
    records = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["tier"] == "live"
    assert record["verdict"] == "PASS"
    assert record["round"] == 2
    assert record["baseline_ref"] is None  # no sr-bench run behind this evidence
    # Y4: the ledger's candidate names the config the verdict graded
    assert record["candidate"]["config_sha256"] == "sha256:cd"
    # without --final the record carries no promotion flag
    assert "final" not in record
    assert "repetitions=1" in record["notes"]
    # the model_eval driver's distribution rides the result, not a preview
    assert "routing_distribution=" in record["notes"]


def test_verify_model_eval_rejects_srbench_baseline(tmp_path, capsys):
    spec_path = _write_model_eval_spec(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"kind": "config", "ref": "cand-1"}), encoding="utf-8"
    )
    rc = main(
        [
            *_model_eval_verify_args(spec_path, tmp_path, candidate),
            "--baseline-run-id",
            "run-b",
        ]
    )
    assert rc == EXIT_CONTRACT_ERROR
    assert "takes no sr-bench baseline" in capsys.readouterr().err


def test_verify_model_eval_requires_tasks_and_flags(tmp_path, capsys):
    spec_path = _write_model_eval_spec(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"kind": "config", "ref": "cand-1"}), encoding="utf-8"
    )
    base = ["verify", "--spec", spec_path, "--candidate", str(candidate)]
    assert main([*base, "--model-eval-dir", str(tmp_path / "s")]) == EXIT_CONTRACT_ERROR
    assert "--tasks" in capsys.readouterr().err
    err = main(
        [
            *base,
            "--tasks",
            str(tmp_path / "tasks.jsonl"),
            "--baseline-router-endpoint",
            "http://ep1/v1",
            "--candidate-router-endpoint",
            "http://ep2/v1",
        ]
    )
    assert err == EXIT_CONTRACT_ERROR
    assert "--labels" in capsys.readouterr().err
    err = main(
        [
            *base,
            "--tasks",
            str(tmp_path / "tasks.jsonl"),
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--baseline-router-endpoint",
            "http://ep1/v1",
            "--candidate-router-endpoint",
            "http://ep2/v1",
        ]
    )
    assert err == EXIT_CONTRACT_ERROR
    assert "--model-eval-dir" in capsys.readouterr().err


def _write_agentic_srbench_spec(tmp_path):
    """Spec whose single workload is an sr-bench agentic workload (Z4)."""
    dev = tmp_path / "dev.jsonl"
    dev.write_bytes(b'{"case_id":"t1","prompt":"q","truth":"A"}\n')
    spec = copy.deepcopy(VALID_SPEC)
    spec.pop("workload", None)
    spec["workloads"] = [
        {
            "name": "agentic-wl",
            "kind": "agentic",
            "benchmark": "terminal-bench-2.1",
            "profile": "quick",
            "splits": {
                "dev": {
                    "manifest": "dev.jsonl",
                    "content_hash": "sha256:"
                    + hashlib.sha256(dev.read_bytes()).hexdigest(),
                }
            },
        }
    ]
    spec_path = tmp_path / "spec-agentic.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def _agentic_verify_args(spec_path, tmp_path):
    return [
        "verify",
        "--spec",
        spec_path,
        "--candidate",
        str(tmp_path / "candidate.json"),
        "--repetitions",
        "1",
        "--work-dir",
        str(tmp_path / "work"),
    ]


def test_verify_agentic_requires_arm_management_endpoints(
    tmp_path, capsys, monkeypatch
):
    # Z4: the replay-sourced stability gates have exactly one evidence
    # source - the arms' own router replay records - so an agentic verify
    # without the management endpoints fails before anything is spent on
    # runs, and the interrupted gate's default baseline pin additionally
    # needs the baseline arm's router.
    import recipe_lab.cli as cli_module

    spec_path = _write_agentic_srbench_spec(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"kind": "config", "srbench_targets": ["mom-router"]}),
        encoding="utf-8",
    )
    rc = main(_agentic_verify_args(spec_path, tmp_path))
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "--candidate-management-endpoint" in err

    rc = main(
        [
            *_agentic_verify_args(spec_path, tmp_path),
            "--candidate-management-endpoint",
            "http://127.0.0.1:8081",
        ]
    )
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "--baseline-management-endpoint" in err
    assert "max_tool_loop_interrupted" in err

    # both endpoints clear the CLI gate and reach verify_live verbatim
    captured: dict = {}

    def fake_verify(**kwargs):
        captured.update(kwargs)
        return {"verdict": "PASS"}

    monkeypatch.setattr(cli_module, "verify_live", fake_verify)
    assert (
        main(
            [
                *_agentic_verify_args(spec_path, tmp_path),
                "--candidate-management-endpoint",
                "http://127.0.0.1:8081",
                "--baseline-management-endpoint",
                "http://127.0.0.1:8080",
            ]
        )
        == 0
    )
    assert captured["candidate_management_endpoint"] == "http://127.0.0.1:8081"
    assert captured["baseline_management_endpoint"] == "http://127.0.0.1:8080"


def test_verify_model_eval_rejects_unbound_tasks(tmp_path, capsys):
    # Y3: --tasks must be byte-identical to the contract's declared
    # rendering - a different task file is refused even when it is a
    # valid tasks JSONL.
    spec_path = _write_model_eval_spec(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"kind": "config", "ref": "cand-1"}), encoding="utf-8"
    )
    other = tmp_path / "other-tasks.jsonl"
    other.write_bytes(b'{"task_id":"t1","prompt_user":"DIFFERENT","answer":"A"}\n')
    rc = main(
        [
            "verify",
            "--spec",
            spec_path,
            "--candidate",
            str(candidate),
            "--tasks",
            str(other),
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--baseline-router-endpoint",
            "http://ep1/v1",
            "--candidate-router-endpoint",
            "http://ep2/v1",
            "--model-eval-dir",
            str(tmp_path / "s"),
        ]
    )
    assert rc == EXIT_CONTRACT_ERROR
    assert "--tasks hash mismatch" in capsys.readouterr().err


def test_verify_model_eval_requires_declared_rendering(tmp_path, capsys):
    # Y3: without the contract declaration the driver refuses to run at
    # all - no hash to bind --tasks to.
    dev = tmp_path / "dev.jsonl"
    dev.write_bytes(b'{"case_id":"t1","prompt":"q","truth":"A"}\n')
    spec = copy.deepcopy(VALID_SPEC)
    spec.pop("workload", None)
    spec["workloads"] = [
        {
            "name": "cold",
            "profile": "quick",
            "driver": "model_eval",
            "splits": {
                "dev": {
                    "manifest": "dev.jsonl",
                    "content_hash": "sha256:"
                    + hashlib.sha256(dev.read_bytes()).hexdigest(),
                }
            },
        }
    ]
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"kind": "config", "ref": "cand-1"}), encoding="utf-8"
    )
    rc = main(
        [
            "verify",
            "--spec",
            str(spec_path),
            "--candidate",
            str(candidate),
            "--tasks",
            str(tmp_path / "tasks.jsonl"),
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--baseline-router-endpoint",
            "http://ep1/v1",
            "--candidate-router-endpoint",
            "http://ep2/v1",
            "--model-eval-dir",
            str(tmp_path / "s"),
        ]
    )
    assert rc == EXIT_CONTRACT_ERROR
    assert "must declare model_eval_tasks" in capsys.readouterr().err
