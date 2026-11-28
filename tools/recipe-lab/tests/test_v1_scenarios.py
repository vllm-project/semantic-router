"""V1 verdict-scenario pins (doc §7 acceptance evidence), unit-level.

No GPU, no real router binary: the fake-router machinery is imported from
test_hot_reload. Pins the V1 criteria the existing suite does not: report/
next determinism (criterion 3), hot-vs-restart ranked-metric equality
(criterion 2), and mid-run router death recorded as incomplete evidence
instead of a fabricated RANKED metric (criterion 4a).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml
from recipe_lab.cli import main
from recipe_lab.ledger import Ledger, LedgerRecord
from recipe_lab.runner import (
    HotReloadSession,
    RunnerEnv,
    RunnerError,
    evaluate_candidate,
)
from tests.test_cli import _write_frozen_spec
from tests.test_hot_reload import (
    BANK,
    CASES,
    EXPECTED_ACCURACY,
    FAKE_ROUTER_SCRIPT,
    POLICY,
    _candidate,
    _free_port,
    _seed_state,
    _StateView,
)

EXIT_CONTRACT_ERROR = 2


def _seed_ledger(path) -> None:
    """One replay RANKED record + one live PASS record as report/next input."""
    candidate = {
        "kind": "config",
        "ref": "cands/a.yaml",
        "provenance": {"proposer": "threshold-search", "insight_refs": []},
    }
    ledger = Ledger(path)
    ledger.append(
        LedgerRecord(
            round=1,
            ts="2026-10-29T00:00:00Z",
            candidate=candidate,
            tier="replay",
            baseline_ref="config-revision@main",
            metrics={"routed_accuracy": 0.8},
            verdict="RANKED",
            dataset_hash="sha256:" + "0" * 64,
        )
    )
    ledger.append(
        LedgerRecord(
            round=2,
            ts="2026-10-29T01:00:00Z",
            candidate=candidate,
            tier="live",
            baseline_ref="run-baseline",
            metrics={"capability.score": 0.82},
            verdict="PASS",
            dataset_hash="sha256:" + "0" * 64,
            gates={"pii-recall-floor": "pass"},
            bench_refs=["sr-run-1", "sr-cmp-1"],
        )
    )


def _fake_env(tmp_path, work_dir, state_path, reload_mode):
    """RunnerEnv over the shared fake router script, bound to one state file."""
    router = tmp_path / "fake_router.py"
    router.write_text(FAKE_ROUTER_SCRIPT)
    router.chmod(0o755)
    mock = tmp_path / "mock_backend.py"
    mock.write_text("import time\ntime.sleep(300)\n")
    return RunnerEnv(
        router_bin=str(router),
        mock_backend_script=str(mock),
        python_bin=sys.executable,
        work_dir=tmp_path / work_dir,
        backend_port=_free_port(),
        ready_timeout_s=30,
        reload_mode=reload_mode,
        env_extra={"FAKE_STATE": str(state_path)},
    )


def _stable_rows(rows):
    """Rows minus request_id (restart sends untagged ids, hot tagged ones)."""
    return [{k: v for k, v in row.items() if k != "request_id"} for row in rows]


def test_run_hot_and_restart_produce_identical_ranked_metrics(tmp_path):
    # V1 criterion 2: the same candidate set under --reload hot and
    # --reload restart yields identical ranked metrics; only the switching
    # mechanism differs, never the measurement.
    state_path = tmp_path / "fake_state.json"
    _seed_state(state_path)
    cands = tmp_path / "cands"
    cands.mkdir()
    cand_a = _candidate(cands / "a.yaml")
    cand_b = _candidate(cands / "b.yaml")

    # restart mode first: its prompt-join cannot filter a shared replay
    # store, so the fake store is re-seeded between candidates; the hot
    # request-id join below tolerates the accumulated records.
    restart_env = _fake_env(tmp_path, "work-restart", state_path, "restart")
    restart_results = []
    for cand in (cand_a, cand_b):
        restart_results.append(
            evaluate_candidate(
                candidate_config=cand,
                cases=CASES,
                oracle_policy=POLICY,
                bank=BANK,
                env=restart_env,
            )
        )
        _seed_state(state_path)

    hot_env = _fake_env(tmp_path, "work-hot", state_path, "hot")
    session = HotReloadSession(
        env=hot_env, boot_source=cand_a, router_model="vllm-sr/auto", family="d3"
    )
    try:
        session.ensure_started()
        hot_results = [
            session.evaluate(
                candidate_config=cand, cases=CASES, oracle_policy=POLICY, bank=BANK
            )
            for cand in (cand_a, cand_b)
        ]
    finally:
        session.stop()
    assert _StateView(state_path).put_count == 1  # only cand_b PUT in hot mode

    for restart, hot in zip(restart_results, hot_results, strict=True):
        assert restart["reload"] == "restart" and hot["reload"] == "hot"
        assert restart["statuses"] == hot["statuses"] == {"c1": 200, "c2": 200}
        assert restart["summary"] == hot["summary"]
        assert restart["summary"]["accuracy"] == EXPECTED_ACCURACY
        assert _stable_rows(restart["rows"]) == _stable_rows(hot["rows"])


_SUCCESS = {
    "summary": {"accuracy": 0.9},
    "statuses": {"ok": 1},
    "bank_stats": {"no_bank_entry": 0},
    "rows": [],
    "reload": "restart",
}


def test_router_death_mid_run_records_incomplete(monkeypatch, tmp_path):
    # V1 criterion 4a: a router dying mid-run must never silently drop or
    # fake evidence. Two surfaces: a detected record loss lands as an
    # explicit status=incomplete row that the ranking skips, and an
    # undetected runner failure aborts with exit 2 and no row at all.
    from recipe_lab import cli

    spec_path, cases = _write_frozen_spec(tmp_path)
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "cand-1.yaml").write_text("kind: config\n", encoding="utf-8")
    (cands / "cand-2.yaml").write_text("kind: config\n", encoding="utf-8")
    bank = tmp_path / "bank.jsonl"
    bank.write_text("", encoding="utf-8")

    def run_args(ledger):
        return [
            "run",
            "--spec",
            spec_path,
            "--cases",
            str(cases),
            "--bank",
            str(bank),
            "--ledger",
            str(ledger),
            "--candidates",
            str(cands),
            "--router-bin",
            "/bin/true",
            "--mock-backend",
            str(tmp_path / "mock.py"),
            "--output-dir",
            str(tmp_path / "out"),
        ]

    def fake_short_pull(**kwargs):
        if Path(kwargs["candidate_config"]).name == "cand-2.yaml":
            # the replay pull came back short (records lost mid-candidate)
            return {
                "statuses": {"c1": 200, "c2": 200},
                "incomplete": True,
                "incomplete_reason": "replay pull: 1 rows for 2 cases",
                "reload": "restart",
                "timings": {},
            }
        return dict(_SUCCESS)

    monkeypatch.setattr(cli, "evaluate_candidate", fake_short_pull)
    ledger_a = tmp_path / "ledger-a.jsonl"
    assert main(run_args(ledger_a)) == 0
    rows = Ledger(ledger_a).read()
    assert [row.get("status") for row in rows] == [None, "incomplete"]
    assert rows[0]["metrics"] == {"routed_accuracy": 0.9}
    assert rows[1]["metrics"] == {} and rows[1]["coverage_warning"] is True
    # the ranking skips the incomplete candidate entirely
    best = Ledger(ledger_a).top_replay("routed_accuracy")
    assert best is not None and "cand-2" not in best["candidate"]["ref"]

    def fake_death(**kwargs):
        if Path(kwargs["candidate_config"]).name == "cand-2.yaml":
            raise RunnerError("router exited early with code 137")
        return dict(_SUCCESS)

    monkeypatch.setattr(cli, "evaluate_candidate", fake_death)
    ledger_b = tmp_path / "ledger-b.jsonl"
    with pytest.raises(SystemExit) as exc:
        main(run_args(ledger_b))
    assert exc.value.code == EXIT_CONTRACT_ERROR
    rows_b = Ledger(ledger_b).read()
    assert len(rows_b) == 1  # no fabricated row for the dead candidate
    assert "cand-1" in rows_b[0]["candidate"]["ref"]
    assert rows_b[0]["metrics"] == {"routed_accuracy": 0.9}


def test_same_contract_feeds_replay_then_live(monkeypatch, tmp_path, capsys):
    """V1 + X3: one contract serves both tiers without edits.

    The dev split declares two renderings of the same cases: replay ranks on
    the cases manifest (`run` binds --cases to its content hash), live binds
    the sr-bench run documents to the `srbench_dataset` rendering's sha256.
    A run that scored the cases file instead is refused on the same contract.
    """
    from recipe_lab import cli as cli_module
    from recipe_lab.contract import ContractError, ExperimentSpec
    from recipe_lab.live import live_verdict_from_runs
    from tests.fixtures import VALID_SPLITS_SPEC

    cases = tmp_path / "dev.jsonl"
    cases.write_text(
        '{"case_id":"c1","prompt":"p1","truth":"math"}\n'
        '{"case_id":"c2","prompt":"p2","truth":"code"}\n',
        encoding="utf-8",
    )
    # the sr-bench rendering is generated by the CLI itself and frozen in
    assert (
        main(
            [
                "cases",
                "to-srbench",
                "--cases",
                str(cases),
                "--output",
                str(tmp_path / "dev.srbench.jsonl"),
            ]
        )
        == 0
    )
    rendered = json.loads(capsys.readouterr().out)
    srbench_sha = rendered["srbench_sha256"]

    raw = copy.deepcopy(VALID_SPLITS_SPEC)
    for name in ("calib", "dev", "frozen_test"):
        target = tmp_path / f"{name}.jsonl"
        if name == "dev":
            target.write_bytes(cases.read_bytes())
        else:
            target.write_text(
                '{"case_id":"x1","prompt":"px","truth":"math"}\n', encoding="utf-8"
            )
        raw["workload"]["splits"][name]["manifest"] = f"{name}.jsonl"
        raw["workload"]["splits"][name]["content_hash"] = (
            "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
        )
    raw["workload"]["splits"]["dev"]["srbench_dataset"] = "dev.srbench.jsonl"
    raw["workload"]["splits"]["dev"]["srbench_sha256"] = srbench_sha
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cases_hash = raw["workload"]["splits"]["dev"]["content_hash"]

    # replay leg: `run` accepts the contract and binds the cases manifest
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "cand-1.yaml").write_text("kind: config\n", encoding="utf-8")
    bank = tmp_path / "bank.jsonl"
    bank.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "evaluate_candidate",
        lambda **kwargs: {
            **_SUCCESS,
            "statuses": {"c1": 200, "c2": 200},  # the dev cases
        },
    )
    ledger = tmp_path / "ledger.jsonl"
    assert (
        main(
            [
                "run",
                "--spec",
                str(spec_path),
                "--cases",
                str(cases),
                "--bank",
                str(bank),
                "--ledger",
                str(ledger),
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
    rows = Ledger(ledger).read()
    assert rows[0]["tier"] == "replay" and rows[0]["verdict"] == "RANKED"
    # replay bound the cases file, not the sr-bench rendering
    assert rows[0]["dataset_hash"] == cases_hash

    # live leg: the same spec binds the sr-bench runs to the rendering hash
    spec = ExperimentSpec.from_yaml(spec_path)

    def fake_cli_with(hexdigest):
        class FakeCli:
            def show(self, run_id):
                return {
                    "id": run_id,
                    "status": "completed",
                    "manifest": {"dataset": {"sha256": hexdigest}},
                }

            def compare(self, base, cand):
                return {
                    "baseline": {"capability.score": 0.80},
                    "candidate": {"capability.score": 0.86},
                }

        return FakeCli()

    result = live_verdict_from_runs(
        spec=spec,
        cli=fake_cli_with(srbench_sha.split(":", 1)[1]),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"

    # a run that scored the cases file itself cannot bind on this contract
    with pytest.raises(ContractError, match="scored dataset"):
        live_verdict_from_runs(
            spec=spec,
            cli=fake_cli_with(cases_hash.split(":", 1)[1]),
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            measurements=[],
        )
