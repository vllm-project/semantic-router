"""Live-tier orchestration tests: guard measurements, metric pairing, verdict flow."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from recipe_lab.cli import main
from recipe_lab.contract import ContractError, CostBasis, ExperimentSpec, References
from recipe_lab.ledger import Ledger, LedgerRecord
from recipe_lab.live import (
    _merge_paired_stats,
    _pair_metrics,
    _worst_rates,
    arm_cost,
    arm_evidence_quality,
    live_verdict_from_runs,
    measurements_from_rows,
    paired_stats_from_comparison,
    pareto_rows,
    reference_metric_values,
    resolve_gates,
    verify_live,
)
from recipe_lab.verdict import PairedStats, evaluate_verdict
from tests.fixtures import VALID_SPEC, VALID_SPLITS_SPEC
from tests.test_contract import _write_splits_dataset

# ---- A3: the contract is the only source of guard floors ----

# VALID_SPEC's legacy workload hash (tests.fixtures); every FakeCli run doc
# carries it so the M1 dataset binding passes unless a test overrides it.
_LEGACY_HASH = "sha256:" + "0" * 64
_RUN_DOC = {
    "id": "run-x",
    "status": "completed",
    "manifest": {"dataset": {"sha256": _LEGACY_HASH}},
}


def test_measurements_from_rows_rejects_floors_and_unknown_keys():
    # a floor inside the measurement file is a contract bypass
    with pytest.raises(ContractError, match="floor"):
        measurements_from_rows(
            [{"name": "pii", "candidate": 0.97, "baseline": 0.96, "floor": 0.95}]
        )
    with pytest.raises(ContractError, match="non-measurement keys"):
        measurements_from_rows([{"name": "pii", "candidate": 0.97, "metric": "g"}])
    with pytest.raises(ContractError, match="name"):
        measurements_from_rows([{"candidate": 0.97}])
    with pytest.raises(ContractError, match="candidate"):
        measurements_from_rows([{"name": "pii", "candidate": "high"}])


def test_resolve_gates_numeric_floor_pass_and_breach():
    from recipe_lab.contract import Gate

    gates = [Gate(name="pii-recall-floor", metric="guard.pii.recall", floor=0.95)]
    guards, _undeclared = resolve_gates(
        gates, [{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.5}]
    )
    assert guards[0].floor == 0.95 and guards[0].value == 0.97 and guards[0].passed
    guards, _ = resolve_gates(
        gates, [{"name": "pii-recall-floor", "candidate": 0.90, "baseline": 0.5}]
    )
    assert guards[0].passed is False


def test_resolve_gates_baseline_floor_uses_measured_baseline():
    gates = ExperimentSpec.from_dict(VALID_SPEC).gates
    guards, undeclared = resolve_gates(
        gates, [{"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}]
    )
    assert undeclared == []
    assert guards[0].floor == 0.96 and guards[0].value == 0.97 and guards[0].passed


def test_resolve_gates_missing_measurement_is_unevaluated():
    gates = ExperimentSpec.from_dict(VALID_SPEC).gates
    # no measurement at all
    guards, _ = resolve_gates(gates, [])
    assert not guards[0].evaluated
    # baseline floor without a measured baseline value
    guards, _ = resolve_gates(gates, [{"name": "pii-recall-floor", "candidate": 0.97}])
    assert not guards[0].evaluated
    # candidate value missing
    guards, _ = resolve_gates(gates, [{"name": "pii-recall-floor", "baseline": 0.96}])
    assert not guards[0].evaluated


def test_resolve_gates_reports_undeclared_without_veto():
    gates = ExperimentSpec.from_dict(VALID_SPEC).gates
    guards, undeclared = resolve_gates(
        gates,
        [
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96},
            {"name": "unheard-of", "candidate": 0.1, "baseline": 0.9},
        ],
    )
    assert undeclared == ["unheard-of"]
    assert guards[0].passed  # the declared gate still resolves normally


def test_pair_metrics_sectioned_payload():
    payload = {
        "baseline": {"m": {"capability.score": 0.80}},
        "candidate": {"capability.score": 0.86},
    }
    base, cand = _pair_metrics(payload, "capability.score", [])
    assert (base, cand) == (0.80, 0.86)
    # sr-bench compare's actual shape: *_targets arrays with accuracy leaves
    srbench_payload = {
        "baseline_targets": [{"id": "single", "accuracy": 1.0}],
        "candidate_targets": [{"id": "mom", "accuracy": 0.833}],
    }
    base, cand = _pair_metrics(srbench_payload, "accuracy", [])
    assert (base, cand) == (1.0, 0.833)


def test_pair_metrics_prefers_routing_arm_over_reference_lane():
    # a candidate run keeps its "no routing" reference single lane before the
    # routing arm; the comparison arm is the router (mom), not the reference
    payload = {
        "baseline_targets": [{"id": "single", "accuracy": 0.5}],
        "candidate_targets": [
            {"id": "single", "accuracy": 0.5},
            {"id": "mom", "accuracy": 0.6667},
        ],
    }
    incomplete: list[str] = []
    base, cand = _pair_metrics(payload, "accuracy", incomplete)
    assert (base, cand) == (0.5, 0.6667)
    assert incomplete == []
    # without a mom kind, the last target carrying the metric is the arm
    payload["candidate_targets"] = [
        {"id": "single", "accuracy": 0.5},
        {"id": "router", "accuracy": 0.7},
    ]
    base, cand = _pair_metrics(payload, "accuracy", [])
    assert (base, cand) == (0.5, 0.7)


def test_pair_metrics_flat_two_values():
    base, cand = _pair_metrics({"a": 0.80, "b": 0.86}, "x", [])
    assert base is None and cand is None  # different key, nothing matched
    incomplete: list[str] = []
    base, cand = _pair_metrics(
        {"capability.score": 0.80, "nested": {"capability.score": 0.86}},
        "capability.score",
        incomplete,
    )
    assert (base, cand) == (0.80, 0.86)
    assert incomplete == []


def test_pair_metrics_ambiguous_is_incomplete():
    incomplete: list[str] = []
    base, cand = _pair_metrics(
        {"capability.score": 0.8}, "capability.score", incomplete
    )
    assert base is None and cand is None
    assert "ambiguous" in incomplete[0]


def _spec():
    from recipe_lab.contract import ExperimentSpec

    return ExperimentSpec.from_dict(VALID_SPEC)


def test_live_verdict_pass_flow():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        # VALID_SPEC declares floor: baseline -> the measured baseline is the floor
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    assert result["gates"] == {"pii-recall-floor": "pass"}
    assert result["undeclared_measurements"] == []
    assert result["bench_refs"] == ["run-a", "run-b"]


def test_live_verdict_baseline_floor_breach_fails():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.90, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "FAIL"
    assert result["gates"] == {"pii-recall-floor": "fail"}
    assert "guard floor violated" in result["reasons"][0]


def test_live_verdict_missing_measurement_is_inconclusive():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["gates"] == {"pii-recall-floor": "incomplete"}
    assert "guard evaluation incomplete" in result["reasons"][0]


def test_live_verdict_undeclared_measurement_does_not_veto():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96},
            {"name": "extra", "candidate": 0.1},
        ],
    )
    assert result["verdict"] == "PASS"
    assert result["undeclared_measurements"] == ["extra"]
    assert any(
        "undeclared guard measurements" in reason for reason in result["reasons"]
    )


def test_live_verdict_empty_gates_is_explicit():
    spec_dict = dict(VALID_SPEC)
    spec_dict["gates"] = []

    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=ExperimentSpec.from_dict(spec_dict),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[{"name": "anything", "candidate": 0.5, "baseline": 0.4}],
    )
    assert result["gates"] == {}
    assert result["gates_note"] == "no guard declared"
    assert result["undeclared_measurements"] == ["anything"]


def test_live_verdict_ambiguous_is_inconclusive():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {"capability.score": 0.8}  # only one value -> ambiguous

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["gates"] == {"pii-recall-floor": "pass"}


# ---- A3: CLI rejects measurement floors with exit code 2 ----


def test_live_verdict_rejects_floor_in_measurement_file(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text(
        json.dumps(
            {
                "name": "pii-recall-floor",
                "candidate": 0.97,
                "baseline": 0.96,
                "floor": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "live-verdict",
                "--spec",
                spec_path,
                "--baseline-run-id",
                "run-a",
                "--candidate-run-id",
                "run-b",
                "--guard-measurements",
                str(measurements),
            ]
        )
        == 2
    )
    assert "floor" in capsys.readouterr().err


# ---- I: verify - sr-bench-driven AB/BA orchestration ----


def _verify_spec():
    spec_dict = copy.deepcopy(VALID_SPEC)
    spec_dict["objective"]["constraints"] = [
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 0.0,
        },
    ]
    return ExperimentSpec.from_dict(spec_dict)


class VerifyStub:
    """Scripted stand-in for SrbenchCli: deterministic run documents."""

    def __init__(
        self,
        *,
        preview_status: str = "completed",
        candidate_statuses: list[str] | None = None,
        baseline_statuses: list[str] | None = None,
        compare_values: dict | None = None,
        baseline_doc: dict | None = None,
        ref_statuses: dict[str, str] | None = None,
        registry: list[dict] | None = None,
    ):
        self.baseline_doc = baseline_doc or {
            "id": "run-base",
            "status": "completed",
            "manifest": {
                "manifest_id": "m1",
                "dataset": {"sha256": _LEGACY_HASH},
                "plan_sha256": "sha256:baseline-review",
            },
        }
        self.preview_status = preview_status
        self.candidate_statuses = list(candidate_statuses or [])
        self.baseline_statuses = list(baseline_statuses or [])
        self.compare_values = dict(compare_values or {})
        self.preview_report_payload = {
            "summary": {
                "targets": [
                    {"selected_models": {"local/model-a": 7, "local/model-b": 5}}
                ]
            }
        }
        # M4: single-target reference-arm runs, keyed by target alias
        self.ref_statuses = dict(ref_statuses or {})
        self.ref_reports: dict[str, dict] = {}
        self.ref_runs = 0
        # R2: operator-registered targets, verbatim from the registry
        self.registry = list(registry or [])
        self.events: list[tuple] = []
        self.manifest_runs = 0
        self.compares: list[tuple] = []
        self._cand = 0

    def show(self, run_id):
        self.events.append(("show", run_id))
        return self.baseline_doc

    def targets(self):
        self.events.append(("targets",))
        return [dict(entry) for entry in self.registry]

    def candidate_plan(
        self, baseline_run_id, *, targets, mode="live", name=None, hypothesis=""
    ):
        self.events.append(("plan", mode))
        if mode == "preview":
            return {"id": "run-preview", "status": self.preview_status}
        self._cand += 1
        status = (
            self.candidate_statuses.pop(0) if self.candidate_statuses else "completed"
        )
        return {"id": f"run-cand-{self._cand}", "status": status}

    def report(self, run_id, out_path):
        self.events.append(("report", run_id))
        if run_id.startswith("run-ref-"):
            alias = run_id[len("run-ref-") :]
            return self.ref_reports.get(alias) or {
                "summary": {"targets": [{"id": alias}]}
            }
        return self.preview_report_payload

    def run_manifest(self, manifest_path):
        # M4: a reference arm submits the frozen baseline manifest with a
        # single registry target; recognize it by its lone target alias
        raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        submitted = raw.get("targets") if isinstance(raw, dict) else None
        if (
            isinstance(submitted, list)
            and len(submitted) == 1
            and isinstance(submitted[0], dict)
            and submitted[0].get("id") in self.ref_statuses
        ):
            alias = submitted[0]["id"]
            self.ref_runs += 1
            self.events.append(("manifest-ref", alias))
            return {"id": f"run-ref-{alias}", "status": self.ref_statuses[alias]}
        self.manifest_runs += 1
        self.events.append(("manifest",))
        status = (
            self.baseline_statuses.pop(0) if self.baseline_statuses else "completed"
        )
        return {"id": f"run-base-trial-{self.manifest_runs}", "status": status}

    def compare(self, base, cand):
        self.compares.append((base, cand))
        return dict(self.compare_values)


def _compare_values(
    base: float, cand: float, spend: tuple[float, float] = (5.0, 4.8)
) -> dict:
    # default spend does not violate the cost-not-worse constraint (lte
    # baseline 5.0), so tests can target the metric/gate path in isolation
    return {
        "baseline": {"capability.score": base, "total_spend_usd": spend[0]},
        "candidate": {"capability.score": cand, "total_spend_usd": spend[1]},
    }


def test_verify_live_pass_flow(tmp_path):
    stub = VerifyStub(
        candidate_statuses=["completed", "completed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "PASS"
    assert result["metrics"] == {"baseline": 0.80, "candidate": 0.86}
    assert result["gates"] == {"pii-recall-floor": "pass"}
    assert result["preview"]["distribution"] == {"local/model-a": 7, "local/model-b": 5}
    # AB/BA: preview first, then each pair runs baseline and candidate arms
    # in alternating order (schedule: base0, cand0, cand1, base1)
    assert stub.events == [
        ("show", "run-base"),
        ("plan", "preview"),
        ("report", "run-preview"),
        ("manifest",),
        ("plan", "live"),
        ("plan", "live"),
        ("manifest",),
    ]
    assert stub.manifest_runs == 2
    assert len(stub.compares) == 2
    assert result["abba_summary"]["baseline"]["median"] == 0.80
    assert result["abba_summary"]["candidate"]["median"] == 0.86
    assert (
        len(result["bench_refs"]) == 1 + 1 + 4
    )  # baseline + preview + both arms of both reps
    assert "run-base" in result["bench_refs"] and "run-preview" in result["bench_refs"]
    # X8: each completed pair's raw compare payload stays on disk so the
    # merged paired statistics remain auditable against their sources
    for rep in range(2):
        payload = json.loads((tmp_path / f"compare-rep{rep}.json").read_text("utf-8"))
        assert payload == stub.compare_values


def test_verify_live_incomplete_pair_is_inconclusive(tmp_path):
    stub = VerifyStub(
        candidate_statuses=["completed", "failed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("pair 1" in reason for reason in result["reasons"])  # reps are 0-indexed
    # only the complete pair is compared; no numbers are invented for the
    # broken one - and only its payload is kept
    assert len(stub.compares) == 1
    assert (tmp_path / "compare-rep0.json").exists()
    assert not (tmp_path / "compare-rep1.json").exists()


def test_verify_live_pair_without_paired_stats_is_inconclusive(tmp_path):
    # review regression: both pairs completed but the second carried no
    # paired statistics, and the verify returned PASS on the first pair's
    # interval alone. With a pp bar (the paired-evidence contract) a pair
    # missing its stats makes the verdict INCONCLUSIVE instead.
    stub = VerifyStub(
        candidate_statuses=["completed", "completed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_paired_verify_payload(),
    )

    def compare(base, cand):
        stub.compares.append((base, cand))
        if len(stub.compares) == 1:
            return dict(stub.compare_values)  # rep0: paired evidence
        return _compare_values(0.80, 0.86)  # rep1: point estimates only

    stub.compare = compare
    result = verify_live(
        spec=_pp_spec(2.0, min_cases=60),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert any(
        "pair 1" in reason and "no paired statistics" in reason
        for reason in result["reasons"]
    )


def _paired_verify_payload() -> dict:
    """A verify-shaped compare payload with one paired comparisons[] entry."""
    return {
        "baseline": {"capability.score": 0.80, "total_spend_usd": 5.0},
        "candidate": {"capability.score": 0.86, "total_spend_usd": 4.8},
        "comparisons": [
            {
                "paired_cases": 60,
                "quality_delta": 0.05,
                "quality_delta_ci95": [0.03, 0.07],
                "quality_delta_bootstrap_ci95": [0.03, 0.07],
                "wins": 40,
                "losses": 10,
                "ties": 10,
            }
        ],
    }


def test_verify_live_baseline_arm_failure_is_inconclusive(tmp_path):
    stub = VerifyStub(
        baseline_statuses=["failed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("baseline ended 'failed'" in reason for reason in result["reasons"])


def test_verify_live_constraint_median_over_pairs(tmp_path):
    # spend medians 5.0 vs 5.75 across two pairs -> constraint violated -> FAIL
    stub = VerifyStub(
        candidate_statuses=["completed", "completed"],
        baseline_statuses=["completed", "completed"],
    )
    stub.compare_values_by_call = {
        0: _compare_values(0.80, 0.86, spend=(5.0, 5.5)),
        1: _compare_values(0.80, 0.86, spend=(5.0, 6.0)),
    }

    def compare(base, cand):
        index = len(stub.compares)
        stub.compares.append((base, cand))
        return dict(stub.compare_values_by_call[index])

    stub.compare = compare
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "FAIL"
    evidence = result["constraints"][0]
    assert evidence["baseline"] == 5.0 and evidence["candidate"] == 5.75
    assert evidence["status"] == "violated"
    assert any(
        "constraint cost-not-worse violated" in reason for reason in result["reasons"]
    )


def test_verify_live_constraint_unmeasured_pair_stays_unevaluated(tmp_path):
    # one pair broken -> the constraint is measured in 1 of 2 pairs -> None
    stub = VerifyStub(
        candidate_statuses=["completed", "failed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=2,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    evidence = result["constraints"][0]
    assert evidence["baseline"] is None and evidence["candidate"] is None
    assert any("measured in 1 of 2 pairs" in reason for reason in result["reasons"])


def test_verify_live_preview_failure_is_diagnostic_only(tmp_path):
    stub = VerifyStub(
        preview_status="failed", compare_values=_compare_values(0.80, 0.86)
    )
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=1,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "PASS"  # a failed preview produces no quality numbers
    assert any("preview" in reason for reason in result["reasons"])
    assert "distribution" not in result["preview"]


def test_verify_live_baseline_must_be_completed(tmp_path):
    stub = VerifyStub(
        baseline_doc={
            "id": "run-base",
            "status": "failed",
            "manifest": {"manifest_id": "m1"},
        }
    )
    with pytest.raises(ContractError, match="not completed"):
        verify_live(
            spec=_verify_spec(),
            cli=stub,
            baseline_run_id="run-base",
            targets=["mom-router"],
            repetitions=1,
            measurements=[],
            work_dir=tmp_path,
        )


def test_verify_live_baseline_must_expose_manifest(tmp_path):
    stub = VerifyStub(baseline_doc={"id": "run-base", "status": "completed"})
    with pytest.raises(ContractError, match="does not expose its frozen manifest"):
        verify_live(
            spec=_verify_spec(),
            cli=stub,
            baseline_run_id="run-base",
            targets=["mom-router"],
            repetitions=1,
            measurements=[],
            work_dir=tmp_path,
        )


def test_verify_live_needs_at_least_one_repetition(tmp_path):
    with pytest.raises(ContractError, match="at least one repetition"):
        verify_live(
            spec=_verify_spec(),
            cli=VerifyStub(),
            baseline_run_id="run-base",
            targets=["mom-router"],
            repetitions=0,
            measurements=[],
            work_dir=tmp_path,
        )


def test_verify_cli_requires_targets(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "kind": "config",
                "ref": "config-rev",
                "change_summary": "t",
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "verify",
                "--spec",
                spec_path,
                "--baseline-run-id",
                "run-a",
                "--candidate",
                str(candidate),
            ]
        )
        == 2
    )
    assert "MoM target ids" in capsys.readouterr().err


# ---- G2: --insight-refs on live-verdict ----


def _seed_replay_ledger(tmp_path) -> tuple[str, str]:
    """An empty ledger with one replay record; returns (ledger_path, record_id)."""
    from recipe_lab.ledger import Ledger, LedgerRecord

    ledger_path = tmp_path / "seed-ledger.jsonl"
    stored = Ledger(ledger_path).append(
        LedgerRecord(
            round=1,
            ts="2026-10-29T00:00:00Z",
            candidate={
                "kind": "config",
                "ref": "seed.yaml",
                "change_summary": "s",
                "provenance": {"proposer": "p", "insight_refs": []},
            },
            tier="replay",
            baseline_ref="base@main",
            metrics={"routed_accuracy": 0.9},
            verdict="RANKED",
            dataset_hash="sha256:" + "0" * 64,
        )
    )
    return str(ledger_path), stored["record_id"]


def _live_verdict_args(spec_path: str, ledger: str | None = None) -> list[str]:
    args = [
        "live-verdict",
        "--spec",
        spec_path,
        "--baseline-run-id",
        "run-a",
        "--candidate-run-id",
        "run-b",
    ]
    if ledger:
        args += ["--ledger", ledger]
    return args


def test_live_verdict_insight_refs_four_forms_merge_and_dedupe(
    monkeypatch, tmp_path, capsys
):
    from recipe_lab import cli

    spec_path = str(_write_splits_dataset(tmp_path))
    seed_ledger, record_id = _seed_replay_ledger(tmp_path)
    ledger_path = Path(seed_ledger)  # rec- refs resolve against --ledger itself
    attr_file = tmp_path / "attr.json"
    attr_file.write_text("{}", encoding="utf-8")
    diag_file = tmp_path / "diagnosis.json"
    diag_file.write_text("{}", encoding="utf-8")
    sidecar = tmp_path / "cand.json"
    sidecar.write_text(
        json.dumps(
            {
                "kind": "config",
                "ref": "cand-1.yaml",
                "change_summary": "s",
                "provenance": {"proposer": "agent", "insight_refs": ["bench:run-a"]},
            }
        ),
        encoding="utf-8",
    )
    attr_hash = hashlib.sha256(attr_file.read_bytes()).hexdigest()
    diag_hash = hashlib.sha256(diag_file.read_bytes()).hexdigest()
    refs = ",".join(
        [
            record_id,
            "bench:run-a",  # already on the sidecar -> deduped
            f"attr:{attr_file}@sha256:{attr_hash}",
            f"diag:{diag_file}@sha256:{diag_hash}#threshold.domain_code_complex",
        ]
    )
    monkeypatch.setattr(
        cli,
        "live_verdict_from_runs",
        lambda **kwargs: {
            "verdict": "PASS",
            "reasons": [],
            "improvement_pct": 1.0,
            "gates": {},
            "constraints": [],
            "bench_refs": ["run-a", "run-b"],
            "metrics": {"baseline": 0.5, "candidate": 0.6},
        },
    )
    args = [
        *_live_verdict_args(spec_path, str(ledger_path)),
        "--candidate",
        str(sidecar),
        "--insight-refs",
        refs,
    ]
    assert main(args) == 0
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["candidate"]["provenance"]["insight_refs"] == [
        "bench:run-a",
        record_id,
        f"attr:{attr_file}@sha256:{attr_hash}",
        f"diag:{diag_file}@sha256:{diag_hash}#threshold.domain_code_complex",
    ]


def test_live_verdict_unknown_record_ref_exits_2(tmp_path, capsys):
    seed_ledger, _ = _seed_replay_ledger(tmp_path)
    spec_path = str(_write_splits_dataset(tmp_path))
    rc = main(
        [
            *_live_verdict_args(spec_path, seed_ledger),
            "--insight-refs",
            "rec-000000000000",
        ]
    )
    assert rc == 2
    assert "not found in ledger" in capsys.readouterr().err


def test_live_verdict_hash_mismatch_exits_2(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    attr_file = tmp_path / "attr.json"
    attr_file.write_text("{}", encoding="utf-8")
    bad_hash = "sha256:" + "f" * 64
    rc = main(
        [
            *_live_verdict_args(spec_path),
            "--insight-refs",
            f"attr:{attr_file}@{bad_hash}",
        ]
    )
    assert rc == 2
    assert "hash mismatch" in capsys.readouterr().err


# ---- E: economic constraints (live part) ----


def _spec_with_constraints(**overrides) -> ExperimentSpec:
    spec_dict = copy.deepcopy(VALID_SPEC)
    spec_dict["objective"]["constraints"] = [
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 0.0,
        }
    ]
    spec_dict["objective"].update(overrides)
    return ExperimentSpec.from_dict(spec_dict)


def _compare_with_costs(candidate_cost):
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80, "total_spend_usd": 5.0},
                "candidate": {
                    "capability.score": 0.86,
                    "total_spend_usd": candidate_cost,
                },
            }

    return FakeCli()


def test_live_verdict_constraint_within_baseline_passes():
    result = live_verdict_from_runs(
        spec=_spec_with_constraints(),
        cli=_compare_with_costs(5.0),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    [constraint] = result["constraints"]
    assert constraint["status"] == "pass"
    assert constraint["bound_value"] == 5.0


def test_live_verdict_constraint_violated_fails():
    result = live_verdict_from_runs(
        spec=_spec_with_constraints(),
        cli=_compare_with_costs(6.0),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "FAIL"
    assert "constraint cost-not-worse violated" in result["reasons"][0]
    assert result["constraints"][0]["status"] == "violated"


def test_live_verdict_unknown_cost_is_inconclusive_not_zero():
    # verified: sr-bench serializes unknown usage as null; an unevaluable
    # constraint is missing evidence, never a violation (and never free)
    result = live_verdict_from_runs(
        spec=_spec_with_constraints(),
        cli=_compare_with_costs(None),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert "not evaluable" in result["reasons"][0]
    assert result["constraints"][0]["status"] == "not_evaluated"


def test_guard_violation_is_listed_before_constraint_violation():
    result = live_verdict_from_runs(
        spec=_spec_with_constraints(),
        cli=_compare_with_costs(6.0),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[{"name": "pii-recall-floor", "candidate": 0.5, "baseline": 0.96}],
    )
    assert result["verdict"] == "FAIL"
    assert result["reasons"][0].startswith("guard floor violated")
    assert any(
        "constraint cost-not-worse violated" in reason for reason in result["reasons"]
    )


def test_numeric_bound_and_tolerance_relax_in_permissive_direction():
    from recipe_lab.verdict import evaluate_verdict

    # tolerance 10% on an lte baseline bound of 5.0 -> 5.5 allowed
    evidence = [
        {
            "name": "c",
            "metric": "m",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 10.0,
            "baseline": 5.0,
            "candidate": 5.4,
        }
    ]
    result = evaluate_verdict(
        baseline_metric=0.8,
        candidate_metric=0.9,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        constraints=evidence,
    )
    assert result.verdict == "PASS" and result.constraints[0]["bound_value"] == 5.5
    # gte with 10% tolerance on a numeric bound of 0.9 -> 0.81 required
    evidence = [
        {
            "name": "c",
            "metric": "m",
            "op": "gte",
            "bound": 0.9,
            "tolerance_pct": 10.0,
            "baseline": 1.0,
            "candidate": 0.85,
        }
    ]
    result = evaluate_verdict(
        baseline_metric=0.8,
        candidate_metric=0.9,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        constraints=evidence,
    )
    assert result.verdict == "PASS" and result.constraints[0][
        "bound_value"
    ] == pytest.approx(0.81)


def test_live_cli_record_carries_constraint_evidence(monkeypatch, tmp_path, capsys):
    from recipe_lab import cli

    spec_path = _write_splits_dataset(tmp_path)
    spec_dict = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    spec_dict["objective"]["constraints"] = [
        {
            "name": "cost-not-worse",
            "metric": "total_spend_usd",
            "op": "lte",
            "bound": "baseline",
            "tolerance_pct": 0.0,
        }
    ]
    spec_path = tmp_path / "spec-constraints.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict), encoding="utf-8")

    def fake_verdict(**kwargs):
        return {
            "verdict": "PASS",
            "reasons": [],
            "gates": {"pii-recall-floor": "pass"},
            "constraints": [
                {
                    "name": "cost-not-worse",
                    "metric": "total_spend_usd",
                    "op": "lte",
                    "bound": "baseline",
                    "tolerance_pct": 0.0,
                    "baseline": 5.0,
                    "candidate": 5.0,
                    "bound_value": 5.0,
                    "status": "pass",
                }
            ],
            "bench_refs": ["run-a", "run-b"],
            "metrics": {"baseline": 0.5, "candidate": 0.6},
        }

    monkeypatch.setattr(cli, "live_verdict_from_runs", fake_verdict)
    ledger_path = tmp_path / "ledger.jsonl"
    assert main([*_live_verdict_args(str(spec_path), str(ledger_path)), "--final"]) == 0
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    constraint = rows[-1]["constraints"]["cost-not-worse"]
    assert constraint["status"] == "pass"
    assert constraint["candidate"] == 5.0


# ---- B: --split quarantine gate on the CLI ----


def test_live_verdict_rejects_frozen_test_without_final(tmp_path, capsys):
    spec_path = str(_write_splits_dataset(tmp_path))
    assert (
        main(
            [
                "live-verdict",
                "--spec",
                spec_path,
                "--baseline-run-id",
                "run-a",
                "--candidate-run-id",
                "run-b",
                "--split",
                "frozen_test",
            ]
        )
        == 2
    )
    assert "frozen_test is quarantined" in capsys.readouterr().err


def test_live_verdict_final_cites_frozen_test(monkeypatch, tmp_path, capsys):
    from recipe_lab import cli

    spec_path = str(_write_splits_dataset(tmp_path))

    def fake_verdict(**kwargs):
        return {
            "verdict": "PASS",
            "reasons": [],
            "gates": {},
            "constraints": [],
            "bench_refs": ["run-a", "run-b"],
            "metrics": {"baseline": 0.5, "candidate": 0.6},
        }

    monkeypatch.setattr(cli, "live_verdict_from_runs", fake_verdict)
    ledger_path = tmp_path / "ledger.jsonl"
    assert (
        main(
            [
                "live-verdict",
                "--spec",
                spec_path,
                "--baseline-run-id",
                "run-a",
                "--candidate-run-id",
                "run-b",
                "--split",
                "frozen_test",
                "--final",
                "--ledger",
                str(ledger_path),
            ]
        )
        == 0
    )
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["verdict"] == "PASS"
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["split"] == "frozen_test"
    expected = (
        "sha256:"
        + hashlib.sha256((tmp_path / "frozen_test.jsonl").read_bytes()).hexdigest()
    )
    assert rows[-1]["dataset_hash"] == expected


# ---- M1: live verdicts bind the runs to the contract's own frozen split ----


def test_live_verdict_rejects_run_from_other_dataset():
    class FakeCli:
        def show(self, run_id):
            return {
                "id": run_id,
                "status": "completed",
                "manifest": {"dataset": {"sha256": "f" * 64}},
            }

        def compare(self, base, cand):
            raise AssertionError("compare must not run on a dataset mismatch")

    with pytest.raises(ContractError) as excinfo:
        live_verdict_from_runs(
            spec=_spec(),
            cli=FakeCli(),
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            measurements=[],
        )
    assert "scored dataset" in str(excinfo.value)
    assert "baseline run run-a" in str(excinfo.value)


def test_live_verdict_rejects_run_without_dataset_identity():
    class FakeCli:
        def show(self, run_id):
            return {"id": run_id, "status": "completed", "manifest": {}}

        def compare(self, base, cand):
            raise AssertionError("compare must not run without a dataset identity")

    with pytest.raises(ContractError) as excinfo:
        live_verdict_from_runs(
            spec=_spec(),
            cli=FakeCli(),
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            measurements=[],
        )
    assert "no dataset sha256" in str(excinfo.value)


def test_live_verdict_accepts_bare_hexdigest_dataset_hash():
    # sr-bench records a bare hexdigest; the A4 sha256: prefix is normalized.
    class FakeCli:
        def show(self, run_id):
            return {
                "id": run_id,
                "status": "completed",
                "manifest": {"dataset": {"sha256": "0" * 64}},
            }

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    assert result["bench_refs"] == ["run-a", "run-b"]


# ---- X3: live binds the declared srbench_dataset hash when present ----


def _srbench_bound_spec():
    """The dev split declares the sr-bench rendering of the same cases."""
    from recipe_lab.contract import ExperimentSpec

    raw = copy.deepcopy(VALID_SPLITS_SPEC)
    raw["workload"]["splits"]["dev"]["srbench_dataset"] = "dev.srbench.jsonl"
    raw["workload"]["splits"]["dev"]["srbench_sha256"] = "sha256:" + "b" * 64
    return ExperimentSpec.from_dict(raw)


def test_live_verdict_binds_srbench_dataset_hash_when_declared():
    # X3: with srbench_dataset declared, the run documents must carry the
    # rendering's sha256 - the cases hash is replay's binding, not live's.
    srbench_hex = "b" * 64

    class FakeCli:
        def show(self, run_id):
            return {
                "id": run_id,
                "status": "completed",
                "manifest": {"dataset": {"sha256": srbench_hex}},
            }

        def compare(self, base, cand):
            return {
                "baseline": {"capability.score": 0.80},
                "candidate": {"capability.score": 0.86},
            }

    result = live_verdict_from_runs(
        spec=_srbench_bound_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"


def test_live_verdict_rejects_cases_hash_when_srbench_dataset_declared():
    # X3: a run that scored the cases file itself (the dev split's
    # content_hash, "sha256:" + "2"*64) cannot bind - the legacy fallback
    # must not mask the declared rendering hash.
    class FakeCli:
        def show(self, run_id):
            return {
                "id": run_id,
                "status": "completed",
                "manifest": {"dataset": {"sha256": "2" * 64}},
            }

        def compare(self, base, cand):
            raise AssertionError("compare must not run on a dataset mismatch")

    with pytest.raises(ContractError) as excinfo:
        live_verdict_from_runs(
            spec=_srbench_bound_spec(),
            cli=FakeCli(),
            baseline_run_id="run-a",
            candidate_run_id="run-b",
            measurements=[],
        )
    assert "scored dataset" in str(excinfo.value)
    assert "sha256:" + "b" * 64 in str(excinfo.value)


def test_verify_live_rejects_baseline_from_other_dataset(tmp_path):
    stub = VerifyStub(
        baseline_doc={
            "id": "run-base",
            "status": "completed",
            "manifest": {"dataset": {"sha256": "f" * 64}},
        },
    )
    with pytest.raises(ContractError) as excinfo:
        verify_live(
            spec=_verify_spec(),
            cli=stub,
            baseline_run_id="run-base",
            targets=["mom-router"],
            repetitions=1,
            measurements=[],
            work_dir=tmp_path,
        )
    assert "scored dataset" in str(excinfo.value)


def test_legacy_verdict_record_uses_null_split(tmp_path):
    # M1: a legacy single-dataset contract records split: null + the
    # legacy_workload marker, while a three-way contract keeps its split.
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    record = LedgerRecord(
        round=1,
        ts="2026-10-31T00:00:00Z",
        candidate={"kind": "config", "ref": "run-b"},
        tier="live",
        baseline_ref="run-a",
        metrics={"capability.score": 0.86},
        verdict="PASS",
        dataset_hash=_LEGACY_HASH,
        gates={},
        bench_refs=["run-a", "run-b"],
        split=None,
        legacy_workload=True,
    )
    ledger.append(record)
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0].get("split") is None
    assert rows[0]["legacy_workload"] is True
    roundtrip = LedgerRecord.from_dict(rows[0])
    assert roundtrip.legacy_workload is True and roundtrip.split is None


# ---- M2: paired statistics decide the verdict from the CI ----


def _pp_spec(pp: float, min_cases: int = 200) -> ExperimentSpec:
    """VALID_SPEC with the objective switched to the absolute pp bar (M2)."""
    objective = {
        k: v for k, v in VALID_SPEC["objective"].items() if k != "min_improvement_pct"
    }
    objective["min_improvement_pp"] = pp
    objective["min_cases"] = min_cases
    return ExperimentSpec.from_dict({**VALID_SPEC, "objective": objective})


def _paired_payload(
    *,
    paired_cases: int = 60,
    delta: float = 1 / 60,
    ci: tuple[float, float] = (-0.13, 0.16),
) -> dict:
    """J3-shaped compare payload: point estimates + one comparisons[] entry."""
    return {
        "baseline": {"capability.score": 17 / 60},
        "candidate": {"capability.score": 18 / 60},
        "comparisons": [
            {
                "paired_cases": paired_cases,
                "quality_delta": delta,
                "quality_delta_ci95": list(ci),
                "quality_delta_bootstrap_ci95": [-0.11, 0.14],
                "wins": 2,
                "losses": 1,
                "ties": paired_cases - 3,
            }
        ],
    }


def test_live_verdict_j3_regression_inconclusive():
    # J3 scored 17/60 -> 18/60 and PASSED on point estimates (+5.88%); under
    # the M2 minimum sample (default 200) the same evidence cannot conclude.
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return _paired_payload()

    result = live_verdict_from_runs(
        spec=_spec(),  # relative bar, default min_cases = 200
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert (
        "insufficient paired sample: 60 paired cases < min_cases 200"
        in result["reasons"][0]
    )
    assert result["paired"]["paired_cases"] == 60


def test_live_verdict_paired_pass_with_pp_bar():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return _paired_payload(paired_cases=400, delta=0.05, ci=(0.03, 0.07))

    result = live_verdict_from_runs(
        spec=_pp_spec(2.0),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    assert "lower bound 3.00pp" in result["reasons"][0]
    assert result["paired"]["quality_delta"] == pytest.approx(0.05)


def test_paired_stats_from_comparison_shapes():
    assert paired_stats_from_comparison(None) is None
    assert paired_stats_from_comparison({"comparisons": []}) is None
    two = _paired_payload()
    two["comparisons"].append(dict(two["comparisons"][0]))
    assert paired_stats_from_comparison(two) is None  # ambiguous, not guessed
    bad = _paired_payload()
    bad["comparisons"][0]["quality_delta"] = "high"
    assert paired_stats_from_comparison(bad) is None
    incomplete = {"comparisons": [{"paired_cases": 60}]}
    assert paired_stats_from_comparison(incomplete) is None
    no_ci = _paired_payload()
    del no_ci["comparisons"][0]["quality_delta_ci95"]
    stats = paired_stats_from_comparison(no_ci)
    assert stats is not None and stats.quality_delta_ci95 is None
    assert stats.paired_cases == 60
    assert stats.quality_delta == pytest.approx(1 / 60)


def test_merge_paired_stats_takes_widest_span():
    def one(cases: int, ci: tuple[float, float] | None) -> PairedStats:
        return PairedStats(
            paired_cases=cases,
            quality_delta=0.05,
            quality_delta_ci95=ci,
            quality_delta_bootstrap_ci95=ci,
            wins=1,
            losses=0,
            ties=cases - 1,
        )

    merged = _merge_paired_stats([one(60, (0.01, 0.03)), one(60, (0.02, 0.06))])
    assert merged.paired_cases == 60
    assert merged.repetitions == 2
    assert merged.paired_observations == 120
    assert merged.quality_delta == pytest.approx(0.05)
    assert merged.quality_delta_ci95 == (0.01, 0.06)
    assert merged.quality_delta_bootstrap_ci95 == (0.01, 0.06)
    assert _merge_paired_stats([]) is None
    assert (
        _merge_paired_stats([one(60, (0.01, 0.03)), one(60, None)]).quality_delta_ci95
        is None
    )


def test_merge_paired_stats_repeated_runs_do_not_dilute_min_cases():
    # Regression (M2): 70 cases x 3 repeated runs of the same case set.
    # Summing paired_cases would read 210 and clear a min_cases 200 bar;
    # the merge keeps the distinct-case count so the verdict stays
    # INCONCLUSIVE on an insufficient sample.
    def run() -> PairedStats:
        return PairedStats(
            paired_cases=70,
            quality_delta=0.05,
            quality_delta_ci95=(0.03, 0.07),
            quality_delta_bootstrap_ci95=(0.03, 0.07),
            wins=40,
            losses=20,
            ties=10,
        )

    merged = _merge_paired_stats([run(), run(), run()])
    assert merged.paired_cases == 70
    assert merged.repetitions == 3
    assert merged.paired_observations == 210
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=merged,
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "insufficient paired sample: 70 paired cases < min_cases 200" in (
        result.reasons[0]
    )


# ---- M3: evidence quality (truncation / missing-result / no-answer) ----

# Captured verbatim from the real J3 sr-bench run
# (run-1a415ad8b047446da884, sr-bench report.py `output_diagnostics`):
# 32 of 60 planned cases were force-stopped by the output token limit.
_J3_OUTPUT_DIAGNOSTICS = {
    "planned_cases": 60,
    "result_cases": 60,
    "output_limit_cases": 32,
    "strict_format": {
        "checked_cases": 28,
        "failed_cases": 28,
        "unassessed_cases": 32,
    },
    "subject_calls": {
        "total": 60,
        "finish_reasons": {"length": 32, "stop": 28},
        "unknown_finish_reason": 0,
    },
}


def _target(
    target_id: str,
    *,
    kind: str | None = None,
    planned: int = 60,
    results: int | None = None,
    output_limit: int | None = None,
    length: int | None = None,
    no_answer: int | None = None,
) -> dict:
    """One sr-bench target summary with the real `output_diagnostics` shape.

    Mirrors sr-bench `report.py`: `output_limit_cases` counts cases
    force-stopped by the token limit, and the finish reasons live under
    `subject_calls` (per call, not per case).
    """
    result_cases = planned if results is None else results
    if length is None:
        length = output_limit if output_limit else 0
    finish: dict = {"length": length} if length else {}
    stop = planned - length
    if stop or not finish:
        finish["stop"] = stop
    diagnostics: dict = {
        "planned_cases": planned,
        "result_cases": result_cases,
        "output_limit_cases": output_limit if output_limit is not None else length,
        "strict_format": {
            "checked_cases": stop,
            "failed_cases": 0,
            "unassessed_cases": length,
        },
        "subject_calls": {
            "total": planned,
            "finish_reasons": finish,
            "unknown_finish_reason": 0,
        },
    }
    if no_answer is not None:
        diagnostics["no_answer_cases"] = no_answer
    entry: dict = {"id": target_id, "output_diagnostics": diagnostics}
    if kind is not None:
        entry["kind"] = kind
    return entry


def _sectioned_payload(
    *,
    baseline_targets: list[dict] | None = None,
    candidate_targets: list[dict] | None = None,
) -> dict:
    """M3-shaped compare payload: target sections with output diagnostics."""
    return {
        "baseline_targets": (
            baseline_targets if baseline_targets is not None else [_target("single-a")]
        ),
        "candidate_targets": (
            candidate_targets
            if candidate_targets is not None
            else [_target("mom-x", kind="mom")]
        ),
        "baseline_selected_target_id": "single-a",
    }


def test_live_verdict_truncated_baseline_is_inconclusive():
    # J3's single reference arm lost 32/60 cases to output-limit force
    # stops; under the M3 ceiling that silent quality loss becomes
    # unreliable evidence. The fixture is the real run's diagnostics.
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return _sectioned_payload(
                baseline_targets=[
                    {"id": "single-a", "output_diagnostics": _J3_OUTPUT_DIAGNOSTICS}
                ],
            )

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert "baseline truncation rate 53.33% exceeds max 2.00%" in result["reasons"][0]
    assert result["evidence_quality"]["baseline"] == {
        "truncation_rate": pytest.approx(32 / 60),
        "missing_result_rate": 0.0,
        "no_answer_rate": None,
        "finish_reason_length_rate": pytest.approx(32 / 60),
    }
    assert result["evidence_quality"]["candidate"] == {
        "truncation_rate": 0.0,
        "missing_result_rate": 0.0,
        "no_answer_rate": None,
        "finish_reason_length_rate": 0.0,
    }


def test_live_verdict_missing_results_over_ceiling_is_inconclusive():
    # Cases with no recorded result contribute no judged answer - the
    # missing-result rate gates INCONCLUSIVE, and is distinct from the
    # per-case no-answer rate.
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return _sectioned_payload(
                candidate_targets=[_target("mom-x", kind="mom", results=52)],
            )

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert (
        "candidate missing-result rate 13.33% exceeds max 2.00%" in result["reasons"][0]
    )
    assert result["evidence_quality"]["candidate"] == {
        "truncation_rate": 0.0,
        "missing_result_rate": pytest.approx(8 / 60),
        "no_answer_rate": None,
        "finish_reason_length_rate": 0.0,
    }


def test_live_verdict_no_answer_over_ceiling_is_inconclusive():
    # When sr-bench publishes a per-case no-answer count it gates too;
    # without one the rate stays null (never guessed as zero).
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return _sectioned_payload(
                candidate_targets=[
                    _target("mom-x", kind="mom", no_answer=8),
                ],
            )

    result = live_verdict_from_runs(
        spec=_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert "candidate no-answer rate 13.33% exceeds max 2.00%" in result["reasons"][0]
    assert result["evidence_quality"]["candidate"]["no_answer_rate"] == pytest.approx(
        8 / 60
    )
    assert result["evidence_quality"]["candidate"]["missing_result_rate"] == 0.0


def test_live_verdict_missing_diagnostics_does_not_veto():
    # A payload without output diagnostics reports unknown rates and lets
    # the paired decision stand - missing evidence is never guessed as zero.
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return _paired_payload(paired_cases=400, delta=0.05, ci=(0.03, 0.07))

    result = live_verdict_from_runs(
        spec=_pp_spec(2.0),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    assert not any("rate" in reason for reason in result["reasons"])
    assert result["evidence_quality"] == {
        "baseline": {
            "truncation_rate": None,
            "missing_result_rate": None,
            "no_answer_rate": None,
            "finish_reason_length_rate": None,
        },
        "candidate": {
            "truncation_rate": None,
            "missing_result_rate": None,
            "no_answer_rate": None,
            "finish_reason_length_rate": None,
        },
    }


def test_arm_evidence_quality_selection_rules():
    reference = _target("single-a", output_limit=6)
    mom = _target("mom-x", kind="mom", planned=50, results=48, output_limit=1)

    # baseline: the target compare pinned via baseline_selected_target_id wins
    evidence = arm_evidence_quality(
        [reference, _target("single-b", output_limit=40)],
        routing_arm=False,
        selected_id="single-a",
    )
    assert evidence["truncation_rate"] == pytest.approx(6 / 60)

    # candidate: the routing (mom) target wins over the reference lane
    evidence = arm_evidence_quality([reference, mom], routing_arm=True)
    assert evidence["truncation_rate"] == pytest.approx(1 / 50)
    assert evidence["missing_result_rate"] == pytest.approx(2 / 50)
    assert evidence["no_answer_rate"] is None
    assert evidence["finish_reason_length_rate"] == pytest.approx(1 / 50)

    # no diagnostics anywhere, or nonsense planned count -> unknown
    assert arm_evidence_quality([{"id": "x"}], routing_arm=False) is None
    assert arm_evidence_quality(None, routing_arm=True) is None
    bad = _target("single-c")
    bad["output_diagnostics"]["planned_cases"] = 0
    assert arm_evidence_quality([bad], routing_arm=False) is None


def test_arm_evidence_quality_truncation_counts_cases_not_calls():
    # output_limit_cases counts cases; subject_calls.finish_reasons.length
    # counts calls - auxiliary retries can add length-finished calls without
    # new cases, so the per-case count is the truncation rate and the call
    # share is reference only.
    entry = _target("single-a", output_limit=6, length=10)
    evidence = arm_evidence_quality([entry], routing_arm=False)
    assert evidence["truncation_rate"] == pytest.approx(6 / 60)
    assert evidence["finish_reason_length_rate"] == pytest.approx(10 / 60)


def test_worst_rates_takes_max_and_keeps_unknown():
    assert _worst_rates(
        [
            {
                "truncation_rate": 0.01,
                "missing_result_rate": 0.02,
                "no_answer_rate": None,
                "finish_reason_length_rate": 0.0,
            },
            {
                "truncation_rate": 0.05,
                "missing_result_rate": None,
                "no_answer_rate": 0.03,
                "finish_reason_length_rate": None,
            },
            None,
        ]
    ) == {
        "truncation_rate": 0.05,
        "missing_result_rate": 0.02,
        "no_answer_rate": 0.03,
        "finish_reason_length_rate": 0.0,
    }
    assert _worst_rates([None]) == {
        "truncation_rate": None,
        "missing_result_rate": None,
        "no_answer_rate": None,
        "finish_reason_length_rate": None,
    }
    assert _worst_rates([]) is None


# ---- M4: reference arms and cost basis ----


def _scored_target(target_id: str, score: float, **extra: object) -> dict:
    """A sectioned-payload target carrying the objective metric value."""
    entry = _target(target_id)
    entry["capability.score"] = score
    entry.update(extra)
    return entry


def _ref_bound_spec(**objective_overrides: object) -> ExperimentSpec:
    spec_dict = copy.deepcopy(VALID_SPEC)
    spec_dict["references"] = {"strong": "ref-strong", "weak": "ref-weak"}
    spec_dict["objective"]["constraints"] = [
        {
            "name": "accuracy-vs-strong",
            "metric": "capability.score",
            "op": "gte",
            "bound": "ref:strong",
        }
    ]
    spec_dict["objective"].update(objective_overrides)
    return ExperimentSpec.from_dict(spec_dict)


def test_live_verdict_ref_bound_resolves_against_sections():
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return _sectioned_payload(
                baseline_targets=[
                    _scored_target("ref-weak", 0.70),
                    _scored_target("ref-strong", 0.85),
                    _scored_target("single-a", 0.80),
                ],
                candidate_targets=[_scored_target("mom-x", 0.90, kind="mom")],
            )

    result = live_verdict_from_runs(
        spec=_ref_bound_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "PASS"
    assert result["references"] == {"strong": 0.85, "weak": 0.70}
    [constraint] = result["constraints"]
    assert constraint["status"] == "pass"
    assert constraint["bound_value"] == pytest.approx(0.85)
    assert constraint["ref_value"] == pytest.approx(0.85)
    rows = result["pareto"]
    assert [(r["arm"], r["id"], r["role"]) for r in rows] == [
        ("baseline", "ref-weak", "weak"),
        ("baseline", "ref-strong", "strong"),
        ("baseline", "single-a", None),
        ("candidate", "mom-x", None),
    ]


def test_live_verdict_missing_reference_arm_is_inconclusive():
    # the ref:strong bound resolves against an arm the payload does not
    # carry: unevaluated (INCONCLUSIVE), never zero.
    class FakeCli:
        def show(self, run_id):
            return dict(_RUN_DOC)

        def compare(self, base, cand):
            return _sectioned_payload(
                baseline_targets=[_scored_target("single-a", 0.80)],
                candidate_targets=[_scored_target("mom-x", 0.90, kind="mom")],
            )

    result = live_verdict_from_runs(
        spec=_ref_bound_spec(),
        cli=FakeCli(),
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["references"] == {"strong": None, "weak": None}
    assert "reference arm 'strong'" in result["reasons"][0]
    assert "never counted as zero" in result["reasons"][0]


def test_reference_metric_values_match_by_id():
    refs = References.from_dict({"strong": "ref-strong", "weak": "ref-weak"})
    targets = [
        {"id": "single-a", "capability.score": 0.80},
        {"id": "ref-weak", "capability.score": 0.70},
        {"id": "ref-strong", "capability.score": 0.85},
    ]
    assert reference_metric_values(targets, refs, "capability.score") == {
        "strong": 0.85,
        "weak": 0.70,
    }
    assert reference_metric_values(None, refs, "capability.score") == {
        "strong": None,
        "weak": None,
    }
    assert reference_metric_values(targets, None, "capability.score") == {
        "strong": None,
        "weak": None,
    }


def test_arm_cost_measured_wins_and_unknown_stays_none():
    assert arm_cost({"total_spend_usd": 1.25}, None) == (1.25, "measured")
    assert arm_cost({"cost_usd": 0.5}, None) == (0.5, "measured")
    # measured wins even when pricing is declared
    basis = CostBasis.from_dict(
        {
            "source": "provider_pricing",
            "assumed_pricing": {
                "model-a": {"prompt_per_1m": 1.0, "completion_per_1m": 2.0}
            },
        }
    )
    assert arm_cost({"total_spend_usd": 1.25}, basis) == (1.25, "measured")
    assert arm_cost({"accuracy": 0.9}, basis) == (None, None)
    assert arm_cost(None, basis) == (None, None)


def test_arm_cost_assumed_from_declared_pricing():
    basis = CostBasis.from_dict(
        {
            "source": "provider_pricing",
            "assumed_pricing": {
                "model-a": {"prompt_per_1m": 1.0, "completion_per_1m": 2.0}
            },
        }
    )
    target = {
        "tokens": {"input_tokens": 1_000_000, "output_tokens": 500_000},
        "selected_models": {"model-a": 12},
    }
    assert arm_cost(target, basis) == (pytest.approx(2.0), "assumed")
    # cached input counts toward the prompt side
    target["tokens"] = {
        "input_tokens": 900_000,
        "cached_input_tokens": 100_000,
        "output_tokens": 500_000,
    }
    assert arm_cost(target, basis)[0] == pytest.approx(2.0)
    # unknown usage parts contribute zero, never block the estimate
    target["tokens"] = {"input_tokens": 1_000_000}
    assert arm_cost(target, basis) == (pytest.approx(1.0), "assumed")


def test_arm_cost_assumed_requires_single_model_with_rates():
    basis = CostBasis.from_dict(
        {
            "source": "provider_pricing",
            "assumed_pricing": {
                "model-a": {"prompt_per_1m": 1.0, "completion_per_1m": 2.0}
            },
        }
    )
    tokens = {"input_tokens": 1_000_000, "output_tokens": 0}
    # a routed arm touching two models is not estimable
    two_models = {"tokens": tokens, "selected_models": {"model-a": 3, "model-b": 4}}
    assert arm_cost(two_models, basis) == (None, None)
    # no declared rate for the model
    assert arm_cost({"tokens": tokens, "selected_models": {"model-z": 3}}, basis) == (
        None,
        None,
    )
    # no pricing section at all
    assert arm_cost({"tokens": tokens, "selected_models": {"model-a": 3}}, None) == (
        None,
        None,
    )
    # no token usage
    assert arm_cost({"selected_models": {"model-a": 3}}, basis) == (None, None)


def test_pareto_rows_shape_and_reference_roles():
    basis = CostBasis.from_dict(
        {
            "source": "provider_pricing",
            "assumed_pricing": {
                "model-a": {"prompt_per_1m": 1.0, "completion_per_1m": 2.0}
            },
        }
    )
    refs = References.from_dict({"strong": "ref-strong", "weak": "ref-weak"})
    baseline_targets = [
        {
            "id": "ref-strong",
            "accuracy": 0.85,
            "accuracy_ci95": [0.80, 0.90],
            "total_spend_usd": 2.0,
            "latency_p50_s": 0.4,
            "latency_p95_s": 0.9,
        },
        {"id": "ref-weak", "accuracy": 0.70},
        {"id": "single-a", "accuracy": 0.80},
    ]
    candidate_targets = [
        {
            "id": "mom-x",
            "accuracy": 0.90,
            "tokens": {"input_tokens": 1_000_000, "output_tokens": 0},
            "selected_models": {"model-a": 5},
        }
    ]
    rows = pareto_rows(baseline_targets, candidate_targets, refs, basis)
    assert [(r["arm"], r["id"], r["role"]) for r in rows] == [
        ("baseline", "ref-strong", "strong"),
        ("baseline", "ref-weak", "weak"),
        ("baseline", "single-a", None),
        ("candidate", "mom-x", None),
    ]
    strong = rows[0]
    assert strong["accuracy"] == 0.85
    assert strong["accuracy_ci95"] == [0.80, 0.90]
    assert strong["cost_usd"] == 2.0 and strong["cost_note"] == "measured"
    assert strong["latency_p50_s"] == 0.4 and strong["latency_p95_s"] == 0.9
    assert rows[1]["cost_usd"] is None and rows[1]["cost_note"] is None
    assert rows[3]["cost_note"] == "assumed"
    # without declared references no role is tagged
    plain = pareto_rows(baseline_targets, candidate_targets, None, basis)
    assert all(r["role"] is None for r in plain)


def _ref_verify_spec() -> ExperimentSpec:
    spec_dict = copy.deepcopy(VALID_SPEC)
    spec_dict["references"] = {"strong": "ref-strong", "weak": "ref-weak"}
    spec_dict["objective"]["constraints"] = [
        {
            "name": "accuracy-vs-strong",
            "metric": "capability.score",
            "op": "gte",
            "bound": "ref:strong",
        }
    ]
    return ExperimentSpec.from_dict(spec_dict)


def _ref_compare_payload() -> dict:
    return _sectioned_payload(
        baseline_targets=[
            _scored_target("ref-weak", 0.70),
            _scored_target("ref-strong", 0.85),
            _scored_target("single-a", 0.80),
        ],
        candidate_targets=[_scored_target("mom-x", 0.90, kind="mom")],
    )


def _ref_report(alias: str, score: float) -> dict:
    return {
        "summary": {
            "targets": [{"id": alias, "capability.score": score, "accuracy": score}]
        }
    }


def test_verify_reference_arms_bind_and_cache(tmp_path):
    cache_path = tmp_path / "run-id-cache.json"
    spec = _ref_verify_spec()
    kwargs = {
        "spec": spec,
        "baseline_run_id": "run-base",
        "targets": ["mom-router"],
        "repetitions": 1,
        "measurements": [
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        "work_dir": tmp_path,
        "run_cache": cache_path,
    }
    reports = {
        "ref-strong": _ref_report("ref-strong", 0.85),
        "ref-weak": _ref_report("ref-weak", 0.70),
    }

    stub = VerifyStub(
        compare_values=_ref_compare_payload(),
        ref_statuses={"ref-strong": "completed", "ref-weak": "completed"},
        registry=[{"id": "ref-strong"}, {"id": "ref-weak"}],
    )
    stub.ref_reports = reports
    result = verify_live(cli=stub, **kwargs)
    assert result["verdict"] == "PASS"
    assert stub.ref_runs == 2
    # the derived reference manifest drops the baseline's stale review
    # receipt before submission (the targets swap invalidates it)
    strong_manifest = json.loads(
        (tmp_path / "reference-strong-manifest.json").read_text("utf-8")
    )
    assert "plan_sha256" not in strong_manifest
    assert strong_manifest["targets"] == [{"id": "ref-strong"}]
    assert result["references"]["strong"] == {
        "alias": "ref-strong",
        "run_id": "run-ref-ref-strong",
        "cached": False,
        "metric": 0.85,
        "cost_usd": None,
        "cost_note": None,
    }
    assert result["references"]["weak"]["metric"] == 0.70
    assert "run-ref-ref-strong" in result["bench_refs"]
    assert any(
        row["arm"] == "reference"
        and row["role"] == "strong"
        and row["id"] == "ref-strong"
        for row in result["pareto"]
    )
    # the completed run ids land in the cache keyed by dataset+manifest+alias
    assert cache_path.is_file()
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert sorted(value for value in cached.values()) == [
        "run-ref-ref-strong",
        "run-ref-ref-weak",
    ]

    # second invocation: cache hit - no new reference runs, same evidence
    stub2 = VerifyStub(
        compare_values=_ref_compare_payload(),
        ref_statuses={"ref-strong": "completed", "ref-weak": "completed"},
        registry=[{"id": "ref-strong"}, {"id": "ref-weak"}],
    )
    stub2.ref_reports = reports
    result2 = verify_live(cli=stub2, **kwargs)
    assert stub2.ref_runs == 0
    assert result2["verdict"] == "PASS"
    assert result2["references"]["strong"]["cached"] is True
    assert result2["references"]["strong"]["metric"] == 0.85


def test_verify_failed_reference_arm_is_inconclusive(tmp_path):
    stub = VerifyStub(
        compare_values=_ref_compare_payload(),
        ref_statuses={"ref-strong": "failed", "ref-weak": "completed"},
        registry=[{"id": "ref-strong"}, {"id": "ref-weak"}],
    )
    stub.ref_reports = {"ref-weak": _ref_report("ref-weak", 0.70)}
    result = verify_live(
        spec=_ref_verify_spec(),
        cli=stub,
        baseline_run_id="run-base",
        targets=["mom-router"],
        repetitions=1,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )
    assert result["verdict"] == "INCONCLUSIVE"
    assert "reference arm 'strong' run run-ref-ref-strong ended 'failed'" in (
        result["reasons"]
    )
    # a failed reference run carries no metric evidence at all
    assert result["references"]["strong"].get("metric") is None
    assert result["references"]["weak"]["metric"] == 0.70


# ---- R2: verify auto two-segment baseline manifest ----


def _mom_manifest(tmp_path: Path, targets: list[dict]) -> Path:
    path = tmp_path / "baseline-manifest-source.json"
    path.write_text(
        json.dumps(
            {
                "version": "sr-bench-1.0",
                "name": "recipe-lab-baseline",
                "targets": targets,
                "cases": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


_REGISTRY_SINGLE = {
    "id": "ref-strong",
    "kind": "single",
    "model": "vllm-sr/auto",
    "base_url": "http://127.0.0.1:9000",
    "config_hash": "a" * 64,
    "max_inference_calls": 1,
    "prices": {
        "input_per_mtok_usd": 0.0,
        "output_per_mtok_usd": 0.0,
        "cache_read_per_mtok_usd": 0.0,
        "cache_write_per_mtok_usd": 0.0,
    },
}

_MOM_TARGET = {
    "id": "mom-router",
    "kind": "mom",
    "model": "vllm-sr/auto",
    "base_url": "http://127.0.0.1:9001",
}


def _manifest_verify_kwargs(manifest: Path, tmp_path: Path) -> dict:
    return {
        "spec": _ref_verify_spec(),
        "cli": None,  # set by the caller
        "baseline_run_id": None,
        "baseline_manifest": str(manifest),
        "targets": ["mom-router"],
        "repetitions": 1,
        "measurements": [
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        "work_dir": tmp_path,
    }


def test_verify_manifest_injects_registry_single_arm(tmp_path):
    # a MoM-only baseline manifest gets the spec's strong reference target
    # embedded verbatim from the sr-bench registry as the S arm
    stub = VerifyStub(
        candidate_statuses=["completed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_ref_compare_payload(),
        ref_statuses={"ref-strong": "completed", "ref-weak": "completed"},
        registry=[_REGISTRY_SINGLE, dict(_REGISTRY_SINGLE, id="ref-weak")],
    )
    stub.ref_reports = {
        "ref-strong": _ref_report("ref-strong", 0.85),
        "ref-weak": _ref_report("ref-weak", 0.70),
    }
    manifest = _mom_manifest(tmp_path, [dict(_MOM_TARGET)])
    kwargs = _manifest_verify_kwargs(manifest, tmp_path)
    kwargs["cli"] = stub
    result = verify_live(**kwargs)

    assert ("targets",) in stub.events
    # submission + pair-0 baseline arm; the two M4 reference arms go through
    # their own single-target manifests
    assert stub.manifest_runs == 2
    assert stub.ref_runs == 2
    frozen = json.loads(
        (tmp_path / "baseline-manifest.submitted.json").read_text("utf-8")
    )
    assert frozen["targets"][0] == _REGISTRY_SINGLE
    assert frozen["targets"][1] == _MOM_TARGET
    assert result["baseline_run_id"] == "run-base-trial-1"
    assert result["bench_refs"][0] == "run-base-trial-1"
    assert result["verdict"] == "PASS"


def test_verify_manifest_with_single_arm_submits_verbatim(tmp_path):
    # a manifest that already carries a single target is submitted as-is:
    # no registry lookup, no injection
    single = {
        "id": "local-strong",
        "kind": "single",
        "model": "local/strong",
        "base_url": "http://127.0.0.1:9000",
    }
    stub = VerifyStub(
        candidate_statuses=["completed"],
        baseline_statuses=["completed", "completed"],
        compare_values=_compare_values(0.80, 0.86),
    )
    manifest = _mom_manifest(tmp_path, [single, dict(_MOM_TARGET)])
    result = verify_live(
        spec=_verify_spec(),
        cli=stub,
        baseline_run_id=None,
        baseline_manifest=str(manifest),
        targets=["mom-router"],
        repetitions=1,
        measurements=[
            {"name": "pii-recall-floor", "candidate": 0.97, "baseline": 0.96}
        ],
        work_dir=tmp_path,
    )

    assert ("targets",) not in stub.events
    frozen = json.loads(
        (tmp_path / "baseline-manifest.submitted.json").read_text("utf-8")
    )
    assert frozen["targets"][0] == single
    assert result["baseline_run_id"] == "run-base-trial-1"
    assert result["verdict"] == "PASS"


def test_verify_rejects_ambiguous_baseline_source(tmp_path):
    stub = VerifyStub()
    manifest = _mom_manifest(tmp_path, [dict(_MOM_TARGET)])
    base_kwargs = {
        "spec": _verify_spec(),
        "cli": stub,
        "targets": ["mom-router"],
        "repetitions": 1,
        "measurements": [],
        "work_dir": tmp_path,
    }

    # both sources: contract error before anything runs
    with pytest.raises(ContractError, match="exactly one"):
        verify_live(
            baseline_run_id="run-base", baseline_manifest=str(manifest), **base_kwargs
        )
    # neither source: same contract error
    with pytest.raises(ContractError, match="exactly one"):
        verify_live(baseline_run_id=None, baseline_manifest=None, **base_kwargs)
    assert stub.events == []


def test_verify_manifest_dataset_mismatch_rejected_before_submission(tmp_path):
    # a file-referenced dataset is checked against the contract split before
    # any run is spent on it: zero-cost rejection (V2-4)
    stub = VerifyStub()
    manifest = _mom_manifest(tmp_path, [dict(_REGISTRY_SINGLE)])
    raw = json.loads(manifest.read_text("utf-8"))
    raw["dataset"] = {"path": "cases.jsonl", "sha256": "f" * 64}
    manifest.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match="binds dataset"):
        verify_live(
            spec=_verify_spec(),
            cli=stub,
            baseline_run_id=None,
            baseline_manifest=str(manifest),
            targets=["mom-router"],
            repetitions=1,
            measurements=[],
            work_dir=tmp_path,
        )
    assert stub.manifest_runs == 0
    assert stub.events == []


def test_verify_manifest_check_binds_srbench_rendering(tmp_path):
    # X3: with a rendering declared, the pre-submission manifest check binds
    # the rendering's hash - the legacy cases hash is no longer accepted
    manifest = _mom_manifest(tmp_path, [dict(_REGISTRY_SINGLE)])
    raw = json.loads(manifest.read_text("utf-8"))
    raw["dataset"] = {"path": "cases.jsonl", "sha256": _LEGACY_HASH}
    manifest.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match="binds dataset"):
        verify_live(
            spec=_srbench_bound_spec(),
            cli=VerifyStub(),
            baseline_run_id=None,
            baseline_manifest=str(manifest),
            targets=["mom-router"],
            repetitions=1,
            measurements=[],
            work_dir=tmp_path,
        )
