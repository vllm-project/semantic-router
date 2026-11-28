"""Sub-issue 4 tests: ledger-only report replay, untrusted next header,
provenance manifest emission, and the G3 full next-round context."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from recipe_lab.contract import ExperimentSpec
from recipe_lab.next import render_next_context
from recipe_lab.provenance import build_run_manifest
from recipe_lab.report import render_run_report
from tests.fixtures import VALID_SPLITS_SPEC

REPLAY = {
    "record_id": "rec-111111111111",
    "round": 1,
    "ts": "t",
    "tier": "replay",
    "verdict": "RANKED",
    "candidate": {
        "kind": "config",
        "ref": "c1.yaml",
        "provenance": {"proposer": "threshold-search"},
    },
    "metrics": {"routed_accuracy": 0.85},
    "dataset_hash": "sha256:aa",
    "coverage_warning": False,
}
LIVE = {
    "record_id": "rec-222222222222",
    "round": 2,
    "ts": "t",
    "tier": "live",
    "verdict": "PASS",
    "candidate": {
        "kind": "config",
        "ref": "c1.yaml",
        "provenance": {"proposer": "threshold-search"},
    },
    "metrics": {"accuracy": 0.9},
    "dataset_hash": "sha256:aa",
    "gates": {"pii": "pass"},
    "bench_refs": ["run-a", "run-b"],
    "coverage_warning": False,
}


def test_report_replays_ledger_and_flags_missing_evidence():
    report = render_run_report([REPLAY, LIVE])
    assert "**PASS**" in report and "RANKED" in report
    assert "rec-111111111111" in report  # G1: every row cites its record id
    assert "replayed from the recipe-lab ledger only" in report.lower()
    broken = dict(LIVE, gates=None)
    report = render_run_report([broken])
    assert "Replay inconsistencies" in report


def test_report_flags_mixed_dataset_hashes():
    other = dict(REPLAY, round=3, dataset_hash="sha256:bb")
    report = render_run_report([REPLAY, LIVE, other])
    assert "Cross-dataset warning" in report
    assert "NOT directly comparable" in report
    assert "sha256:bb" in report


def test_report_keeps_reasons_units_and_unguarded_pass():
    # review regression: the report dropped the ledger's reasons, rewrote a
    # 0.4 s latency as 40.00%, and flagged a valid PASS whose spec declared
    # no guards (an empty gates dict is not missing evidence).
    unguarded = dict(
        LIVE,
        gates={},
        metrics={"latency_s": 0.4, "accuracy": 0.4},
        reasons=["candidate metric missing"],
    )
    report = render_run_report([unguarded])
    assert "candidate metric missing" in report
    assert "latency_s=0.4" in report
    assert "40.00%" not in report
    assert "Replay inconsistencies" not in report
    assert "Replay inconsistencies" in render_run_report([dict(LIVE, gates=None)])

    from recipe_lab.report import render_promotion_draft

    draft = render_promotion_draft(LIVE)
    assert "DRAFT" in draft and "human review" in draft.lower()
    assert "rec-222222222222" in draft  # G1: the draft cites the ledger record
    assert "c1.yaml" in draft and "run-a" in draft
    assert "Standard holdout" in draft


def test_next_context_carries_untrusted_header():
    spec = ExperimentSpec.from_dict(VALID_SPLITS_SPEC)
    context = render_next_context([REPLAY, LIVE], spec)
    assert context.startswith("# recipe-lab knowledge context")
    assert "untrusted reference material" in context
    assert "c1.yaml" in context
    assert "rec-111111111111" in context  # G1: the tail cites record ids
    assert "Discipline reminders" in context


def _next_spec(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(VALID_SPLITS_SPEC), encoding="utf-8")
    return str(spec_path)


def test_next_contract_summary_sections(tmp_path):
    spec = ExperimentSpec.from_dict(VALID_SPLITS_SPEC)
    context = render_next_context([REPLAY, LIVE], spec)
    assert "## Contract summary" in context
    assert "- objective: metric_key=capability.score direction=maximize" in context
    assert "- min_cases: 200" in context  # M2: the effective paired floor
    assert "- replay_metric: routed_accuracy" in context
    assert "- gates:" in context and "pii-recall-floor" in context
    # split names and hashes only; no frozen_test case content
    assert "calib=sha256:" + "1" * 64 in context
    assert "frozen_test=sha256:" + "3" * 64 in context
    assert "- budget: rounds=5 used=2 remaining=3" in context


def test_next_contract_summary_shows_the_declared_paired_threshold():
    # review regression: a spec that declares min_improvement_pp rendered
    # min_improvement_pct=0.0 and hid the configured bar entirely.
    raw = dict(VALID_SPLITS_SPEC)
    raw["objective"] = {
        "metric_key": "capability.score",
        "direction": "maximize",
        "min_improvement_pp": 2,
        "min_cases": 24,
        "replay_metric": "routed_accuracy",
    }
    spec = ExperimentSpec.from_dict(raw)
    context = render_next_context([REPLAY, LIVE], spec)
    assert "min_improvement_pp=2" in context
    assert "min_improvement_pct" not in context
    assert "- min_cases: 24" in context


def test_next_best_replay_uses_metric_not_recency():
    # A2: the buggy selection let the last RANKED record win regardless of
    # metric; an earlier candidate with a higher metric must win instead.
    spec = ExperimentSpec.from_dict(VALID_SPLITS_SPEC)
    early_high = dict(
        REPLAY, record_id="rec-aaaaaaaaaaaa", round=1, metrics={"routed_accuracy": 0.9}
    )
    late_low = dict(
        REPLAY, record_id="rec-bbbbbbbbbbbb", round=2, metrics={"routed_accuracy": 0.7}
    )
    context = render_next_context([early_high, late_low, LIVE], spec)
    assert "rec-aaaaaaaaaaaa" in context
    best_section = context.split("## Best replay-tier candidate")[1].split("##")[0]
    assert "rec-aaaaaaaaaaaa" in best_section
    assert "rec-bbbbbbbbbbbb" not in best_section


def test_next_best_replay_is_scoped_to_the_newest_dataset_and_split():
    # review regression: a mixed ledger let a higher-scoring candidate from
    # another dataset or split outrank the comparable one.
    spec = ExperimentSpec.from_dict(VALID_SPLITS_SPEC)
    good = dict(
        REPLAY,
        record_id="rec-cccccccccccc",
        round=1,
        split="dev",
        metrics={"routed_accuracy": 0.8},
    )
    other_dataset = dict(
        REPLAY,
        record_id="rec-dddddddddddd",
        round=2,
        split="dev",
        metrics={"routed_accuracy": 0.95},
        dataset_hash="sha256:bb",
    )
    other_split = dict(
        REPLAY,
        record_id="rec-eeeeeeeeeeee",
        round=3,
        split="calib",
        metrics={"routed_accuracy": 0.9},
    )
    current = dict(
        REPLAY,
        record_id="rec-ffffffffffff",
        round=4,
        split="dev",
        metrics={"routed_accuracy": 0.7},
    )
    context = render_next_context([good, other_dataset, other_split, current], spec)
    best_section = context.split("## Best replay-tier candidate")[1].split("##")[0]
    assert "rec-cccccccccccc" in best_section
    assert "rec-dddddddddddd" not in best_section
    assert "rec-eeeeeeeeeeee" not in best_section

    # A2: coverage_warning records and timeout trials never rank.
    spec = ExperimentSpec.from_dict(VALID_SPLITS_SPEC)
    good = dict(
        REPLAY, record_id="rec-cccccccccccc", round=1, metrics={"routed_accuracy": 0.8}
    )
    flagged = dict(
        REPLAY,
        record_id="rec-dddddddddddd",
        round=2,
        metrics={"routed_accuracy": 0.99},
        coverage_warning=True,
    )
    timeout = dict(
        REPLAY,
        record_id="rec-eeeeeeeeeeee",
        round=3,
        metrics={"routed_accuracy": 0.98},
        status="timeout",
    )
    context = render_next_context([flagged, timeout, good, LIVE], spec)
    best_section = context.split("## Best replay-tier candidate")[1].split("##")[0]
    assert "rec-cccccccccccc" in best_section
    assert "rec-dddddddddddd" not in best_section
    assert "rec-eeeeeeeeeeee" not in best_section


ATTR_ROWS = [
    {
        "case_id": "c1",
        "error_class": "correct",
        "answer_ok": True,
        "expected_decision": "math",
        "matched_decision": "math",
        "routed": "local/model-math",
        "expected": "local/model-math",
    },
    {
        "case_id": "c2",
        "error_class": "decision_mismatch",
        "answer_ok": False,
        "expected_decision": "math",
        "matched_decision": "code",
        "routed": "local/model-code",
        "expected": "local/model-math",
    },
]

ATTR_BYTES = "".join(json.dumps(row) + "\n" for row in ATTR_ROWS).encode()

# a summary that contradicts its artifact: `next` must never read it
TAMPERED_SUMMARY = {"total": 0, "answered": 0, "accuracy": 0.0}


def _write_attribution_artifacts(tmp_path) -> str:
    attr_dir = tmp_path / "attr"
    attr_dir.mkdir()
    (attr_dir / "attribution.jsonl").write_bytes(ATTR_BYTES)
    (attr_dir / "summary.json").write_text(
        json.dumps(TAMPERED_SUMMARY, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return f"{attr_dir / 'attribution.jsonl'}@sha256:{hashlib.sha256(ATTR_BYTES).hexdigest()}"


def _next_ledger(tmp_path, with_attribution: bool = True):
    replay_ref = dict(
        REPLAY,
        record_id="rec-333333333333",
        round=2,
        attribution_ref=(
            _write_attribution_artifacts(tmp_path) if with_attribution else None
        ),
    )
    live_last = dict(
        LIVE,
        record_id="rec-222222222222",
        round=3,
        reasons=["accuracy improved"],
        constraints={"cost": {"status": "pass"}},
    )
    records = [REPLAY, live_last]
    if with_attribution:
        records.insert(1, replay_ref)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return str(ledger)


def test_next_full_context_is_byte_stable(tmp_path):
    from recipe_lab.cli import main

    spec_path = _next_spec(tmp_path)
    ledger_path = _next_ledger(tmp_path)
    first, second = tmp_path / "ctx1.md", tmp_path / "ctx2.md"
    assert (
        main(
            [
                "next",
                "--spec",
                spec_path,
                "--ledger",
                ledger_path,
                "--output",
                str(first),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "next",
                "--spec",
                spec_path,
                "--ledger",
                ledger_path,
                "--output",
                str(second),
            ]
        )
        == 0
    )
    context = first.read_text(encoding="utf-8")
    assert context == second.read_text(encoding="utf-8")  # G3: byte-stable export
    for section in (
        "# recipe-lab knowledge context",
        "## Contract summary",
        "## Ledger tail (most recent last)",
        "## Best replay-tier candidate so far",
        "## Last live verdict",
        "## Latest attribution summary",
        "## Citable insight refs",
        "## Discipline reminders",
    ):
        assert section in context
    # last live verdict carries reasons and its citable refs
    assert "verdict=PASS" in context and "accuracy improved" in context
    assert "bench:run-a" in context
    # attribution summary read via the hash-verified attribution_ref
    assert "attr:" in context and "decision_mismatch=1" in context
    assert "50.00%" in context
    # the best candidate cites its record id
    assert "- insight-ref: rec-111111111111" in context


def test_next_empty_ledger(tmp_path):
    from recipe_lab.cli import main

    spec_path = _next_spec(tmp_path)
    ledger_path = _next_ledger(tmp_path, with_attribution=False)
    Path(ledger_path).write_text("", encoding="utf-8")
    output = tmp_path / "ctx.md"
    assert (
        main(
            [
                "next",
                "--spec",
                spec_path,
                "--ledger",
                ledger_path,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    context = output.read_text(encoding="utf-8")
    assert "none yet: run the replay tier first" in context
    assert "none yet: no live-tier record in the ledger" in context
    assert "none yet: run `recipe-lab attribute`" in context
    assert "- budget: rounds=5 used=0 remaining=5" in context


def test_next_attribution_hash_mismatch_exits_2(tmp_path, capsys):
    from recipe_lab.cli import main

    spec_path = _next_spec(tmp_path)
    attr_dir = tmp_path / "attr"
    attr_dir.mkdir()
    (attr_dir / "attribution.jsonl").write_bytes(ATTR_BYTES)
    (attr_dir / "summary.json").write_text("{}", encoding="utf-8")
    stale_ref = f"{attr_dir / 'attribution.jsonl'}@sha256:{'0' * 64}"
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(dict(REPLAY, attribution_ref=stale_ref)) + "\n", encoding="utf-8"
    )
    assert (
        main(
            [
                "next",
                "--spec",
                spec_path,
                "--ledger",
                str(ledger),
                "--output",
                str(tmp_path / "ctx.md"),
            ]
        )
        == 2
    )
    assert "attribution_ref hash mismatch" in capsys.readouterr().err


def test_next_derives_the_attribution_summary_from_verified_rows(tmp_path):
    # review regression: the hash covered attribution.jsonl but `next` read
    # the adjacent summary.json unverified, so editing only the summary
    # changed the exported accuracy. The summary is now recomputed from the
    # hash-verified rows, and a contradictory summary file is ignored.
    from recipe_lab.cli import main

    spec_path = _next_spec(tmp_path)
    attr_dir = tmp_path / "attr"
    attr_dir.mkdir()
    (attr_dir / "attribution.jsonl").write_bytes(ATTR_BYTES)
    ref = f"{attr_dir / 'attribution.jsonl'}@sha256:{hashlib.sha256(ATTR_BYTES).hexdigest()}"
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(dict(REPLAY, attribution_ref=ref)) + "\n", encoding="utf-8"
    )
    first = tmp_path / "ctx1.md"
    assert (
        main(
            [
                "next",
                "--spec",
                spec_path,
                "--ledger",
                str(ledger),
                "--output",
                str(first),
            ]
        )
        == 0
    )
    without_summary = first.read_text(encoding="utf-8")
    assert "accuracy: 50.00%" in without_summary  # derived, not read
    (attr_dir / "summary.json").write_text(
        json.dumps({"accuracy": 0.0, "total": 0}), encoding="utf-8"
    )
    second = tmp_path / "ctx2.md"
    assert (
        main(
            [
                "next",
                "--spec",
                spec_path,
                "--ledger",
                str(ledger),
                "--output",
                str(second),
            ]
        )
        == 0
    )
    assert second.read_text(encoding="utf-8") == without_summary


def test_next_diagnosis_sections_and_citable_refs(tmp_path):
    from recipe_lab.cli import main
    from recipe_lab.insights import _DIAG_RE, resolve_insight_refs

    spec_path = _next_spec(tmp_path)
    ledger_path = _next_ledger(tmp_path, with_attribution=False)
    diagnosis_path = tmp_path / "diagnosis.json"
    diagnosis_path.write_text(
        json.dumps(
            {
                "kind": "recipe-lab-diagnosis-v1",
                "dead_decisions": ["domain_code_complex"],
                "rare_decisions": [{"decision": "complex_general", "hit_pct": 0.4}],
                "unreachable_sites": [
                    {
                        "decision": "domain_code_complex",
                        "signal": "difficulty",
                        "op": "gte",
                        "threshold": 2.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "ctx.md"
    assert (
        main(
            [
                "next",
                "--spec",
                spec_path,
                "--ledger",
                ledger_path,
                "--diagnosis",
                str(diagnosis_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    context = output.read_text(encoding="utf-8")
    assert "## Diagnosis highlights" in context
    assert "- dead decisions: domain_code_complex" in context
    assert "- rare decisions: complex_general (0.4% hits)" in context
    assert "- unreachable sites: difficulty gte 2.5 @ domain_code_complex" in context
    # every citable ref in the consolidated list must be a valid insight-ref
    citable = context.split("## Citable insight refs")[1].split("##")[0]
    refs = [line[2:] for line in citable.strip().splitlines() if line.startswith("- ")]
    refs = [r for r in refs if not r.startswith("paste one")]
    from recipe_lab.ledger import Ledger

    resolved = resolve_insight_refs(refs, Ledger(ledger_path))
    diag_refs = [r for r in resolved if r.startswith("diag:")]
    assert diag_refs, "diagnosis highlights must contribute citable refs"
    for ref in diag_refs:
        assert _DIAG_RE.match(ref[len("diag:") :]), ref


def test_next_without_replay_records(tmp_path):
    spec = ExperimentSpec.from_dict(VALID_SPLITS_SPEC)
    context = render_next_context([LIVE], spec)
    assert "none yet: run the replay tier first" in context


def test_provenance_uses_the_shared_eligible_selection(tmp_path):
    # review regression: provenance had its own ranking that ignored
    # coverage_warning and compared candidates across datasets.
    good = dict(
        REPLAY,
        record_id="rec-cccccccccccc",
        round=1,
        split="dev",
        metrics={"routed_accuracy": 0.8},
    )
    flagged = dict(
        REPLAY,
        record_id="rec-dddddddddddd",
        round=2,
        split="dev",
        metrics={"routed_accuracy": 0.99},
        coverage_warning=True,
    )
    other_dataset = dict(
        REPLAY,
        record_id="rec-eeeeeeeeeeee",
        round=3,
        split="dev",
        metrics={"routed_accuracy": 0.95},
        dataset_hash="sha256:bb",
    )
    current = dict(
        REPLAY,
        record_id="rec-ffffffffffff",
        round=4,
        split="dev",
        metrics={"routed_accuracy": 0.7},
    )
    records = [good, flagged, other_dataset, current]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    manifest = build_run_manifest(str(ledger), records)
    assert manifest["best_replay_candidate"]["record_id"] == "rec-cccccccccccc"
    assert manifest["best_replay_candidate"]["ref"] == "c1.yaml"
    assert manifest["best_replay_candidate"]["score"] == 0.8


def test_provenance_manifest_from_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(REPLAY) + "\n" + json.dumps(LIVE) + "\n", encoding="utf-8"
    )
    manifest = build_run_manifest(str(ledger), [REPLAY, LIVE])
    assert manifest["kind"] == "recipe-lab-run"
    assert manifest["dataset_ref"]["content_hash"] == "sha256:aa"
    assert manifest["evidence"]["ledger"]["record_ids"] == [
        "rec-111111111111",
        "rec-222222222222",
    ]
    assert manifest["evidence"]["bench_refs"] == ["run-a", "run-b"]
    assert manifest["best_replay_candidate"]["ref"] == "c1.yaml"
    assert manifest["rounds"]["verdict_counts"] == {"PASS": 1, "RANKED": 1}
    # ledger content hash is verifiable
    import hashlib

    expected = f"sha256:{hashlib.sha256(ledger.read_bytes()).hexdigest()}"
    assert manifest["evidence"]["ledger"]["content_hash"] == expected


def test_attribute_writes_summary_json(tmp_path, capsys):
    # G3: attribute writes the summarize() result next to attribution.jsonl
    # so `next` can resolve a record's attribution_ref and read it.
    from recipe_lab.cli import main
    from tests.fixtures import VALID_SPEC

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
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "case_id": "c1",
                "truth": "math",
                "matched_decision": "code",
                "routed": "local/model-code",
                "answer_ok": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "attr-out"
    assert (
        main(
            [
                "attribute",
                "--spec",
                str(spec_path),
                "--records",
                str(records),
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    parsed = json.loads(capsys.readouterr().out)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["class_counts"]["decision_mismatch"] == 1
    # the decision_mismatch case has a known answer, so it counts against
    # the metrics: 0.0, not the None the answered-classes denominator gave
    assert summary["accuracy"] == 0.0
    assert summary["answer_accuracy"] == 0.0
    assert summary["routing_correctness"] == 0.0
    assert parsed["summary_json"] == str(out_dir / "summary.json")
