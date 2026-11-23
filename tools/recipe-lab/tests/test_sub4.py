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


def test_promotion_draft_from_pass_record():
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
    assert "- replay_metric: routed_accuracy" in context
    assert "- gates:" in context and "pii-recall-floor" in context
    # split names and hashes only; no frozen_test case content
    assert "calib=sha256:" + "1" * 64 in context
    assert "frozen_test=sha256:" + "3" * 64 in context
    assert "- budget: rounds=5 used=2 remaining=3" in context


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


def test_next_best_replay_excludes_flagged_records():
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


ATTR_DIR_SUMMARY = {
    "total": 5,
    "answered": 4,
    "correct": 2,
    "accuracy": 0.5,
    "class_counts": {
        "correct": 2,
        "E1_signal": 0,
        "decision_mismatch": 1,
        "E2_policy": 1,
        "E3_capability": 0,
        "suboptimal_routing": 0,
        "no_evidence": 0,
        "policy_gap": 1,
    },
    "e1_by_family": {},
    "e3_by_family": {},
    "e1_confusion_top": [],
    "decision_mismatch_top": [
        {"truth": "math", "matched_decision": "code", "count": 1}
    ],
    "e2_deviations_top": [
        {"routed": "local/model-code", "expected": "local/model-math", "count": 1}
    ],
}


ATTR_BYTES = b'{"case_id":"c1"}\n'


def _write_attribution_artifacts(tmp_path) -> str:
    attr_dir = tmp_path / "attr"
    attr_dir.mkdir()
    (attr_dir / "attribution.jsonl").write_bytes(ATTR_BYTES)
    (attr_dir / "summary.json").write_text(
        json.dumps(ATTR_DIR_SUMMARY, ensure_ascii=False, sort_keys=True) + "\n",
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


def test_next_missing_summary_json_exits_2(tmp_path, capsys):
    from recipe_lab.cli import main

    spec_path = _next_spec(tmp_path)
    attr_dir = tmp_path / "attr"
    attr_dir.mkdir()
    (attr_dir / "attribution.jsonl").write_bytes(ATTR_BYTES)  # no summary.json
    ref = f"{attr_dir / 'attribution.jsonl'}@sha256:{hashlib.sha256(ATTR_BYTES).hexdigest()}"
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(dict(REPLAY, attribution_ref=ref)) + "\n", encoding="utf-8"
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
    assert "attribution summary not found" in capsys.readouterr().err


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
    assert summary["accuracy"] is None  # the single case produced no answered class
    assert parsed["summary_json"] == str(out_dir / "summary.json")
