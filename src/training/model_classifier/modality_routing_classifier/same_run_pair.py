#!/usr/bin/env python3
"""Join two single-stream run JSONs on row_id (#3856).

Refuses to pair runs whose host identity differs (cpu_model, core_count, ram_gb).
That mismatch is the failure mode that produces a confident wrong Δ p99.

Routing agreement is reported as a pairing diagnostic. Quality metrics stay
with the #3194 contract; this script does not emit a pooled accuracy.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from same_run_harness import host_identity, percentile, summarize


def output_of(record: dict) -> str:
    return (
        record.get("output")
        or record.get("baseline_output")
        or record.get("candidate_output")
    )


def host_of(run: dict) -> dict | None:
    if isinstance(run.get("host"), dict):
        return run["host"]
    nested = run.get("run")
    if isinstance(nested, dict) and isinstance(nested.get("host"), dict):
        return nested["host"]
    return None


def refuse_cross_host(baseline: dict, candidate: dict) -> None:
    left = host_identity(host_of(baseline))
    right = host_identity(host_of(candidate))
    if left != right:
        raise SystemExit(
            "refusing to pair cross-host runs "
            f"(baseline={dict(zip(('cpu_model', 'core_count', 'ram_gb'), left))}, "
            f"candidate={dict(zip(('cpu_model', 'core_count', 'ram_gb'), right))})"
        )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pair baseline and candidate same-run files"
    )
    parser.add_argument(
        "--baseline", type=Path, default=here / "same_run_bert_singlestream.json"
    )
    parser.add_argument(
        "--candidate", type=Path, default=here / "same_run_distilbert_singlestream.json"
    )
    parser.add_argument("--output", type=Path, default=here / "same_run_paired.json")
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    refuse_cross_host(baseline, candidate)

    by_id = {row["row_id"]: row for row in candidate["records"]}
    paired = []
    missing = 0
    agree = 0
    candidate_gold = 0
    baseline_gold = 0
    both_wrong = 0
    for row in baseline["records"]:
        other = by_id.get(row["row_id"])
        if other is None:
            missing += 1
            continue
        base_out = output_of(row)
        cand_out = output_of(other)
        gold = row["label"]
        if base_out == cand_out:
            agree += 1
        elif base_out == gold and cand_out != gold:
            baseline_gold += 1
        elif cand_out == gold and base_out != gold:
            candidate_gold += 1
        else:
            both_wrong += 1
        paired.append(
            {
                "row_id": row["row_id"],
                "qsl_index": row["qsl_index"],
                "input_hash": row["input_hash"],
                "label": gold,
                "baseline_output": base_out,
                "candidate_output": cand_out,
                "baseline_e2e_ms": row["e2e_ms"],
                "candidate_e2e_ms": other["e2e_ms"],
                "baseline_forward_ms": row["forward_ms"],
                "candidate_forward_ms": other["forward_ms"],
                "delta_forward_ms": round(other["forward_ms"] - row["forward_ms"], 3),
            }
        )

    n = len(paired)
    base_fwd = [r["baseline_forward_ms"] for r in paired]
    cand_fwd = [r["candidate_forward_ms"] for r in paired]
    report = {
        "issue": "#3856",
        "scenario": "single-stream-paired",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host_of(baseline),
        "quality": {
            "metric_contract": "https://github.com/vllm-project/semantic-router/issues/3194",
            "note": "This file reports latency deltas and routing agreement only.",
        },
        "baseline": {
            "file": str(args.baseline),
            "model": baseline.get("model"),
            "binding": (baseline.get("run") or {}).get("binding")
            or baseline.get("binding"),
            "run": baseline.get("run"),
        },
        "candidate": {
            "file": str(args.candidate),
            "model": candidate.get("model"),
            "binding": (candidate.get("run") or {}).get("binding")
            or candidate.get("binding"),
            "run": candidate.get("run"),
        },
        "paired": {
            "n": n,
            "missing_row_ids": missing,
            "routing_agreement_rate": round(agree / n, 4) if n else None,
            "num_agree": agree,
            "disagreement_breakdown": {
                "baseline_matched_gold_candidate_wrong": baseline_gold,
                "candidate_matched_gold_baseline_wrong": candidate_gold,
                "both_wrong_different_labels": both_wrong,
            },
            "baseline_forward": summarize(base_fwd),
            "candidate_forward": summarize(cand_fwd),
            "delta_forward_p99_ms": (
                round(percentile(cand_fwd, 99) - percentile(base_fwd, 99), 3)
                if n
                else None
            ),
            "delta_peak_rss_mb": round(
                candidate["run"]["peak_rss_mb"] - baseline["run"]["peak_rss_mb"], 1
            ),
            "delta_cpu_s": round(
                candidate["run"]["cpu_s"] - baseline["run"]["cpu_s"], 3
            ),
        },
        "records": paired,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **report["paired"]}, indent=2))


if __name__ == "__main__":
    main()
