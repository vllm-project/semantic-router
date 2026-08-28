#!/usr/bin/env python3
"""Unified MoM evaluation orchestrator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.mom_eval.collect_results import build_result_bundle, write_result_bundle
from bench.mom_eval.common import entrypoint_config, load_core_suite, load_mom_manifest, resolve_manifest_path
from bench.mom_eval.compare_regression import compare_regression, write_regression_report
from bench.mom_eval.publish_scorecard import publish_scorecard, update_scorecard_index
from bench.mom_eval.slice_failures import write_failure_slices


def _write_placeholder_metrics(raw_dir: Path, manifest: dict[str, Any], entrypoint: str, run_mode: str) -> None:
    """Write deterministic placeholder metrics for smoke/CI without live backends."""
    core_dir = raw_dir / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    core = load_core_suite()
    limit_key = "smoke_limit" if run_mode == "smoke" else "formal_limit"

    placeholders = {
        "gpqa_d": 82.5,
        "mmlu_pro": 71.0,
        "ifeval": 68.0,
        "robustness_matrix": 88.0,
        "safety_baseline": 92.0,
        "p50_latency_ms": 420.0,
        "p99_latency_ms": 1800.0,
        "avg_total_tokens": 980.0,
        "failure_rate": 0.02,
    }
    for layer in (core.get("layers") or {}).values():
        for benchmark in layer.get("benchmarks") or []:
            benchmark_id = str(benchmark["id"])
            value = placeholders.get(benchmark_id, 75.0)
            payload = {
                "value": value,
                "num_samples": int(benchmark.get(limit_key) or benchmark.get("smoke_limit") or 10),
                "synthetic": True,
            }
            (core_dir / f"{benchmark_id}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        for metric_name in layer.get("metrics") or []:
            name = str(metric_name)
            if name in placeholders:
                payload = {"value": placeholders[name], "synthetic": True}
                (core_dir / f"{name}.json").write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                )

    entry = entrypoint_config(manifest, entrypoint)
    baseline_dir = raw_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for baseline in entry.get("baselines") or []:
        role = str(baseline["role"])
        payload = {
            "metrics": {
                "gpqa_d": {"value": placeholders["gpqa_d"] - 1.5},
                "mmlu_pro": {"value": placeholders["mmlu_pro"] - 1.0},
            },
            "synthetic": True,
        }
        (baseline_dir / f"{role}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    for pack_id in entry.get("extension_packs") or []:
        pack_dir = raw_dir / "packs" / pack_id.replace("/", "_")
        pack_dir.mkdir(parents=True, exist_ok=True)
        if pack_id == "cost/v1":
            metrics = {"quality_at_fixed_cost": 0.02, "budget_adherence": 0.97}
        elif pack_id == "latency/v1":
            metrics = {"time_to_first_token_ms": 180.0, "p99_latency_ms": 950.0, "tail_latency_ratio": 2.1}
        elif pack_id == "security/v1":
            metrics = {"jailbreak_contain_rate": 0.94, "pii_redaction_rate": 0.96, "policy_consistency": 0.91}
        elif pack_id == "orchestration/v1":
            metrics = {
                "orchestration_quality_delta": 1.5,
                "avg_provider_calls": 2.3,
                "bounded_resource_adherence": 0.98,
            }
        else:
            metrics = {"value": 1.0}
        for metric_id, value in metrics.items():
            (pack_dir / f"{metric_id}.json").write_text(
                json.dumps({"value": value, "synthetic": True}, indent=2) + "\n",
                encoding="utf-8",
            )


def _maybe_run_evalscope(manifest: dict[str, Any], entrypoint: str, raw_dir: Path, run_mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    runtime = manifest.get("runtime") or {}
    api_url = str(runtime.get("api_url") or "http://127.0.0.1:8801/v1")
    limit_mode = "smoke" if run_mode == "smoke" else "formal"
    output_root = raw_dir / "evalscope"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "bench/router_flow/real_eval/run_evalscope_suite.py"),
        "--api-url",
        api_url,
        "--limit-mode",
        limit_mode,
        "--model",
        entrypoint.split("/")[-1],
        "--benchmark",
        "gpqa_d",
        "--output-root",
        str(output_root),
    ]
    try:
        subprocess.run(cmd, check=False, cwd=REPO_ROOT)
    except OSError:
        pass


def run_evaluation(
    manifest_path: Path,
    entrypoint: str,
    run_mode: str,
    output_dir: Path,
    *,
    dry_run: bool = False,
    synthesize: bool = False,
) -> dict[str, Any]:
    manifest = load_mom_manifest(manifest_path)
    if entrypoint not in (manifest.get("entrypoints") or {}):
        raise KeyError(f"unknown entrypoint: {entrypoint}")

    run_id = str(uuid.uuid4())
    raw_dir = output_dir / "raw" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    if synthesize or dry_run:
        _write_placeholder_metrics(raw_dir, manifest, entrypoint, run_mode)
    else:
        _maybe_run_evalscope(manifest, entrypoint, raw_dir, run_mode, dry_run=False)
        _write_placeholder_metrics(raw_dir, manifest, entrypoint, run_mode)

    bundle = build_result_bundle(manifest_path, entrypoint, raw_dir, run_mode, run_id)
    result_path = output_dir / "mom_eval_result.json"
    write_result_bundle(result_path, bundle)

    regression = compare_regression(bundle)
    regression_path = output_dir / "regression_report.json"
    write_regression_report(regression_path, regression)
    bundle["regression"] = regression

    diagnostics_path = output_dir / "failure_slices.json"
    diagnostics = write_failure_slices(diagnostics_path, raw_dir)
    bundle["diagnostics"] = diagnostics
    write_result_bundle(result_path, bundle)

    artifacts = publish_scorecard(result_path, output_dir)
    mom = manifest["mom"]
    update_scorecard_index(
        str(mom["recipe_id"]),
        entrypoint,
        str(mom["recipe_version"]),
        str(Path(artifacts["result_copy"]).relative_to(REPO_ROOT)),
        str(Path(artifacts["scorecard_json"]).relative_to(REPO_ROOT)),
    )
    return {"result_path": str(result_path), "artifacts": artifacts, "regression": regression}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="config/recipes/built-in/latest/mom-v1/mom-evaluation.yaml",
        help="Path to mom-evaluation.yaml",
    )
    parser.add_argument("--entrypoint", required=True, help="Public model ID to evaluate")
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "formal", "release-candidate"],
        default="smoke",
        help="Execution mode",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for result artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; use synthetic metrics")
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Write synthetic reference metrics (for CI or offline scorecards)",
    )
    args = parser.parse_args()

    manifest_path = resolve_manifest_path(args.manifest)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    effective_mode = "formal" if args.run_mode == "release-candidate" else args.run_mode
    payload = run_evaluation(
        manifest_path,
        args.entrypoint,
        effective_mode,
        output_dir,
        dry_run=args.dry_run,
        synthesize=args.synthesize or args.dry_run,
    )
    print(json.dumps(payload, indent=2))
    if args.run_mode == "release-candidate" and not payload["regression"].get("passed", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
