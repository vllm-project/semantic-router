"""Collect and normalize MoM evaluation outputs into result bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.mom_eval.common import (
    REPO_ROOT,
    entrypoint_config,
    load_core_suite,
    load_mom_manifest,
    recipe_digest,
    utc_now_iso,
    write_json,
)
from bench.mom_eval.packs import PackResult, import_pack


def _load_raw_metric(raw_dir: Path, name: str) -> dict[str, Any] | None:
    path = raw_dir / f"{name}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def collect_core_metrics(raw_dir: Path, run_mode: str) -> dict[str, dict[str, Any]]:
    core = load_core_suite()
    metrics: dict[str, dict[str, Any]] = {}
    limit_key = "smoke_limit" if run_mode == "smoke" else "formal_limit"
    classification = "diagnostic" if run_mode == "smoke" else "blocking"

    for layer_name, layer in (core.get("layers") or {}).items():
        if layer_name == "regression":
            continue
        for benchmark in layer.get("benchmarks") or []:
            benchmark_id = str(benchmark["id"])
            payload = _load_raw_metric(raw_dir / "core", benchmark_id)
            metrics[benchmark_id] = {
                "value": None if payload is None else payload.get("value"),
                "unit": "ratio",
                "num_samples": None if payload is None else payload.get("num_samples"),
                "classification": "blocking" if layer.get("blocking") else classification,
                "benchmark_id": benchmark_id,
                "layer": layer_name,
                "missing": payload is None,
            }
        for metric_name in layer.get("metrics") or []:
            payload = _load_raw_metric(raw_dir / "core", str(metric_name))
            metrics[str(metric_name)] = {
                "value": None if payload is None else payload.get("value"),
                "unit": "ms" if "latency" in str(metric_name) else "ratio",
                "classification": classification,
                "layer": layer_name,
                "missing": payload is None,
            }
    return metrics


def collect_baseline_metrics(raw_dir: Path, manifest: dict[str, Any], entrypoint: str) -> list[dict[str, Any]]:
    entry = entrypoint_config(manifest, entrypoint)
    baselines: list[dict[str, Any]] = []
    for baseline in entry.get("baselines") or []:
        role = str(baseline["role"])
        payload = _load_raw_metric(raw_dir / "baselines", role)
        baselines.append(
            {
                "model": baseline["model"],
                "role": role,
                "match": baseline["match"],
                "metrics": {} if payload is None else payload.get("metrics", payload),
            }
        )
    return baselines


def collect_pack_metrics(
    manifest: dict[str, Any], entrypoint: str, raw_dir: Path, run_mode: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    entry = entrypoint_config(manifest, entrypoint)
    pack_versions: list[dict[str, Any]] = []
    pack_metrics: dict[str, dict[str, Any]] = {}
    for pack_id in entry.get("extension_packs") or []:
        pack = import_pack(pack_id)
        steps = pack.plan(manifest, entrypoint, run_mode)
        result: PackResult = pack.collect(raw_dir / "packs" / pack_id.replace("/", "_"), steps)
        pack_versions.append({"id": pack_id, "version": "1.0.0"})
        for metric_id, metric in result.metrics.items():
            pack_metrics[f"{pack_id}:{metric_id}"] = metric
    return pack_versions, pack_metrics


def build_result_bundle(
    manifest_path: Path,
    entrypoint: str,
    raw_dir: Path,
    run_mode: str,
    run_id: str,
) -> dict[str, Any]:
    manifest = load_mom_manifest(manifest_path)
    recipe_dir = manifest_path.parent
    entry = entrypoint_config(manifest, entrypoint)
    mom = manifest["mom"]
    runtime = manifest.get("runtime") or {}

    core_metrics = collect_core_metrics(raw_dir, run_mode)
    pack_versions, pack_metrics = collect_pack_metrics(manifest, entrypoint, raw_dir, run_mode)
    metrics = {**core_metrics, **pack_metrics}

    publication = {
        "publishable": run_mode != "smoke",
        "classification": "diagnostic" if run_mode == "smoke" else "launch",
        "blocking_reasons": [],
    }
    if run_mode == "smoke":
        publication["blocking_reasons"].append("smoke run is diagnostic only")

    missing_blocking = [
        name
        for name, metric in metrics.items()
        if metric.get("classification") == "blocking" and (metric.get("missing") or metric.get("value") is None)
    ]
    if missing_blocking:
        publication["publishable"] = False
        publication["classification"] = "blocked"
        publication["blocking_reasons"].append(
            f"missing blocking metrics: {', '.join(sorted(missing_blocking))}"
        )

    return {
        "schema_version": "vllm-sr/mom-eval-result/v1",
        "identity": {
            "recipe_id": mom["recipe_id"],
            "recipe_version": mom["recipe_version"],
            "entrypoint": entrypoint,
            "objective": entry.get("objective"),
            "recipe_digest": recipe_digest(recipe_dir),
            "pool_membership": (manifest.get("provenance") or {}).get("provider_pool", []),
            "run_mode": run_mode,
            "run_id": run_id,
            "generated_at": utc_now_iso(),
        },
        "contract": {
            "core_suite_version": load_core_suite().get("version", "1.0.0"),
            "extension_packs": pack_versions,
        },
        "environment": {
            "platform": "local-dev",
            "router_image": (manifest.get("provenance") or {}).get("router_image"),
            "generation": runtime.get("generation"),
            "api_url": runtime.get("api_url"),
        },
        "metrics": metrics,
        "baselines": collect_baseline_metrics(raw_dir, manifest, entrypoint),
        "publication": publication,
        "artifacts": {
            "raw_dir": str(raw_dir.relative_to(REPO_ROOT)) if raw_dir.is_relative_to(REPO_ROOT) else str(raw_dir),
            "manifest": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.is_relative_to(REPO_ROOT) else str(manifest_path),
        },
    }


def write_result_bundle(output_path: Path, bundle: dict[str, Any]) -> None:
    write_json(output_path, bundle)
