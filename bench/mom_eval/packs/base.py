"""Base helpers for extension pack implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.mom_eval.packs import PackResult, RunStep, load_pack_manifest


class BasePack:
    pack_id: str = ""

    def validate(self, manifest: dict[str, Any], entrypoint: str) -> list[str]:
        errors: list[str] = []
        entry = (manifest.get("entrypoints") or {}).get(entrypoint)
        if entry is None:
            return [f"entrypoint {entrypoint!r} missing from manifest"]
        declared = entry.get("extension_packs") or []
        if self.pack_id not in declared:
            errors.append(f"entrypoint {entrypoint!r} does not declare pack {self.pack_id}")
        return errors

    def _plan_from_manifest(self, run_mode: str) -> list[RunStep]:
        spec = load_pack_manifest(self.pack_id)
        limit_key = "smoke_limit" if run_mode == "smoke" else "formal_limit"
        steps: list[RunStep] = []
        for dataset in spec.get("datasets") or []:
            limit = int(dataset.get(limit_key) or dataset.get("smoke_limit") or 10)
            steps.append(
                RunStep(
                    step_id=str(dataset["id"]),
                    harness=str(dataset.get("harness") or dataset.get("benchmark_id") or ""),
                    benchmark_id=str(dataset.get("benchmark_id") or dataset["id"]),
                    metric=str(dataset.get("metric") or "value"),
                    limit=limit,
                    params=dict(dataset),
                )
            )
        return steps

    def collect_placeholder(self, raw_dir: Path, steps: list[RunStep]) -> PackResult:
        metrics: dict[str, dict[str, Any]] = {}
        spec = load_pack_manifest(self.pack_id)
        for metric in spec.get("metrics") or []:
            metric_id = str(metric["id"])
            result_path = raw_dir / f"{metric_id}.json"
            if result_path.is_file():
                import json

                payload = json.loads(result_path.read_text(encoding="utf-8"))
                metrics[metric_id] = {
                    "value": payload.get("value"),
                    "unit": metric.get("unit"),
                    "classification": "blocking" if metric.get("blocking") else "diagnostic",
                }
            else:
                metrics[metric_id] = {
                    "value": None,
                    "unit": metric.get("unit"),
                    "classification": "diagnostic",
                    "missing": True,
                }
        return PackResult(pack_id=self.pack_id, metrics=metrics, artifacts={"raw_dir": str(raw_dir)})
