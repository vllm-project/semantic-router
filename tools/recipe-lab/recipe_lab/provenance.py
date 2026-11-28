"""Provenance emission for recipe-lab runs (Sub-issue 4).

recipe-lab emits a `recipe-lab-run-v1` manifest whose vocabulary aligns with
the upstream provenance schemas (src/training/model_eval/provenance):
schema_version / kind / id / task / dataset_ref / harness / metrics.

It deliberately does NOT fabricate a `router-model-evaluation-v1` manifest:
that schema describes a signal-model artifact evaluation (macro F1,
calibration, abstention) which a recipe-round verdict does not measure.
When a Tier-3 artifact candidate is evaluated for real, model_eval's own
tooling emits the genuine evaluation manifest; this module then only
references it. Honesty over shape-fitting.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .contract import DEFAULT_REPLAY_METRIC
from .ledger import best_replay, scoped_records

MANIFEST_SCHEMA_VERSION = "v1"
MANIFEST_KIND = "recipe-lab-run"


def portable_path(text: str) -> str:
    """Display paths under the account's login or mounted home as ~/... ."""
    if not os.path.isabs(text):
        return text
    home = os.fspath(Path.home())
    path = os.path.normcase(text)
    # Keep the actual executable or candidate name, including symlink names.
    for prefix in (home, os.path.realpath(home)):
        normalized = os.path.normcase(prefix)
        if path == normalized or path.startswith(normalized + os.sep):
            return "~" + text[len(prefix) :]
    return text


def promotion_candidate(candidate: dict) -> dict:
    """Record a portable display ref before deriving the final record ID."""
    ref = candidate.get("ref")
    if candidate.get("kind") == "config" and isinstance(ref, str):
        return {**candidate, "ref": portable_path(ref)}
    return candidate


def _content_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def build_run_manifest(
    ledger_path: str,
    records: list[dict[str, Any]],
    metric_key: str = DEFAULT_REPLAY_METRIC,
) -> dict[str, Any]:
    ledger_file = Path(ledger_path)
    rounds = sorted({r.get("round") for r in records})
    verdicts = [r.get("verdict") for r in records]
    live_records = [r for r in records if r.get("tier") == "live"]
    bench_refs = sorted(
        {ref for r in live_records for ref in (r.get("bench_refs") or [])}
    )
    # A2: the manifest names the same best candidate the report and the
    # agent context rank - the shared pure selection over the newest
    # dataset/split scope, so a flagged or cross-dataset record cannot win.
    best = best_replay(scoped_records(records), metric_key)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "id": f"recipe-lab-run:{ledger_file.stem}",
        "task": "routing-recipe-improvement",
        "dataset_ref": {
            "content_hash": records[0].get("dataset_hash") if records else None,
            "note": "frozen workload from the experiment spec; see the ledger's dataset_hash",
        },
        "harness": {
            "code": "recipe-lab",
            "tool": "vllm-sr benchmark (sr-bench 1.0)",
        },
        "rounds": {
            "count": len(rounds),
            "verdict_counts": {
                verdict: verdicts.count(verdict)
                for verdict in sorted(set(verdicts))
                if verdict is not None
            },
        },
        "evidence": {
            "ledger": {
                "path": str(ledger_file),
                "content_hash": _content_hash(ledger_file),
                "record_ids": sorted(
                    {r["record_id"] for r in records if r.get("record_id")}
                ),
            },
            "bench_refs": bench_refs,
        },
        "best_replay_candidate": (
            None
            if best is None
            else {
                "ref": (best.get("candidate") or {}).get("ref"),
                "score": (best.get("metrics") or {}).get(metric_key),
                "record_id": best.get("record_id"),
            }
        ),
    }


def write_run_manifest(ledger_path: str, output_path: str) -> dict[str, Any]:
    ledger_file = Path(ledger_path)
    records = [
        json.loads(line)
        for line in ledger_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = build_run_manifest(ledger_path, records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return manifest
