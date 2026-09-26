"""Rank exact same-panel JevBench public reruns and find size/quality frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jev_arena.arena import _pareto
from jev_arena.jevbench_public import SCORE_VERSION

RANK_VERSION = "jevarena-jevbench-public-rank/1"


def rank(manifest_path: Path, report_root: Path | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("models")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("Public-only rank needs at least two models")
    rows, seen, panel = [], set(), None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "key",
            "label",
            "group",
            "model_id",
            "revision",
            "size_b",
            "report",
        }:
            raise ValueError("Public-only rank roster entry has missing/unknown fields")
        if entry["key"] in seen or not isinstance(entry["key"], str):
            raise ValueError("Duplicate or invalid public-only model key")
        seen.add(entry["key"])
        path = Path(entry["report"])
        if not path.is_absolute():
            path = (
                report_root if report_root is not None else manifest_path.parent
            ) / path
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("score_version") != SCORE_VERSION
            or report.get("items") != 231
            or report.get("model_revision") != entry["revision"]
            or report.get("model_id") not in (entry["model_id"], None)
        ):
            raise ValueError(f"{entry['key']}: report identity or panel size mismatch")
        digests = {
            field: report.get(field)
            for field in ("prompts_sha256", "targets_sha256", "panel_manifest_sha256")
        }
        if any(
            not isinstance(value, str) or len(value) != 64 for value in digests.values()
        ):
            raise ValueError("Public panel has missing digest")
        if panel is None:
            panel = digests
        elif panel != digests:
            raise ValueError("Public panel differs across models")
        size = entry["size_b"]
        if size is not None and (
            type(size) not in (int, float) or not math.isfinite(size) or size <= 0
        ):
            raise ValueError("Size must be positive finite billions or null")
        score = report["accuracy_all"]
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Public accuracy is invalid")
        rows.append(
            {
                "key": entry["key"],
                "label": entry["label"],
                "group": entry["group"],
                "model_id": entry["model_id"],
                "revision": entry["revision"],
                "size_b": float(size) if size is not None else None,
                "score": 100 * score,
                "accuracy_all": score,
                "tier_macro_accuracy": report["tier_macro_accuracy"],
                "tiers": {
                    tier: report["tiers"][tier]["accuracy_all"]
                    for tier in ("easy", "standard", "hard")
                },
                "valid": report["valid"],
                "items": report["items"],
                "brier_valid": report.get("brier_valid"),
                "ece_pmax_15": report.get("ece_pmax_15"),
                "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    rows.sort(key=lambda row: (-row["score"], -row["tier_macro_accuracy"], row["key"]))
    for position, row in enumerate(rows, 1):
        row["rank"] = position
    _pareto(rows)
    return {
        "schema_version": RANK_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "panel_sha256": panel,
        "items": 231,
        "scope": "Independent public-only JevBench rerun; not official sealed/composite rank",
        "headline": "Raw all-item accuracy; invalid/missing count wrong",
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--report-root",
        type=Path,
        help="Base directory for relative report paths; defaults to manifest directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = rank(args.manifest, args.report_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"models": len(report["models"]), "top": report["models"][0]["key"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
