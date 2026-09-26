"""Frozen multi-panel JevArena ranking and size/score Pareto audit.

This command consumes existing score reports. It never loads prompts, gold,
models, APIs, or unscored predictions. Development panels may be ranked with
``--phase dev``; release requires the independently frozen final panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jev_arena.jevbench_public import SCORE_VERSION as JEVBENCH_SCORE_VERSION

ARENA_VERSION = "jevarena-ranking/1"
RELATIONS = ("counterfactual", "order_invariance", "label_invariance")
AXES = ("typed", "transfer", "authored", "robustness")
FINAL_SYNTHETIC_ITEMS = 1600
FINAL_CSS_ITEMS = 6547
CSS_TASKS = 15
PUBLIC_JEVBENCH_ITEMS = 231


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score(value: Any, name: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name}: expected a finite fraction in [0,1]")
    return float(value)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _axes(
    synthetic: dict[str, Any], css: dict[str, Any], public: dict[str, Any], phase: str
) -> dict[str, float]:
    if synthetic.get("schema_version") != "typed-decision-report/2":
        raise ValueError("Unknown synthetic score schema")
    if phase == "release":
        if (
            synthetic.get("split") != "final"
            or synthetic.get("items") != FINAL_SYNTHETIC_ITEMS
        ):
            raise ValueError("Release requires frozen 1600-item synthetic final")
        transfer = css.get("roles", {}).get("evaluation", {})
        if (
            transfer.get("items") != FINAL_CSS_ITEMS
            or transfer.get("tasks") != CSS_TASKS
        ):
            raise ValueError(
                "Release requires all 15 CSS evaluation tasks and 6547 items"
            )
    else:
        if synthetic.get("split") != "dev":
            raise ValueError("Development rank requires synthetic DEV")
        transfer = css.get("roles", {}).get("pilot", {})
        if transfer.get("tasks") != 3 or transfer.get("items") != 1430:
            raise ValueError("Development rank requires the three CSS pilot tasks")
    if (
        public.get("score_version") != JEVBENCH_SCORE_VERSION
        or public.get("items") != PUBLIC_JEVBENCH_ITEMS
    ):
        raise ValueError("JevBench report is not the pinned 231-item public panel")
    pairs = synthetic.get("pairs", {})
    if set(pairs) != set(RELATIONS):
        raise ValueError("Synthetic paired robustness report is incomplete")
    robust = sum(
        _score(pairs[relation].get("joint_accuracy_all"), relation)
        for relation in RELATIONS
    ) / len(RELATIONS)
    return {
        "typed": _score(synthetic.get("macro_family_accuracy"), "typed"),
        "transfer": _score(transfer.get("median_task_macro_f1_all"), "transfer"),
        "authored": _score(public.get("tier_macro_accuracy"), "authored"),
        "robustness": robust,
    }


def _pareto(rows: list[dict[str, Any]]) -> None:
    eligible = [row for row in rows if row["size_b"] is not None]
    for row in rows:
        row["pareto_frontier"] = (
            None
            if row["size_b"] is None
            else not any(
                other is not row
                and other["size_b"] <= row["size_b"]
                and other["score"] >= row["score"]
                and (other["size_b"] < row["size_b"] or other["score"] > row["score"])
                for other in eligible
            )
        )


def rank(manifest_path: Path, phase: str) -> dict[str, Any]:
    manifest = _load(manifest_path)
    if phase not in ("dev", "release") or manifest.get("phase") != phase:
        raise ValueError("Manifest phase must be dev or release and match request")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("JevArena needs two or more same-panel models")
    rows, seen = [], set()
    common: dict[str, str] = {}
    for entry in models:
        if not isinstance(entry, dict) or set(entry) != {
            "key",
            "label",
            "group",
            "model_id",
            "revision",
            "size_b",
            "synthetic_report",
            "css_report",
            "jevbench_public_report",
        }:
            raise ValueError("Incomplete JevArena model roster entry")
        key = entry["key"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Missing or duplicate model key")
        seen.add(key)
        size = entry["size_b"]
        if size is not None and (
            type(size) not in (int, float) or not math.isfinite(size) or size <= 0
        ):
            raise ValueError("size_b must be a positive finite parameter count or null")
        paths = {
            name: Path(entry[field])
            for name, field in (
                ("synthetic", "synthetic_report"),
                ("css", "css_report"),
                ("public", "jevbench_public_report"),
            )
        }
        reports = {name: _load(path) for name, path in paths.items()}
        synthetic, css, public = (
            reports[name] for name in ("synthetic", "css", "public")
        )
        if (
            synthetic.get("model", {}).get("id") != entry["model_id"]
            or synthetic.get("model", {}).get("revision") != entry["revision"]
            or public.get("model_revision") != entry["revision"]
            or public.get("model_id") not in (entry["model_id"], None)
        ):
            raise ValueError(f"{key}: score report identity/revision mismatch")
        panel_hashes = {
            "synthetic_gold": synthetic.get("gold_sha256"),
            "css_gold": css.get("gold_sha256"),
            "jevbench_prompts": public.get("prompts_sha256"),
            "jevbench_targets": public.get("targets_sha256"),
        }
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in panel_hashes.values()
        ):
            raise ValueError(f"{key}: missing panel digest")
        if not common:
            common = panel_hashes
        elif panel_hashes != common:
            raise ValueError(f"{key}: panel digest differs from roster")
        axes = _axes(synthetic, css, public, phase)
        score = 100 * math.prod(axes.values()) ** (1 / len(AXES))
        rows.append(
            {
                "key": key,
                "label": entry["label"],
                "group": entry["group"],
                "model_id": entry["model_id"],
                "revision": entry["revision"],
                "size_b": float(size) if size is not None else None,
                "axes": axes,
                "score": score,
                "report_sha256": {name: _sha(path) for name, path in paths.items()},
                "coverage": {
                    "synthetic_items": synthetic.get("items"),
                    "css_items": css.get("roles", {})
                    .get("evaluation" if phase == "release" else "pilot", {})
                    .get("items"),
                    "jevbench_public_items": public.get("items"),
                },
            }
        )
    rows.sort(
        key=lambda row: (
            -row["score"],
            -row["axes"]["transfer"],
            -row["axes"]["authored"],
            row["key"],
        )
    )
    for position, row in enumerate(rows, 1):
        row["rank"] = position
    _pareto(rows)
    return {
        "schema_version": ARENA_VERSION,
        "phase": phase,
        "manifest_sha256": _sha(manifest_path),
        "panel_sha256": common,
        "policy": {
            "axes": "Equal-weight geometric mean of four fractions, multiplied by 100.",
            "typed": "Synthetic family-macro accuracy; invalid/missing answers count wrong.",
            "transfer": "Median task macro-F1 over the 15 held-out CSS tasks (release) or three pilot tasks (dev).",
            "authored": "Equal average of easy, standard and hard accuracy on 231 *public* JevBench items.",
            "robustness": "Equal average of joint-correct pair fractions for counterfactual, order and label invariance.",
            "size": "Millions/billions of actual parameters, not model memory or hardware cost; a frontier point has no smaller or equal model with a higher or equal score.",
            "calibration_latency": "Brier, ECE, validity and latency are mandatory side tables, not mixed into the rank; latency only comparable on matched hardware/runtime.",
            "public_caveat": "JevBench public items are exposed. JevArena is independently reproduced and not an official JevBench leaderboard.",
        },
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("dev", "release"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = rank(args.manifest, args.phase)
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
