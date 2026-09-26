"""Frozen six-axis JevArena protocol for development and release rankings.

The command consumes completed, identity-bound score reports only. It does
not inspect gold or run models. Release also requires a separately authored,
sealed panel; no model can enter the release ranking without every panel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from decision_bench_v4.bench import SCORE_VERSION as DECISION_BENCH_SCORE_VERSION

from jev_arena.arena import (
    CSS_TASKS,
    FINAL_CSS_ITEMS,
    FINAL_SYNTHETIC_ITEMS,
    PUBLIC_JEVBENCH_ITEMS,
    RELATIONS,
    _load,
    _pareto,
    _score,
    _sha,
)
from jev_arena.jevbench_public import SCORE_VERSION as JEVBENCH_SCORE_VERSION

ARENA_VERSION = "jevarena-ranking/2"
AUTHORED_SCORE_VERSION = "jevarena-authored-score/1"
AXES = (
    "typed",
    "transfer",
    "jevbench_public",
    "decision_bench_v4",
    "sealed_authored",
    "robustness",
)
RELEASE_AUTHORED_MIN = 1200
RELEASE_AUTHORED_MAX = 1480
DEV_AUTHORED_MIN = 100
DEV_AUTHORED_MAX = 250
ROSTER_FIELDS = {
    "key",
    "label",
    "group",
    "model_id",
    "revision",
    "size_b",
    "synthetic_report",
    "css_report",
    "jevbench_public_report",
    "decision_bench_v4_report",
    "sealed_authored_report",
}


def _validate_panel(
    reports: dict[str, dict[str, Any]], phase: str, model_id: str, revision: str
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    synthetic, css, public, dbv4, authored = (
        reports[name] for name in ("synthetic", "css", "public", "dbv4", "authored")
    )
    if synthetic.get("schema_version") != "typed-decision-report/2":
        raise ValueError("Unknown typed synthetic score schema")
    if synthetic.get("split") != ("final" if phase == "release" else "dev"):
        raise ValueError("Typed split does not match JevArena phase")
    if synthetic.get("items") != FINAL_SYNTHETIC_ITEMS:
        raise ValueError("Typed panel must contain 1600 scored items")
    transfer = css.get("roles", {}).get(
        "evaluation" if phase == "release" else "pilot", {}
    )
    expected_css = (FINAL_CSS_ITEMS, CSS_TASKS) if phase == "release" else (1430, 3)
    if (transfer.get("items"), transfer.get("tasks")) != expected_css:
        raise ValueError("Human transfer panel is incomplete")
    if (
        public.get("score_version") != JEVBENCH_SCORE_VERSION
        or public.get("items") != PUBLIC_JEVBENCH_ITEMS
    ):
        raise ValueError("Public JevBench report is not the pinned 231-item panel")
    if (
        dbv4.get("score_version") != DECISION_BENCH_SCORE_VERSION
        or dbv4.get("eligible_items") != 1041
        or dbv4.get("ineligible_items") != 30
    ):
        raise ValueError("Decision Bench v4 must report 1041 eligible and 30 N/E")
    if authored.get("score_version") != AUTHORED_SCORE_VERSION:
        raise ValueError("Unknown sealed authored score version")
    if authored.get("phase") != phase:
        raise ValueError("Sealed authored phase mismatch")
    if authored.get("quality_gate", {}).get("status") != "passed":
        raise ValueError("Sealed authored quality gate has not passed")
    authored_items = authored.get("items")
    bounds = (
        (RELEASE_AUTHORED_MIN, RELEASE_AUTHORED_MAX)
        if phase == "release"
        else (DEV_AUTHORED_MIN, DEV_AUTHORED_MAX)
    )
    if type(authored_items) is not int or not bounds[0] <= authored_items <= bounds[1]:
        raise ValueError("Sealed authored item count is outside frozen bounds")
    groups = authored.get("independent_groups")
    if type(groups) is not int or not 0 < groups <= authored_items:
        raise ValueError("Sealed authored independent group count is missing")
    for name, report in (
        ("synthetic", synthetic),
        ("public", public),
        ("dbv4", dbv4),
        ("authored", authored),
    ):
        identity = report.get("model", {}) if name == "synthetic" else report
        if identity.get("id" if name == "synthetic" else "model_id") != model_id:
            raise ValueError(f"{name}: model identity mismatch")
        if (
            identity.get("revision" if name == "synthetic" else "model_revision")
            != revision
        ):
            raise ValueError(f"{name}: model revision mismatch")
    if set(synthetic.get("pairs", {})) != set(RELATIONS):
        raise ValueError("Three paired robustness relations are required")
    robust = sum(
        _score(synthetic["pairs"][relation].get("joint_accuracy_all"), relation)
        for relation in RELATIONS
    ) / len(RELATIONS)
    axes = {
        "typed": _score(synthetic.get("macro_family_accuracy"), "typed"),
        "transfer": _score(transfer.get("median_task_macro_f1_all"), "transfer"),
        "jevbench_public": _score(public.get("tier_macro_accuracy"), "jevbench"),
        "decision_bench_v4": _score(dbv4.get("task_macro_accuracy"), "dbv4"),
        "sealed_authored": _score(
            authored.get("macro_family_accuracy"), "sealed_authored"
        ),
        "robustness": robust,
    }
    panel_hashes = {
        "synthetic_gold": synthetic.get("gold_sha256"),
        "css_gold": css.get("gold_sha256"),
        "jevbench_prompts": public.get("prompts_sha256"),
        "jevbench_targets": public.get("targets_sha256"),
        "dbv4_prompts": dbv4.get("prompts_sha256"),
        "dbv4_targets": dbv4.get("targets_sha256"),
        "authored_prompts": authored.get("prompts_sha256"),
        "authored_targets": authored.get("targets_sha256"),
    }
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in panel_hashes.values()
    ):
        raise ValueError("Every panel requires immutable SHA-256 digests")
    coverage = {
        "synthetic_items": synthetic["items"],
        "css_items": transfer["items"],
        "jevbench_public_items": public["items"],
        "decision_bench_v4_eligible": dbv4["eligible_items"],
        "decision_bench_v4_ineligible_ne": dbv4["ineligible_items"],
        "sealed_authored_items": authored_items,
        "sealed_authored_independent_groups": groups,
        "effective_text_answers": (
            synthetic["items"]
            + transfer["items"]
            + public["items"]
            + dbv4["eligible_items"]
            + authored_items
        ),
    }
    return axes, panel_hashes, coverage


def rank(manifest_path: Path, phase: str) -> dict[str, Any]:
    manifest = _load(manifest_path)
    if phase not in ("dev", "release") or manifest.get("phase") != phase:
        raise ValueError("Manifest phase must match dev or release")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("JevArena requires at least two same-panel models")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    common: dict[str, str] | None = None
    for entry in models:
        if not isinstance(entry, dict) or set(entry) != ROSTER_FIELDS:
            raise ValueError("Incomplete JevArena v2 model entry")
        key = entry["key"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Missing or duplicate model key")
        seen.add(key)
        size = entry["size_b"]
        if size is not None and (
            type(size) not in (int, float) or not math.isfinite(size) or size <= 0
        ):
            raise ValueError("size_b must be positive actual parameter count or null")
        paths = {
            name: Path(entry[field])
            for name, field in (
                ("synthetic", "synthetic_report"),
                ("css", "css_report"),
                ("public", "jevbench_public_report"),
                ("dbv4", "decision_bench_v4_report"),
                ("authored", "sealed_authored_report"),
            )
        }
        reports = {name: _load(path) for name, path in paths.items()}
        axes, panel_hashes, coverage = _validate_panel(
            reports, phase, entry["model_id"], entry["revision"]
        )
        if common is None:
            common = panel_hashes
        elif panel_hashes != common:
            raise ValueError(f"{key}: panel digest differs from roster")
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
                "coverage": coverage,
                "report_sha256": {name: _sha(path) for name, path in paths.items()},
            }
        )
    rows.sort(key=lambda row: (-row["score"], -row["axes"]["transfer"], row["key"]))
    for position, row in enumerate(rows, 1):
        row["rank"] = position
    _pareto(rows)
    return {
        "schema_version": ARENA_VERSION,
        "phase": phase,
        "manifest_sha256": _sha(manifest_path),
        "panel_sha256": common,
        "policy": {
            "score": "100 times the equal-weight geometric mean of six fractions.",
            "axes": list(AXES),
            "typed": "Choice/Noul/Score family-macro accuracy on 1600 items.",
            "transfer": "Median task macro-F1 over 15 held-out tasks (release), three pilot tasks (dev).",
            "jevbench_public": "Equal easy/standard/hard accuracy on 231 exposed public items; not the official sealed rank.",
            "decision_bench_v4": "Task-macro accuracy on 1041 text-readable items; 30 visual-only are N/E.",
            "sealed_authored": "Choice/Noul/Score family-macro accuracy on independent, frozen authored items.",
            "robustness": "Mean joint-correct pair fraction for counterfactual, order and label variants; variants are not independent items.",
            "side_tables": "Calibration, invalidity, latency, throughput and cost are published separately; performance comparison requires matched hardware/runtime.",
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
    result = rank(args.manifest, args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"models": len(result["models"]), "top": result["models"][0]["key"]})
    )


if __name__ == "__main__":
    main()
