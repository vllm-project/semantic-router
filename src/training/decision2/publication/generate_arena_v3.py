"""Generate first-release cards from sealed-core v3 and separate public231 ranks.

The v3 score contains only typed FINAL and the 15-task human transfer panel.
JevBench public is a required, separately ranked diagnostic; neither it nor
authored v3.1 questions enter the v3 headline. This module reads scored reports
only and cannot certify the independent release audit by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jev_arena.arena_v3 import SCORER_SOURCE_PATHS
from jev_arena.compare_v3 import DEFAULT_REPLICATES, DEFAULT_SEED
from jev_arena.render import ranking_svg as public_ranking_svg

from .generate_arena import (
    _difference,
    _digest,
    _fraction,
    _safe,
    _screen_public_text,
    load,
    relative,
    sha_file,
)
from .render_arena_v3 import (
    axis_matrix_svg,
    ranking_svg,
    task_matrix_svg,
)

VERSION = "decision-model-card-artifacts/6"
FIGURES = (
    "jevarena-rank.svg",
    "jevarena-axis-matrix.svg",
    "jevarena-task-matrix.svg",
    "jevbench-public-rank.svg",
)
ARTIFACTS = ("score-table.md", *FIGURES)
PUBLIC_TIERS = ("easy", "standard", "hard")


def matched_models(
    arena: dict[str, Any], public: dict[str, Any]
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    if (
        arena.get("schema_version") != "jevarena-ranking/3"
        or arena.get("phase") != "release"
        or arena.get("status") != "scored_pending_independent_release_audit"
    ):
        raise ValueError(
            "First-release artifacts need the scored v3 sealed-core ranking"
        )
    if (
        public.get("schema_version") != "jevarena-jevbench-public-rank/1"
        or public.get("items") != 231
    ):
        raise ValueError("First-release artifacts need the pinned public231 ranking")
    for key in ("manifest_sha256", "freeze_sha256"):
        _digest(arena.get(key), key)
    panels = arena.get("panel_sha256")
    if not isinstance(panels, dict) or set(panels) != {
        "typed_gold_sha256",
        "css_gold_sha256",
    }:
        raise ValueError("V3 sealed panel digests are incomplete")
    for key, value in panels.items():
        _digest(value, key)
    public_panels = public.get("panel_sha256")
    if not isinstance(public_panels, dict) or set(public_panels) != {
        "prompts_sha256",
        "targets_sha256",
        "panel_manifest_sha256",
    }:
        raise ValueError("Public JevBench panel digests are incomplete")
    for key, value in public_panels.items():
        _digest(value, key)
    arena_rows, public_rows = arena.get("models"), public.get("models")
    if (
        not isinstance(arena_rows, list)
        or len(arena_rows) < 2
        or not isinstance(public_rows, list)
        or len(public_rows) != len(arena_rows)
        or any(not isinstance(row, dict) for row in (*arena_rows, *public_rows))
    ):
        raise ValueError("First release needs a matched multi-model panel")
    arena_keys = [row.get("key") for row in arena_rows]
    public_keys = [row.get("key") for row in public_rows]
    if (
        any(not isinstance(key, str) or not key for key in arena_keys)
        or len(set(arena_keys)) != len(arena_keys)
        or len(set(public_keys)) != len(public_keys)
        or set(arena_keys) != set(public_keys)
    ):
        raise ValueError(
            "V3 and public rankings have different or duplicate model keys"
        )
    by_public = {row["key"]: row for row in public_rows}
    matched = {}
    for row in arena_rows:
        key = row["key"]
        peer = by_public[key]
        for field in ("model_id", "revision", "size_b", "group"):
            if row.get(field) != peer.get(field):
                raise ValueError(f"{key}: model identity or actual size differs")
        size = row.get("size_b")
        if type(size) not in (int, float) or not math.isfinite(size) or size <= 0:
            raise ValueError(f"{key}: actual parameter count is required")
        axes = row.get("axes")
        if not isinstance(axes, dict) or set(axes) != {"typed", "transfer"}:
            raise ValueError(f"{key}: v3 score must contain only two sealed axes")
        expected = 100 * math.sqrt(
            _fraction(axes["typed"], "typed") * _fraction(axes["transfer"], "transfer")
        )
        if type(row.get("score")) not in (int, float) or not math.isclose(
            row["score"], expected, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError(f"{key}: v3 score differs from two sealed axes")
        coverage = row.get("coverage", {})
        if coverage != {
            "typed_items": 1600,
            "css_items": 6547,
            "sealed_core_items": 8147,
        }:
            raise ValueError(f"{key}: incomplete 8,147-item sealed-core coverage")
        tasks = row.get("task_scores", {})
        if (
            set(tasks.get("typed", {})) != {"choice", "noul", "score"}
            or len(tasks.get("transfer", {})) != 15
        ):
            raise ValueError(f"{key}: typed or human task matrix is incomplete")
        reports = row.get("report_sha256")
        if not isinstance(reports, dict) or set(reports) != {"typed", "css"}:
            raise ValueError(f"{key}: sealed score report binding is incomplete")
        for name, value in reports.items():
            _digest(value, f"{key}.{name} report")
        for field in ("adapter_sha256", "calibration_sha256"):
            _digest(row.get(field), f"{key}.{field}")
        if row.get("group") == "decision2":
            _digest(row.get("native_model_sha256"), f"{key}.native_model_sha256")
        elif row.get("native_model_sha256") is not None:
            _digest(row["native_model_sha256"], f"{key}.native_model_sha256")
        public_raw = _fraction(peer.get("accuracy_all"), "public accuracy")
        if (
            type(peer.get("score")) not in (int, float)
            or not math.isclose(
                peer["score"], 100 * public_raw, rel_tol=0, abs_tol=1e-8
            )
            or peer.get("items") != 231
            or type(peer.get("valid")) is not int
            or not 0 <= peer["valid"] <= 231
        ):
            raise ValueError(f"{key}: public231 score or coverage is invalid")
        tiers = peer.get("tiers")
        if not isinstance(tiers, dict) or set(tiers) != set(PUBLIC_TIERS):
            raise ValueError(f"{key}: public231 difficulty tiers are incomplete")
        for tier, value in tiers.items():
            _fraction(value, f"{key}.{tier}")
        _fraction(peer.get("tier_macro_accuracy"), "public tier macro")
        _digest(peer.get("report_sha256"), f"{key}.public report")
        matched[key] = row, peer
    for label, rows, sort_key in (
        (
            "JevArena v3",
            arena_rows,
            lambda item: (-item["score"], -item["axes"]["transfer"], item["key"]),
        ),
        (
            "JevBench public",
            public_rows,
            lambda item: (-item["score"], -item["tier_macro_accuracy"], item["key"]),
        ),
    ):
        if rows != sorted(rows, key=sort_key) or any(
            row.get("rank") != index for index, row in enumerate(rows, 1)
        ):
            raise ValueError(f"{label}: ranking order or rank number is inconsistent")
        for row in rows:
            expected_frontier = not any(
                other is not row
                and other["size_b"] <= row["size_b"]
                and other["score"] >= row["score"]
                and (other["size_b"] < row["size_b"] or other["score"] > row["score"])
                for other in rows
            )
            if row.get("pareto_frontier") is not expected_frontier:
                raise ValueError(f"{label}: parameter Pareto label is inconsistent")
    return matched


def _pair(
    config_path: Path,
    spec: dict[str, Any],
    models: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    arena: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "new",
        "old",
        "typed_comparison",
        "transfer_comparison",
        "joint_comparison",
        "new_typed_report",
        "old_typed_report",
        "new_transfer_report",
        "old_transfer_report",
    }
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Paired comparison has missing or unknown fields")
    new, old = spec["new"], spec["old"]
    if new not in models or old not in models or new == old:
        raise ValueError("Paired comparison names unknown or identical models")
    newer, older = models[new][0], models[old][0]
    if newer["group"] != "decision2" or older["group"] != "decision1":
        raise ValueError("Paired comparison must match Decision 2.0 with 1.0")
    paths = {
        name: relative(config_path, spec[name]) for name in required - {"new", "old"}
    }
    reports = {name: load(path) for name, path in paths.items()}
    for name, row, field in (
        ("new_typed_report", newer, "typed"),
        ("old_typed_report", older, "typed"),
        ("new_transfer_report", newer, "css"),
        ("old_transfer_report", older, "css"),
    ):
        if sha_file(paths[name]) != row["report_sha256"][field]:
            raise ValueError(f"{name}: scorer report differs from v3 ranking")
    typed, transfer = reports["typed_comparison"], reports["transfer_comparison"]
    joint = reports["joint_comparison"]
    if (
        typed.get("schema_version") != "typed-decision-comparison/1"
        or typed.get("split") != "final"
        or typed.get("models")
        != {"left": newer["model_id"], "right": older["model_id"]}
        or typed.get("gold_sha256") != arena["panel_sha256"]["typed_gold_sha256"]
        or typed.get("left_sha256")
        != reports["new_typed_report"].get("predictions_sha256")
        or typed.get("right_sha256")
        != reports["old_typed_report"].get("predictions_sha256")
    ):
        raise ValueError("Typed paired comparison is not bound to v3 predictions")
    if (
        transfer.get("comparison_version") != "css-paired-item-bootstrap/1"
        or transfer.get("model_a") != newer["model_id"]
        or transfer.get("model_b") != older["model_id"]
        or transfer.get("gold_sha256") != arena["panel_sha256"]["css_gold_sha256"]
        or transfer.get("predictions_a_sha256")
        != reports["new_transfer_report"].get("predictions_sha256")
        or transfer.get("predictions_b_sha256")
        != reports["old_transfer_report"].get("predictions_sha256")
    ):
        raise ValueError("Transfer paired comparison is not bound to v3 predictions")
    typed_result = typed.get("family_macro", {})
    transfer_result = transfer.get("evaluation_median_over_15_tasks", {}).get(
        "macro_f1_all", {}
    )
    for point, expected, name in (
        (typed_result.get("left"), newer["axes"]["typed"], "new typed"),
        (typed_result.get("right"), older["axes"]["typed"], "old typed"),
        (transfer_result.get("median_a"), newer["axes"]["transfer"], "new transfer"),
        (transfer_result.get("median_b"), older["axes"]["transfer"], "old transfer"),
    ):
        if abs(_fraction(point, name) - expected) > 1e-9:
            raise ValueError(f"{name}: paired estimate differs from v3 score")
    if (
        typed.get("iterations", 0) < 100
        or transfer.get("bootstrap", {}).get("replicates", 0) < 100
    ):
        raise ValueError("Paired intervals require at least 100 draws")
    typed_ci = typed_result.get("delta_ci95")
    transfer_ci = transfer_result.get("difference_interval95")
    if (
        not isinstance(typed_ci, list)
        or len(typed_ci) != 2
        or not isinstance(transfer_ci, dict)
    ):
        raise ValueError("Paired 95% interval is missing")
    typed_ci = [_difference(value, "typed interval") for value in typed_ci]
    transfer_ci = [
        _difference(transfer_ci.get(name), "transfer interval")
        for name in ("low", "high")
    ]
    if typed_ci[0] > typed_ci[1] or transfer_ci[0] > transfer_ci[1]:
        raise ValueError("Paired 95% interval is reversed")
    if (
        joint.get("schema_version") != "jevarena-v3-paired-aggregate/1"
        or joint.get("models")
        != {"left": newer["model_id"], "right": older["model_id"]}
        or joint.get("typed_gold_sha256") != arena["panel_sha256"]["typed_gold_sha256"]
        or joint.get("css_gold_sha256") != arena["panel_sha256"]["css_gold_sha256"]
        or joint.get("predictions_sha256")
        != {
            "left": {
                "typed": reports["new_typed_report"]["predictions_sha256"],
                "css": reports["new_transfer_report"]["predictions_sha256"],
            },
            "right": {
                "typed": reports["old_typed_report"]["predictions_sha256"],
                "css": reports["old_transfer_report"]["predictions_sha256"],
            },
        }
        or joint.get("replicates") != DEFAULT_REPLICATES
        or joint.get("seed") != DEFAULT_SEED
        or joint.get("source_sha256")
        != {name: sha_file(path) for name, path in SCORER_SOURCE_PATHS.items()}
        or joint.get("panel_sha256")
        != hashlib.sha256(
            json.dumps(
                arena["panel_sha256"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("Joint v3 comparison is not bound to the frozen panel")
    points = joint.get("point", {})
    left, right, delta = (points.get(name, {}) for name in ("left", "right", "delta"))
    for name, field in (("T", "typed"), ("H", "transfer"), ("score", "score")):
        expected_left = newer["axes"][field] if name != "score" else newer["score"]
        expected_right = older["axes"][field] if name != "score" else older["score"]
        if any(
            type(value) not in (int, float)
            or not math.isclose(value, expected, abs_tol=1e-9)
            for value, expected in (
                (left.get(name), expected_left),
                (right.get(name), expected_right),
                (delta.get(name), expected_left - expected_right),
            )
        ):
            raise ValueError("Joint v3 paired points differ from the ranking")
    score_ci = joint.get("ci95", {})
    if (
        not isinstance(score_ci, dict)
        or any(type(score_ci.get(key)) not in (int, float) for key in ("low", "high"))
        or not all(math.isfinite(score_ci[key]) for key in ("low", "high"))
        or score_ci["low"] > score_ci["high"]
    ):
        raise ValueError("Joint v3 score interval is missing or reversed")
    return {
        "new": new,
        "old": old,
        "typed_delta": _difference(
            typed_result.get("delta_left_minus_right"), "typed delta"
        ),
        "typed_ci95": typed_ci,
        "transfer_delta": _difference(
            transfer_result.get("difference_a_minus_b"), "transfer delta"
        ),
        "transfer_ci95": transfer_ci,
        "joint_delta": delta["score"],
        "joint_ci95": [score_ci["low"], score_ci["high"]],
        "report_sha256": {name: sha_file(path) for name, path in paths.items()},
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _delta(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def table(
    arena: dict[str, Any],
    models: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    comparisons: list[dict[str, Any]],
) -> str:
    lines = [
        "# Decision 2.0: JevArena v3 first-release panel",
        "",
        "JevArena v3 scores 8,147 sealed-core items: 1,600 typed FINAL decisions and 6,547 human-labeled items across 15 transfer tasks. Typed FINAL is the macro average across four semantic families; human transfer is the median of 15 task-level macro-F1 scores. The headline is 100 times the geometric mean of those two fractions. Missing and invalid answers count as failures. The JevBench public 231-item rerun is separate and does not enter the JevArena score.",
        "",
        "| JevArena rank | Model | Actual parameters | JevArena v3 | Typed FINAL | Human transfer | JevBench public rank / raw | Easy | Standard | Hard |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in arena["models"]:
        peer = models[row["key"]][1]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["rank"]),
                    _safe(row["label"]),
                    f"{row['size_b']:.3f}B",
                    f"{row['score']:.2f}",
                    _pct(row["axes"]["typed"]),
                    _pct(row["axes"]["transfer"]),
                    f"#{peer['rank']} · {peer['score']:.2f}% ({round(peer['accuracy_all'] * 231)}/231)",
                    *(_pct(peer["tiers"][tier]) for tier in PUBLIC_TIERS),
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Paired Decision 2.0 vs 1.0 differences",
        "",
        "The joint interval resamples independent four-variant typed groups and CSS tasks with paired items. Component intervals are shown separately and do not substitute for the joint interval.",
        "",
        "| Decision 2.0 | Matched 1.0 | JevArena score Δ [95% CI] | Typed Δ [95% CI] | Transfer Δ [95% CI] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for pair in comparisons:
        lines.append(
            "| "
            + " | ".join(
                (
                    _safe(models[pair["new"]][0]["label"]),
                    _safe(models[pair["old"]][0]["label"]),
                    f"{pair['joint_delta']:+.2f} [{pair['joint_ci95'][0]:+.2f}, {pair['joint_ci95'][1]:+.2f}] points",
                    f"{_delta(pair['typed_delta'])} [{_delta(pair['typed_ci95'][0])}, {_delta(pair['typed_ci95'][1])}]",
                    f"{_delta(pair['transfer_delta'])} [{_delta(pair['transfer_ci95'][0])}, {_delta(pair['transfer_ci95'][1])}]",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Source and scope",
        "",
        "The [JevBench public 231-question subset](https://github.com/fstandhartinger/jevbench) is an independent rerun among the displayed models, not a score or official rank on the upstream sealed benchmark. Easy, standard, and hard use the pinned public items. JevArena v3 excludes this public subset, Decision Bench, and prospective authored v3.1 questions from its headline. These sources and versions remain separately attributed.",
        "",
        "Ranks apply only to the displayed same-panel roster. Historical versions are not mixed. Robustness, calibration, language, invalidity, latency, throughput and cost require separate side reports; speed and cost are comparable only under matched hardware and runtime.",
        "",
    ]
    return "\n".join(lines)


def generate(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load(config_path)
    if set(config) != {"arena_rank", "jevbench_public_rank", "comparison_pairs"}:
        raise ValueError(
            "Expected arena_rank, jevbench_public_rank and comparison_pairs"
        )
    arena_path = relative(config_path, config["arena_rank"])
    public_path = relative(config_path, config["jevbench_public_rank"])
    arena, public = load(arena_path), load(public_path)
    models = matched_models(arena, public)
    specs = config["comparison_pairs"]
    if not isinstance(specs, list):
        raise TypeError("comparison_pairs must be a list")
    comparisons = [_pair(config_path, spec, models, arena) for spec in specs]
    if len({item["new"] for item in comparisons}) != len(comparisons):
        raise ValueError("Duplicate Decision 2.0 comparison pair")
    for row in arena["models"]:
        if (
            row["group"] == "decision2"
            and any(
                old["group"] == "decision1"
                and abs(old["size_b"] - row["size_b"]) / row["size_b"] < 0.2
                for old in arena["models"]
            )
            and row["key"] not in {item["new"] for item in comparisons}
        ):
            raise ValueError(
                "Size-matched Decision 1.0 baseline needs paired comparisons"
            )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        products = {
            "score-table.md": table(arena, models, comparisons),
            "jevarena-rank.svg": ranking_svg(arena),
            "jevarena-axis-matrix.svg": axis_matrix_svg(arena),
            "jevarena-task-matrix.svg": task_matrix_svg(arena),
            "jevbench-public-rank.svg": public_ranking_svg(public),
        }
        for name, content in products.items():
            _screen_public_text(content)
            (stage / name).write_text(content, encoding="utf-8")
        manifest = {
            "publication_version": VERSION,
            "phase": "release",
            "arena_schema_version": "jevarena-ranking/3",
            "generator_code_sha256": {
                "generator": sha_file(Path(__file__)),
                "renderer": sha_file(Path(__file__).with_name("render_arena_v3.py")),
            },
            "config_sha256": sha_file(config_path),
            "ranking_sha256": {
                "arena": sha_file(arena_path),
                "jevbench_public": sha_file(public_path),
            },
            "panel_sha256": {
                "sealed_core": arena["panel_sha256"],
                "jevbench_public": public["panel_sha256"],
            },
            "models": [
                {
                    "key": row["key"],
                    "model_id": row["model_id"],
                    "revision": row["revision"],
                    "size_b": row["size_b"],
                    "arena_rank": row["rank"],
                    "jevbench_public_rank": models[row["key"]][1]["rank"],
                }
                for row in arena["models"]
            ],
            "comparison_pairs": comparisons,
            "coverage": {
                "sealed_core_items": 8147,
                "jevbench_public_items": 231,
                "public_items_in_arena_score": 0,
                "authored_items_in_arena_score": 0,
            },
            "artifacts_sha256": {name: sha_file(stage / name) for name in ARTIFACTS},
        }
        manifest_text = (
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        )
        _screen_public_text(manifest_text)
        (stage / "manifest.json").write_text(manifest_text, encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        stage.rename(output_dir)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.config, args.output_dir)
    print(json.dumps({"models": len(result["models"]), "version": VERSION}))


if __name__ == "__main__":
    main()
