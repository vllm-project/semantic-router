"""Build release-only Decision 2.0 tables and charts from matched rankings.

This command cannot produce a release artifact from a development ranking or
from mismatched JevArena and JevBench-public panels. It does not read prompts,
gold, model weights or private training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jev_arena.render import matrix_svg, pareto_svg, ranking_svg, task_matrix_svg

VERSION = "decision-model-card-artifacts/3"
FIGURES = (
    "jevarena-rank.svg",
    "jevarena-pareto.svg",
    "jevarena-axis-matrix.svg",
    "jevarena-task-matrix.svg",
    "jevbench-public-rank.svg",
    "jevbench-public-pareto.svg",
)
ARTIFACTS = ("score-table.md", *FIGURES)
SHA = re.compile(r"[a-f0-9]{64}\Z")
SECRET = re.compile(
    r"(?i)(?:\bhf_[A-Za-z0-9]{20,}|\bjv_live_[A-Za-z0-9_-]{16,}"
    r"|\bapikey_[A-Za-z0-9_-]{16,}|\bsk-[A-Za-z0-9_-]{16,}"
    r"|Authorization\s*:\s*Bearer\s+\S+)"
)
HOST_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:home|root|data|work|mnt|tmp|private|Users|var|opt)/[^\s\"'<>]+"
    r"|\b[A-Za-z]:[\\/](?:Users|Documents|ProgramData|Windows)[\\/][^\s\"'<>]+"
)
IP_ADDRESS = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name}: expected a JSON object")
    return value


def relative(config_path: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Report paths must be nonempty and relative to config")
    candidate = (config_path.parent / name).resolve()
    if not candidate.is_relative_to(config_path.parent.resolve()):
        raise ValueError("Report path escapes config directory")
    return candidate


def _fraction(value: Any, name: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name}: expected finite fraction")
    return float(value)


def _difference(value: Any, name: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not -1 <= value <= 1
    ):
        raise ValueError(f"{name}: expected finite fraction difference")
    return float(value)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f"{name}: expected SHA-256")
    return value


def _safe(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _screen_public_text(content: str) -> None:
    if (
        SECRET.search(content)
        or HOST_PATH.search(content)
        or IP_ADDRESS.search(content)
    ):
        raise ValueError(
            "Generated card artifact contains private infrastructure or credential-like text"
        )


def matched_models(arena: dict[str, Any], public: dict[str, Any]) -> dict[str, Any]:
    if (
        arena.get("schema_version") != "jevarena-ranking/2"
        or arena.get("phase") != "release"
    ):
        raise ValueError("Publication needs a completed six-axis release ranking")
    if (
        public.get("schema_version") != "jevarena-jevbench-public-rank/1"
        or public.get("items") != 231
    ):
        raise ValueError("Publication needs the pinned 231-item public ranking")
    panel = public.get("panel_sha256", {})
    frozen = arena.get("panel_sha256", {})
    if not isinstance(panel, dict) or not isinstance(frozen, dict):
        raise TypeError("Missing panel digests")
    for first, second in (
        ("prompts_sha256", "jevbench_prompts"),
        ("targets_sha256", "jevbench_targets"),
    ):
        if _digest(panel.get(first), first) != _digest(frozen.get(second), second):
            raise ValueError("JevBench public panels differ between rankings")
    for key in (
        "synthetic_gold",
        "css_gold",
        "dbv4_prompts",
        "dbv4_targets",
        "authored_prompts",
        "authored_targets",
    ):
        _digest(frozen.get(key), key)
    arena_rows = arena.get("models")
    public_rows = public.get("models")
    if (
        not isinstance(arena_rows, list)
        or not isinstance(public_rows, list)
        or len(arena_rows) < 2
    ):
        raise ValueError("A matched multi-model panel is required")
    by_public = {row.get("key"): row for row in public_rows if isinstance(row, dict)}
    if len(by_public) != len(arena_rows) or set(by_public) != {
        row.get("key") for row in arena_rows
    }:
        raise ValueError("The two rankings must contain the same unique model keys")
    selected = {}
    for row in arena_rows:
        peer = by_public[row["key"]]
        axes = row.get("axes", {})
        required_axes = (
            "typed",
            "transfer",
            "jevbench_public",
            "decision_bench_v4",
            "sealed_authored",
            "robustness",
        )
        if set(axes) != set(required_axes):
            raise ValueError(f"{row['key']}: six-axis score is incomplete")
        expected_score = 100 * math.prod(
            _fraction(axes[key], key) for key in required_axes
        ) ** (1 / 6)
        if abs(row.get("score", -1) - expected_score) > 1e-8:
            raise ValueError(f"{row['key']}: JevArena aggregate differs from axes")
        raw = _fraction(peer.get("accuracy_all"), "public accuracy")
        if abs(peer.get("score", -1) - 100 * raw) > 1e-8:
            raise ValueError(f"{row['key']}: public aggregate differs from accuracy")
        for field in ("model_id", "revision", "size_b", "group"):
            if row.get(field) != peer.get(field):
                raise ValueError(f"{row['key']}: model identity or actual size differs")
        if row.get("report_sha256", {}).get("public") != peer.get("report_sha256"):
            raise ValueError(f"{row['key']}: public scorer report differs")
        if (
            abs(
                _fraction(row.get("axes", {}).get("jevbench_public"), "tier macro")
                - _fraction(peer.get("tier_macro_accuracy"), "tier macro")
            )
            > 1e-9
        ):
            raise ValueError(f"{row['key']}: tier-macro score differs")
        coverage = row.get("coverage", {})
        if (
            coverage.get("synthetic_items") != 1600
            or coverage.get("css_items") != 6547
            or coverage.get("jevbench_public_items") != 231
            or coverage.get("decision_bench_v4_eligible") != 1041
            or coverage.get("decision_bench_v4_ineligible_ne") != 30
            or not 1200 <= coverage.get("sealed_authored_items", 0) <= 1480
        ):
            raise ValueError(f"{row['key']}: incomplete release coverage")
        expected = 1600 + 6547 + 231 + 1041 + coverage["sealed_authored_items"]
        if coverage.get("effective_text_answers") != expected:
            raise ValueError(f"{row['key']}: wrong effective text answer count")
        if not isinstance(row.get("size_b"), (int, float)) or row["size_b"] <= 0:
            raise ValueError(f"{row['key']}: actual parameter count is required")
        selected[row["key"]] = (row, peer)
    return selected


def _pair(
    config_path: Path,
    spec: dict[str, Any],
    models: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    arena: dict[str, Any],
) -> dict[str, Any]:
    if set(spec) != {
        "new",
        "old",
        "typed_comparison",
        "transfer_comparison",
        "new_typed_report",
        "old_typed_report",
        "new_transfer_report",
        "old_transfer_report",
    }:
        raise ValueError("Comparison pair has missing or unknown fields")
    new, old = spec["new"], spec["old"]
    if new not in models or old not in models or new == old:
        raise ValueError("Comparison keys must name two ranked models")
    newer, older = models[new][0], models[old][0]
    if newer["group"] != "decision2" or older["group"] != "decision1":
        raise ValueError("Comparison must pair Decision 2.0 with Decision 1.0")
    paths = {
        name: relative(config_path, spec[name])
        for name in spec
        if name not in {"new", "old"}
    }
    reports = {name: load(path) for name, path in paths.items()}
    for name, row, field in (
        ("new_typed_report", newer, "synthetic"),
        ("old_typed_report", older, "synthetic"),
        ("new_transfer_report", newer, "css"),
        ("old_transfer_report", older, "css"),
    ):
        if sha_file(paths[name]) != row["report_sha256"][field]:
            raise ValueError(f"{name}: scorer report differs from ranked model")
    typed = reports["typed_comparison"]
    transfer = reports["transfer_comparison"]
    if (
        typed.get("schema_version") != "typed-decision-comparison/1"
        or typed.get("split") != "final"
        or typed.get("models")
        != {"left": newer["model_id"], "right": older["model_id"]}
        or typed.get("gold_sha256") != arena["panel_sha256"]["synthetic_gold"]
        or typed.get("left_sha256")
        != reports["new_typed_report"].get("predictions_sha256")
        or typed.get("right_sha256")
        != reports["old_typed_report"].get("predictions_sha256")
    ):
        raise ValueError("Typed paired comparison is not bound to ranked predictions")
    if (
        transfer.get("comparison_version") != "css-paired-item-bootstrap/1"
        or transfer.get("model_a") != newer["model_id"]
        or transfer.get("model_b") != older["model_id"]
        or transfer.get("gold_sha256") != arena["panel_sha256"]["css_gold"]
        or transfer.get("predictions_a_sha256")
        != reports["new_transfer_report"].get("predictions_sha256")
        or transfer.get("predictions_b_sha256")
        != reports["old_transfer_report"].get("predictions_sha256")
    ):
        raise ValueError(
            "Transfer paired comparison is not bound to ranked predictions"
        )
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
        if abs(_fraction(point, name) - _fraction(expected, name)) > 1e-9:
            raise ValueError(f"{name}: paired estimate differs from ranked score")
    if (
        typed.get("iterations", 0) < 100
        or transfer.get("bootstrap", {}).get("replicates", 0) < 100
    ):
        raise ValueError("Paired confidence intervals need at least 100 draws")
    typed_interval = typed_result.get("delta_ci95")
    transfer_interval = transfer_result.get("difference_interval95", {})
    if not isinstance(typed_interval, list) or len(typed_interval) != 2:
        raise ValueError("Typed paired 95% interval is missing")
    typed_interval = [_difference(value, "typed interval") for value in typed_interval]
    if not isinstance(transfer_interval, dict):
        raise TypeError("Transfer paired 95% interval is missing")
    transfer_interval = [transfer_interval.get("low"), transfer_interval.get("high")]
    for name, interval in (("typed", typed_interval), ("transfer", transfer_interval)):
        interval = [_difference(value, f"{name} interval") for value in interval]
        if interval[0] > interval[1]:
            raise ValueError(f"{name}: reversed paired 95% interval")
    return {
        "new": new,
        "old": old,
        "typed_delta": _difference(
            typed_result["delta_left_minus_right"], "typed difference"
        ),
        "typed_ci95": typed_interval,
        "transfer_delta": _difference(
            transfer_result["difference_a_minus_b"], "transfer difference"
        ),
        "transfer_ci95": transfer_interval,
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
    rows = [
        "# Decision 2.0: matched release panel",
        "",
        "All models below use identical frozen items and scoring versions. Invalid or missing answers count as incorrect. JevArena is the six-axis geometric mean; JevBench public uses raw accuracy over 231 exposed questions.",
        "",
        "| JevArena rank | Model | Actual parameters | JevArena | Typed | Human transfer | JevBench tier macro | Decision Bench text | Sealed authored | Paired robustness | JevBench public rank and raw |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in arena["models"]:
        peer = models[row["key"]][1]
        axes = row["axes"]
        cells = [
            str(row["rank"]),
            _safe(row["label"]),
            f"{row['size_b']:.3f}B",
            f"{row['score']:.2f}",
            *(
                _pct(axes[key])
                for key in (
                    "typed",
                    "transfer",
                    "jevbench_public",
                    "decision_bench_v4",
                    "sealed_authored",
                    "robustness",
                )
            ),
            f"#{peer['rank']} · {peer['score']:.2f}% ({round(peer['accuracy_all'] * 231)}/231)",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    coverage = arena["models"][0]["coverage"]
    rows += [
        "",
        f"Coverage per model: {coverage['effective_text_answers']:,} effective text answers; 1,600 typed decisions from 400 independent groups, 6,547 human transfer items across 15 tasks, 231 public JevBench items, 1,041 text-readable Decision Bench v4 items, and {coverage['sealed_authored_items']:,} independently authored sealed items. The 30 image-required Decision Bench items are N/E for text-only models. Paired variants are not additional independent questions.",
        "",
        "## Paired Decision 2.0 vs 1.0 differences",
        "",
        "The intervals resample four-variant groups for typed decisions and paired items within each human transfer task. They do not describe the six-axis aggregate unless a separate aggregate interval is supplied.",
        "",
        "| Decision 2.0 | Matched 1.0 | Typed Δ [95% CI] | Human transfer Δ [95% CI] |",
        "| --- | --- | ---: | ---: |",
    ]
    for pair in comparisons:
        rows.append(
            "| "
            + " | ".join(
                (
                    _safe(models[pair["new"]][0]["label"]),
                    _safe(models[pair["old"]][0]["label"]),
                    f"{_delta(pair['typed_delta'])} [{_delta(pair['typed_ci95'][0])}, {_delta(pair['typed_ci95'][1])}]",
                    f"{_delta(pair['transfer_delta'])} [{_delta(pair['transfer_ci95'][0])}, {_delta(pair['transfer_ci95'][1])}]",
                )
            )
            + " |"
        )
    rows += [
        "",
        "## Source and scoring notes",
        "",
        "The JevArena headline contains independently scored public [JevBench v1.2 exposed questions](https://github.com/fstandhartinger/jevbench) and [Decision Bench bench-v4](https://github.com/atlanai/decision-bench) subsets; their provenance and versions must remain visible. The public JevBench rank is an independent rerun among the models shown here, not an official score on closed questions. Sealed authored questions must pass the separate editorial and overlap gate before this report exists.",
        "",
        "Ranks and Pareto status apply only to the displayed, same-panel model roster. Historical scores from different versions are not mixed into these figures. Calibration, invalidity, latency, throughput and cost require separate side tables; speed and cost are comparable only under matched hardware and runtime.",
        "",
    ]
    return "\n".join(rows)


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
    if len({entry["new"] for entry in comparisons}) != len(comparisons):
        raise ValueError("Duplicate Decision 2.0 comparison pair")
    if any(
        row["group"] == "decision2"
        and row["key"] not in {entry["new"] for entry in comparisons}
        and any(
            old["group"] == "decision1"
            and abs(old["size_b"] - row["size_b"]) / row["size_b"] < 0.2
            for old in arena["models"]
        )
        for row in arena["models"]
    ):
        raise ValueError(
            "A size-matched Decision 1.0 baseline needs a paired comparison"
        )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        products = {
            "score-table.md": table(arena, models, comparisons),
            "jevarena-rank.svg": ranking_svg(arena),
            "jevarena-pareto.svg": pareto_svg(arena),
            "jevarena-axis-matrix.svg": matrix_svg(arena),
            "jevarena-task-matrix.svg": task_matrix_svg(arena),
            "jevbench-public-rank.svg": ranking_svg(public),
            "jevbench-public-pareto.svg": pareto_svg(public),
        }
        for name, content in products.items():
            _screen_public_text(content)
            (temporary / name).write_text(content, encoding="utf-8")
        manifest = {
            "publication_version": VERSION,
            "phase": "release",
            "generator_code_sha256": sha_file(Path(__file__)),
            "config_sha256": sha_file(config_path),
            "ranking_sha256": {
                "arena": sha_file(arena_path),
                "jevbench_public": sha_file(public_path),
            },
            "panel_sha256": arena["panel_sha256"],
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
            "coverage": arena["models"][0]["coverage"],
            "artifacts_sha256": {
                name: sha_file(temporary / name) for name in ARTIFACTS
            },
        }
        rendered_manifest = (
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        )
        _screen_public_text(rendered_manifest)
        (temporary / "manifest.json").write_text(rendered_manifest, encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        temporary.rename(output_dir)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.config, args.output_dir)
    print(
        json.dumps(
            {"models": len(result["models"]), "version": result["publication_version"]}
        )
    )


if __name__ == "__main__":
    main()
