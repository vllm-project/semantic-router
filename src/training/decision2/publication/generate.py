"""Generate auditable model-card tables and figures from frozen score reports."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from transfer.build import sha_file

from .load import load_inputs
from .render import render_markdown, render_matrix_svg, render_ranking_svg

ARTIFACTS = ("score-table.md", "ranking.svg", "matrix.svg")


def _manifest(data: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    models = []
    for model in data["models"]:
        bench = model["benchmark"]
        entry: dict[str, Any] = {
            "key": model["key"],
            "label": model["label"],
            "group": model["group"],
            "size_display": model.get("size"),
            "identity": bench["model"],
            "benchmark": {
                "report_filename": model["benchmark_path"].name,
                "report_sha256": model["benchmark_file_sha256"],
                "predictions_sha256": bench["predictions_sha256"],
                "family_macro_accuracy": bench["macro_family_accuracy"],
                "overall_accuracy_all": bench["overall"]["accuracy_all"],
                "invalid_or_missing_n": bench["overall"]["invalid_or_missing_n"],
            },
        }
        if model["css"] is not None:
            css = model["css"]
            entry["css"] = {
                "report_filename": model["css_path"].name,
                "report_sha256": model["css_file_sha256"],
                "predictions_sha256": css["predictions_sha256"],
                "evaluation_median_task_macro_f1_all": css["roles"]["evaluation"][
                    "median_task_macro_f1_all"
                ],
                "evaluation_median_task_accuracy_all": css["roles"]["evaluation"][
                    "median_task_accuracy_all"
                ],
            }
        models.append(entry)
    comparisons = []
    for pair in data["pairs"]:
        entry = {"new": pair["new"]["key"], "old": pair["old"]["key"]}
        if pair["css_comparison"] is not None:
            comparison = pair["css_comparison"]
            entry["css_paired_bootstrap"] = {
                "report_filename": pair["css_comparison_path"].name,
                "report_sha256": pair["css_comparison_file_sha256"],
                "bootstrap": comparison.get("bootstrap"),
                "prediction_a_sha256": comparison["predictions_a_sha256"],
                "prediction_b_sha256": comparison["predictions_b_sha256"],
            }
        comparisons.append(entry)
    return {
        "publication_version": "decision-model-card-artifacts/2",
        "config_sha256": data["config_sha256"],
        "generator_code_sha256": {
            name: sha_file(Path(__file__).with_name(name))
            for name in ("generate.py", "load.py", "render.py")
        },
        "metric_policy": {
            "benchmark_headline": "Unweighted mean of four final-family accuracy_all values; invalid/missing count as misses",
            "benchmark_uncertainty": "No confidence interval supplied by frozen scorer; point estimates only",
            "ranking_scope": "Only models explicitly listed in the generator configuration",
            "css_headline": "Median macro_f1_all and accuracy_all over 15 CSS evaluation tasks; pilot tasks excluded",
            "css_paired_interval": "Shown only when a matching transfer.compare report is supplied",
        },
        "frozen_benchmark": {
            "split": "final",
            "gold_sha256": data["gold_sha256"],
            "items": data["items"],
            "questions": data["questions"],
        },
        "css_gold_sha256": data["css_gold_sha256"],
        "models": models,
        "comparison_pairs": comparisons,
        "artifacts_sha256": {name: sha_file(artifact_dir / name) for name in ARTIFACTS},
    }


def generate(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate inputs before creating a new output directory; never overwrite."""
    data = load_inputs(config_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        (temporary / "score-table.md").write_text(
            render_markdown(data), encoding="utf-8"
        )
        (temporary / "ranking.svg").write_text(
            render_ranking_svg(data), encoding="utf-8"
        )
        (temporary / "matrix.svg").write_text(render_matrix_svg(data), encoding="utf-8")
        manifest = _manifest(data, temporary)
        (temporary / "manifest.json").write_text(
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(output_dir)
        temporary.rename(output_dir)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON config containing report paths and display metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for model-card artifacts",
    )
    args = parser.parse_args()
    manifest = generate(args.config, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "models": len(manifest["models"]),
                "gold_sha256": manifest["frozen_benchmark"]["gold_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
