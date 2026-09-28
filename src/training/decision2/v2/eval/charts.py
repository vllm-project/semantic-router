"""Product-card charts from same-panel reports only (no Pareto chart on cards).

Builds the JevArena v3 rank chart, the model x task matrix (Choice/Noul/Score
plus the 15 human-transfer tasks) and the separate JevBench public-231 rank chart
with the existing publication renderers. Every input must be a
``dev2-same-panel-report/1`` produced on identical panel and scorer hashes;
Decision Index or historical numbers cannot enter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2.eval.same_panel import REPORT_SCHEMA, sha_file, utc_now, write_json

GROUPS = {
    "decision1": "decision1",
    "decision2": "decision2",
    "peer": "open",
    "hosted": "hosted",
}
# Licence class of each peer's weights; own models are "own". Cards include only the
# allowed classes (default: own and permissive); unknown peers are refused.
LICENCE_CLASS = {
    "Hanno-Labs/bosun-v3.1-0.6b": "permissive",
    "Hanno-Labs/bosun-v3.1-1.7b": "permissive",
    "fastino/GLiNER2.5-Decide": "permissive",
    "Mapika/decider-2b": "permissive",
    "Mapika/decider-4b": "permissive",
    "denis-pplx/autojev-27b": "permissive",
    "jaredpalmer/kev-0.8b": "permissive",
    "internlm/Intern-Decision-0.8B": "permissive",
    "flock-io/this-that-model-1.2": "permissive",
    "michaljach/jet": "permissive",
    "bespokelabs/Bespoke-Nimble-9B-v2": "permissive",
    "frontier-infra/jebadiah-27b": "permissive",
    "caiovicentino1/Eikos-27B-FP8": "permissive",
    "caiovicentino1/Eikos-27B": "permissive",
    "kirp/jpt-0.8b": "non-commercial",
    "kirp/jpt-4b": "non-commercial",
    "kirp/jpt-9b": "non-commercial",
    "HopitAI/hopper-g": "research-only",
}
DEFAULT_LICENCES = ("own", "permissive")
SCORER_KEYS = (
    "benchmark/score.py",
    "transfer/score.py",
    "jev_arena/jevbench_public.py",
)


def licence_class(report: dict[str, Any]) -> str:
    if report["model"].get("family") in ("decision1", "decision2"):
        return "own"
    model_id = report["model"].get("model_id")
    if model_id not in LICENCE_CLASS:
        raise ValueError(f"{report['model']['label']}: no licence class for {model_id}")
    return LICENCE_CLASS[model_id]


def filter_licences(
    reports: list[dict[str, Any]], allowed: tuple[str, ...] = DEFAULT_LICENCES
) -> tuple[list[dict[str, Any]], list[str]]:
    kept, excluded = [], []
    for report in reports:
        (kept if licence_class(report) in allowed else excluded).append(report)
    return kept, [f"{r['model']['label']} ({licence_class(r)})" for r in excluded]


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(reports) < 2:
        raise ValueError("charts need at least two same-panel reports")
    for report in reports:
        if report.get("schema") != REPORT_SCHEMA or "v3" not in report:
            raise ValueError(
                f"{report.get('model', {}).get('label')}: not a complete same-panel report"
            )
    reference = reports[0]
    for report in reports[1:]:
        if report.get("panel_sha256") != reference.get("panel_sha256"):
            raise ValueError(f"{report['model']['label']}: different panel files")
        for key in SCORER_KEYS:
            if report["sources"].get(key) != reference["sources"].get(key):
                raise ValueError(f"{report['model']['label']}: different scorer {key}")
    return reports


def _key(report: dict[str, Any]) -> str:
    return report["model"].get("model_id") or report["model"]["label"]


def v3_ranking(reports: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(reports, key=lambda r: -r["v3"]["score"])
    models = []
    for rank, report in enumerate(ordered, 1):
        typed = report["panels"]["typed-final"]
        tasks = report["panels"]["css15"]["tasks"]
        models.append(
            {
                "key": _key(report),
                "label": report["model"]["label"],
                "group": GROUPS.get(report["model"].get("family"), "other"),
                "rank": rank,
                "score": report["v3"]["score"],
                "axes": {"typed": report["v3"]["T"], "transfer": report["v3"]["H"]},
                "task_scores": {
                    "typed": {
                        kind: typed["by_type"][kind]["accuracy"]
                        for kind in ("choice", "noul", "score")
                    },
                    "transfer": {
                        task: value["macro_f1"] for task, value in tasks.items()
                    },
                },
            }
        )
    return {
        "schema_version": "jevarena-ranking/3",
        "scope": "post-key same-panel",
        "models": models,
    }


def public_ranking(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in reports if "public231" in r["panels"]]
    ordered = sorted(rows, key=lambda r: -r["panels"]["public231"]["correct"])
    return {
        "schema_version": "jevarena-jevbench-public-rank/1",
        "models": [
            {
                "key": _key(report),
                "label": report["model"]["label"],
                "group": GROUPS.get(report["model"].get("family"), "other"),
                "rank": rank,
                "score": 100
                * report["panels"]["public231"]["correct"]
                / report["panels"]["public231"]["items"],
            }
            for rank, report in enumerate(ordered, 1)
        ],
    }


def render(
    paths: list[Path],
    output_dir: Path,
    allowed_licences: tuple[str, ...] = DEFAULT_LICENCES,
) -> dict[str, Any]:
    from jev_arena.render import ranking_svg as public_ranking_svg
    from publication.render_arena_v3 import ranking_svg, task_matrix_svg

    reports, excluded = filter_licences(load_reports(paths), allowed_licences)
    if len(reports) < 2:
        raise ValueError("fewer than two reports remain after the licence filter")
    output_dir.mkdir(parents=True, exist_ok=True)
    v3 = v3_ranking(reports)
    public = public_ranking(reports)
    figures = {
        "jevarena-v3-rank.svg": ranking_svg(v3),
        "jevarena-v3-model-task.svg": task_matrix_svg(v3),
    }
    if len(public["models"]) >= 2:
        figures["jevbench-public231-rank.svg"] = public_ranking_svg(public)
    for name, svg in figures.items():
        (output_dir / name).write_text(svg, encoding="utf-8")
    receipt = {
        "schema": "dev2-card-charts/1",
        "label": "post-key same-panel",
        "created_utc": utc_now(),
        "reports": {str(path): sha_file(path) for path in paths},
        "allowed_licences": list(allowed_licences),
        "excluded_by_licence": excluded,
        "figures": {name: sha_file(output_dir / name) for name in figures},
        "v3_ranking": v3,
        "public_ranking": public,
    }
    write_json(output_dir / "charts.json", receipt, exclusive=False)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-licence",
        action="append",
        choices=("own", "permissive", "non-commercial", "research-only"),
        help="licence classes shown on the card (default: own and permissive)",
    )
    args = parser.parse_args()
    allowed = tuple(args.allow_licence) if args.allow_licence else DEFAULT_LICENCES
    receipt = render(args.report, args.output_dir, allowed)
    print(
        json.dumps(
            {"figures": receipt["figures"], "excluded": receipt["excluded_by_licence"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
