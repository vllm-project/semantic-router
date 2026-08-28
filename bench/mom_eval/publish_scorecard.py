"""Generate scorecard artifacts from MoM evaluation result bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from bench.mom_eval.common import REPO_ROOT, write_json


def build_scorecard_json(result: dict[str, Any]) -> dict[str, Any]:
    identity = result.get("identity") or {}
    metrics = result.get("metrics") or {}
    baselines = result.get("baselines") or []
    publication = result.get("publication") or {}
    rows = []
    for metric_id, metric in sorted(metrics.items()):
        if metric.get("value") is None:
            continue
        rows.append(
            {
                "metric": metric_id,
                "value": metric.get("value"),
                "unit": metric.get("unit"),
                "classification": metric.get("classification"),
                "layer": metric.get("layer"),
            }
        )
    baseline_rows = []
    for baseline in baselines:
        baseline_rows.append(
            {
                "model": baseline.get("model"),
                "role": baseline.get("role"),
                "metrics": baseline.get("metrics"),
            }
        )
    return {
        "schema_version": "vllm-sr/mom-scorecard/v1",
        "entrypoint": identity.get("entrypoint"),
        "recipe_id": identity.get("recipe_id"),
        "recipe_version": identity.get("recipe_version"),
        "objective": identity.get("objective"),
        "run_mode": identity.get("run_mode"),
        "generated_at": identity.get("generated_at"),
        "contract_version": (result.get("contract") or {}).get("core_suite_version"),
        "publication": publication,
        "metrics": rows,
        "baselines": baseline_rows,
    }


def render_scorecard_markdown(scorecard: dict[str, Any], result_path: str) -> str:
    lines = [
        "### Launch scorecard",
        "",
        f"- Evaluation contract: `vllm-sr/mom-evaluation/v1`",
        f"- Core suite version: `{scorecard.get('contract_version')}`",
        f"- Entrypoint: `{scorecard.get('entrypoint')}`",
        f"- Recipe version: `{scorecard.get('recipe_version')}`",
        f"- Run mode: `{scorecard.get('run_mode')}`",
        f"- Generated: `{scorecard.get('generated_at')}`",
        "",
        "| Metric | Value | Layer | Classification |",
        "| --- | ---: | --- | --- |",
    ]
    for row in scorecard.get("metrics") or []:
        value = row.get("value")
        display = "n/a" if value is None else f"{value}"
        lines.append(
            f"| `{row.get('metric')}` | {display} | {row.get('layer') or '-'} | {row.get('classification')} |"
        )
    lines.extend(["", "### Baseline comparison", ""])
    for baseline in scorecard.get("baselines") or []:
        lines.append(
            f"- `{baseline.get('model')}` ({baseline.get('role')}): "
            f"{json.dumps(baseline.get('metrics') or {}, sort_keys=True)}"
        )
    lines.extend(
        [
            "",
            "### Known limitations",
            "",
            "- Smoke or diagnostic runs are not publishable launch scores.",
            "- Full formal scores require all seven provider backends to be reachable.",
            "",
            f"Full result bundle: [`mom_eval_result.json`]({result_path})",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def publish_scorecard(result_path: Path, output_dir: Path) -> dict[str, str]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    scorecard = build_scorecard_json(result)
    output_dir.mkdir(parents=True, exist_ok=True)

    scorecard_json = output_dir / "scorecard.json"
    scorecard_md = output_dir / "scorecard.md"
    result_copy = output_dir / "mom_eval_result.json"

    write_json(scorecard_json, scorecard)
    rel_result = result_copy.relative_to(REPO_ROOT).as_posix()
    scorecard_md.write_text(render_scorecard_markdown(scorecard, rel_result), encoding="utf-8")
    write_json(result_copy, result)

    provenance = {
        "entrypoint": scorecard.get("entrypoint"),
        "recipe_version": scorecard.get("recipe_version"),
        "run_mode": scorecard.get("run_mode"),
        "generated_at": scorecard.get("generated_at"),
        "source_result": str(result_path.relative_to(REPO_ROOT)) if result_path.is_relative_to(REPO_ROOT) else str(result_path),
    }
    provenance_path = output_dir / "provenance.yaml"
    provenance_path.write_text(yaml.safe_dump(provenance, sort_keys=False), encoding="utf-8")

    return {
        "scorecard_json": str(scorecard_json),
        "scorecard_md": str(scorecard_md),
        "result_copy": str(result_copy),
        "provenance": str(provenance_path),
    }


def update_scorecard_index(
    recipe_id: str,
    entrypoint: str,
    recipe_version: str,
    result_rel_path: str,
    scorecard_rel_path: str,
) -> None:
    index_path = REPO_ROOT / "config/evaluation/scorecards/index.yaml"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.is_file():
        index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    else:
        index = {"schema_version": "vllm-sr/scorecard-index/v1", "scorecards": {}}

    scorecards = index.setdefault("scorecards", {})
    recipe_entry = scorecards.setdefault(recipe_id, {})
    endpoint_entry = recipe_entry.setdefault(entrypoint, {})
    endpoint_entry[recipe_version] = {
        "result_path": result_rel_path,
        "scorecard_path": scorecard_rel_path,
    }
    index_path.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
