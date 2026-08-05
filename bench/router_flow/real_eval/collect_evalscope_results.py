#!/usr/bin/env python3
"""Collect EvalScope reports into Fugu-aligned VSR tables.

The runner in this directory executes official EvalScope adapters. This script
is the second half of that loop: read EvalScope report JSON files, extract the
configured metric for each benchmark/model arm, join public reference scores,
and emit stable artifacts for the blog/table/chart path.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by users without PyYAML.
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


DEFAULT_SUITE = Path(__file__).with_name("evalscope_suite.yaml")
DEFAULT_OUTPUT_ROOT = Path("bench/router_flow/results/evalscope")
DEFAULT_REFERENCE_JSON = Path(__file__).parents[1] / "public_reference_scores.json"
DEFAULT_REPORT_DIR = Path("bench/router_flow/results/evalscope-report")
DEFAULT_MODEL_LABELS = {
    "auto": "VSR 1.0 Pro",
    "glm52_native": "GLM 5.2",
    "kimi_k27_code_native": "Kimi K2.7 Code",
    "remom": "vllm-sr/remom",
    "fusion": "vllm-sr/fusion",
    "flow": "vllm-sr/flow",
    "flow_dynamic": "vllm-sr/flow-dynamic",
}
REFERENCE_COLUMNS = [
    "GLM 5.2",
    "Fugu",
    "Fugu Ultra",
    "Opus 4.8",
    "Gemini 3.1 Pro",
    "GPT 5.5",
]
PRIMARY_REFERENCE_COLUMNS = ["GLM 5.2", "Fugu", "Fugu Ultra"]
SERIES_COLORS = {
    "vllm-sr/flow": "#8b0f14",
    "vllm-sr/flow-dynamic": "#0f766e",
    "vllm-sr/fusion": "#d3272f",
    "vllm-sr/remom": "#2563eb",
    "VSR 1.0": "#0b66c3",
    "VSR 1.0 Pro": "#0b66c3",
    "GLM 5.2": "#64748b",
    "Kimi K2.7 Code": "#64748b",
    "Fugu Ultra": "#9b111e",
    "Fugu": "#cf1f2a",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    public_name: str
    dataset: str
    metric: str
    default_run: bool
    tier: str


@dataclass(frozen=True)
class ScoreCell:
    score: float | None
    raw_score: float | None
    num: int | None
    metric: str
    report_path: str
    status: str


@dataclass(frozen=True)
class ScoreSeries:
    name: str
    values: dict[str, float]
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model key from suite.models. Repeatable. Defaults to every suite model.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Benchmark id from suite.benchmarks. Defaults to default_run benchmarks.",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Include heavy default_run=false benchmarks when no --benchmark is supplied.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero if any selected benchmark/model score is missing.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_reference_scores(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    references: dict[str, dict[str, float]] = {}
    for row in data.get("benchmarks", []):
        benchmark = str(row.get("benchmark") or "")
        if not benchmark:
            continue
        values: dict[str, float] = {}
        for key, value in row.items():
            if key == "benchmark":
                continue
            if isinstance(value, int | float):
                values[str(key)] = float(value)
        references[benchmark] = values
    return references


def suite_models(suite: dict[str, Any], requested: list[str]) -> dict[str, str]:
    models = suite.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("suite.models must be a mapping")
    keys = requested or list(models)
    missing = [key for key in keys if key not in models]
    if missing:
        raise ValueError(f"unknown model key(s): {', '.join(missing)}")
    return {key: str(models[key]) for key in keys}


def suite_model_labels(suite: dict[str, Any], models: dict[str, str]) -> dict[str, str]:
    labels = suite.get("model_labels") or {}
    if labels and not isinstance(labels, dict):
        raise ValueError("suite.model_labels must be a mapping")
    return {
        key: str(labels.get(key) or DEFAULT_MODEL_LABELS.get(key) or model_name)
        for key, model_name in models.items()
    }


def suite_benchmarks(
    suite: dict[str, Any],
    requested: list[str],
    include_heavy: bool,
) -> list[BenchmarkSpec]:
    raw_benchmarks = suite.get("benchmarks") or []
    if not isinstance(raw_benchmarks, list):
        raise ValueError("suite.benchmarks must be a list")
    selected = select_raw_benchmarks(raw_benchmarks, requested, include_heavy)
    return [benchmark_spec(raw) for raw in selected]


def select_raw_benchmarks(
    benchmarks: list[dict[str, Any]],
    requested: list[str],
    include_heavy: bool,
) -> list[dict[str, Any]]:
    if requested:
        requested_set = set(requested)
        selected = [bench for bench in benchmarks if bench.get("id") in requested_set]
        missing = requested_set - {str(bench.get("id")) for bench in selected}
        if missing:
            raise ValueError(f"unknown benchmark id(s): {', '.join(sorted(missing))}")
        return selected

    selected = []
    for bench in benchmarks:
        if not bench.get("default_run"):
            continue
        if str(bench.get("tier") or "") == "heavy" and not include_heavy:
            continue
        selected.append(bench)
    return selected


def benchmark_spec(raw: dict[str, Any]) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id=str(raw["id"]),
        public_name=str(raw.get("public_name") or raw["id"]),
        dataset=str(raw["dataset"]),
        metric=str(raw["metric"]),
        default_run=bool(raw.get("default_run")),
        tier=str(raw.get("tier") or "core"),
    )


def collect_scores(
    *,
    suite: dict[str, Any],
    output_root: Path,
    references: dict[str, dict[str, float]],
    requested_models: list[str],
    requested_benchmarks: list[str],
    include_heavy: bool,
) -> dict[str, Any]:
    models = suite_models(suite, requested_models)
    model_labels = suite_model_labels(suite, models)
    benchmarks = suite_benchmarks(suite, requested_benchmarks, include_heavy)
    rows = []
    missing = []
    for bench in benchmarks:
        model_scores: dict[str, dict[str, Any]] = {}
        for model_key in models:
            cell = read_evalscope_score(output_root, bench, model_key)
            model_scores[model_key] = score_cell_payload(cell)
            if cell.score is None:
                missing.append({"benchmark": bench.benchmark_id, "model": model_key})
        reference = references.get(bench.public_name, {})
        rows.append(
            {
                "id": bench.benchmark_id,
                "benchmark": bench.public_name,
                "dataset": bench.dataset,
                "metric": bench.metric,
                "tier": bench.tier,
                "models": model_scores,
                "reference": reference,
                "comparison": comparison_payload(model_scores, reference),
            }
        )
    return {
        "score_unit": "0_to_100",
        "models": models,
        "model_labels": model_labels,
        "benchmarks": rows,
        "averages": average_payload(rows, models),
        "missing": missing,
        "source": "EvalScope reports joined with public reference scores",
    }


def score_cell_payload(cell: ScoreCell) -> dict[str, Any]:
    return {
        "score": cell.score,
        "raw_score": cell.raw_score,
        "num": cell.num,
        "metric": cell.metric,
        "report_path": cell.report_path,
        "status": cell.status,
    }


def read_evalscope_score(
    output_root: Path,
    bench: BenchmarkSpec,
    model_key: str,
) -> ScoreCell:
    work_dir = output_root / bench.benchmark_id / model_key
    report_paths = sorted((work_dir / "reports").glob("**/*.json"))
    if not report_paths:
        return ScoreCell(None, None, None, bench.metric, "", "missing_report")

    errors: list[str] = []
    for path in report_paths:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: {exc}")
            continue
        raw = first_metric_value(data, bench.metric)
        if raw is None:
            continue
        score = normalize_score(raw)
        return ScoreCell(
            score=round(score, 4),
            raw_score=raw,
            num=first_count_value(data),
            metric=bench.metric,
            report_path=str(path),
            status="ok",
        )
    status = "metric_not_found"
    if errors:
        status += ": " + "; ".join(errors[:2])
    return ScoreCell(None, None, None, bench.metric, "", status)


def first_metric_value(data: Any, metric: str) -> float | None:
    for candidate in metric_candidates(metric):
        for value in metric_values(data, candidate):
            return value
    return None


def first_count_value(data: Any) -> int | None:
    for candidate in ("num", "total", "num_samples", "sample_count"):
        value = first_integer_value(data, candidate)
        if value is not None:
            return value
    return None


def first_integer_value(data: Any, key: str) -> int | None:
    if isinstance(data, dict):
        value = as_number(data.get(key))
        if value is not None and value >= 0 and float(value).is_integer():
            return int(value)
        for child in data.values():
            found = first_integer_value(child, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = first_integer_value(item, key)
            if found is not None:
                return found
    return None


def metric_candidates(metric: str) -> list[str]:
    candidates = [metric]
    aliases = {
        "acc": ["mean_acc"],
        "accuracy": ["acc", "mean_acc"],
        "mrcr_score": ["overall_mrcr_score"],
        "pass_rate": ["main_problem_pass_rate", "mean_acc"],
    }
    for alias in aliases.get(metric, []):
        if alias not in candidates:
            candidates.append(alias)
    return candidates


def metric_values(data: Any, metric: str) -> list[float]:
    found: list[float] = []
    if isinstance(data, dict):
        if metric in data:
            number = as_number(data[metric])
            if number is not None:
                found.append(number)
        metric_name = str(data.get("metric") or data.get("name") or "")
        for key in ("score", "value"):
            if metric_name == metric:
                number = as_number(data.get(key))
                if number is not None:
                    found.append(number)
        for value in data.values():
            found.extend(metric_values(value, metric))
    elif isinstance(data, list):
        for item in data:
            found.extend(metric_values(item, metric))
    return found


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def normalize_score(raw: float) -> float:
    if 0.0 <= raw <= 1.0:
        return raw * 100.0
    return raw


def comparison_payload(
    model_scores: dict[str, dict[str, Any]],
    reference: dict[str, float],
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for model_key, payload in model_scores.items():
        score = payload.get("score")
        item: dict[str, Any] = {}
        for ref_name in PRIMARY_REFERENCE_COLUMNS:
            ref_score = reference.get(ref_name)
            if isinstance(score, int | float) and isinstance(ref_score, int | float):
                item[f"delta_vs_{slug(ref_name)}"] = round(float(score) - ref_score, 4)
                item[f"beats_{slug(ref_name)}"] = float(score) > ref_score
        comparison[model_key] = item
    return comparison


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def average_payload(
    rows: list[dict[str, Any]], models: dict[str, str]
) -> dict[str, Any]:
    payload: dict[str, Any] = {"models": {}, "reference": {}}
    for model_key in models:
        values = [
            float(row["models"][model_key]["score"])
            for row in rows
            if isinstance(row["models"][model_key].get("score"), int | float)
        ]
        payload["models"][model_key] = (
            round(statistics.fmean(values), 4) if values else None
        )
    for ref_name in REFERENCE_COLUMNS:
        values = [
            float(row["reference"][ref_name])
            for row in rows
            if isinstance(row["reference"].get(ref_name), int | float)
        ]
        payload["reference"][ref_name] = (
            round(statistics.fmean(values), 4) if values else None
        )
    return payload


def write_outputs(output_dir: Path, collected: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evalscope_scores.json").write_text(
        json.dumps(collected, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output_dir, collected)
    write_markdown(output_dir, collected)
    write_overall_svg(output_dir, collected)
    write_benchmark_svg(output_dir, collected)


def display_columns(collected: dict[str, Any]) -> list[str]:
    model_keys = list((collected.get("models") or {}).keys())
    return [*model_keys, *REFERENCE_COLUMNS]


def display_label(collected: dict[str, Any], column: str) -> str:
    models = collected.get("models") or {}
    if column in models:
        labels = collected.get("model_labels") or {}
        return str(
            labels.get(column) or DEFAULT_MODEL_LABELS.get(column) or models[column]
        )
    return column


def row_score(row: dict[str, Any], column: str) -> float | None:
    if column in row.get("models", {}):
        return row["models"][column].get("score")
    return row.get("reference", {}).get(column)


def write_csv(output_dir: Path, collected: dict[str, Any]) -> Path:
    path = output_dir / "benchmark_table.csv"
    columns = display_columns(collected)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Benchmark", *[display_label(collected, col) for col in columns]]
        )
        for row in collected["benchmarks"]:
            writer.writerow(
                [
                    row["benchmark"],
                    *[format_score(row_score(row, col)) for col in columns],
                ]
            )
        averages = collected["averages"]
        writer.writerow(
            [
                "Average",
                *[
                    format_score(
                        averages["models"].get(col)
                        if col in averages["models"]
                        else averages["reference"].get(col)
                    )
                    for col in columns
                ],
            ]
        )
    return path


def write_markdown(output_dir: Path, collected: dict[str, Any]) -> Path:
    path = output_dir / "benchmark_table.md"
    columns = display_columns(collected)
    labels = [display_label(collected, col) for col in columns]
    lines = [
        "| Benchmark | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for row in collected["benchmarks"]:
        cells = [format_score(row_score(row, col)) for col in columns]
        lines.append("| " + row["benchmark"] + " | " + " | ".join(cells) + " |")
    averages = collected["averages"]
    avg_cells = [
        format_score(
            averages["models"].get(col)
            if col in averages["models"]
            else averages["reference"].get(col)
        )
        for col in columns
    ]
    lines.append("| Average | " + " | ".join(avg_cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path


def format_score(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}"


def chart_series(collected: dict[str, Any]) -> list[ScoreSeries]:
    columns = display_columns(collected)
    series = []
    for col in columns:
        values = {
            row["benchmark"]: float(score)
            for row in collected["benchmarks"]
            if isinstance((score := row_score(row, col)), int | float)
        }
        kind = "eval" if col in collected.get("models", {}) else "reference"
        series.append(ScoreSeries(display_label(collected, col), values, kind))
    return series


def mean_value(item: ScoreSeries, benchmarks: list[str]) -> float:
    values = [
        item.values[benchmark] for benchmark in benchmarks if benchmark in item.values
    ]
    if not values:
        return 0.0
    return round(statistics.fmean(values), 1)


def series_order(series: list[ScoreSeries]) -> list[ScoreSeries]:
    preferred = {
        "VSR 1.0": 0,
        "VSR 1.0 Pro": 0,
        "vllm-sr/flow": 0,
        "vllm-sr/flow-dynamic": 1,
        "vllm-sr/fusion": 2,
        "vllm-sr/remom": 3,
        "vllm-sr/auto": 4,
        "GLM 5.2": 4,
        "Kimi K2.7 Code": 4,
        "Fugu Ultra": 5,
        "Fugu": 6,
    }
    return sorted(series, key=lambda item: (preferred.get(item.name, 20), item.name))


def color_for(item: ScoreSeries) -> str:
    if item.name in SERIES_COLORS:
        return SERIES_COLORS[item.name]
    if item.kind == "eval":
        return "#374151"
    return "#9ca3af"


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    weight: str = "400",
    anchor: str = "start",
    fill: str = "#111827",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(text)}</text>"
    )


def write_overall_svg(output_dir: Path, collected: dict[str, Any]) -> Path:
    path = output_dir / "overall_bars.svg"
    benchmarks = [row["benchmark"] for row in collected["benchmarks"]]
    ordered = series_order(chart_series(collected))
    width = 1120
    height = 520
    margin_left = 90
    margin_right = 40
    chart_top = 92
    chart_bottom = 420
    bar_gap = 18
    bar_width = 78
    plot_height = chart_bottom - chart_top
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="1120" height="520" fill="#ffffff"/>',
        svg_text(
            40, 42, "Router Flow EvalScope eval: average score", size=24, weight="700"
        ),
        svg_text(
            40,
            66,
            "vLLM-SR rows are EvalScope reports; public rows are reference scores from the Fugu table.",
            size=13,
            fill="#6b7280",
        ),
    ]
    for tick in range(0, 101, 20):
        y = chart_bottom - (tick / 100.0) * plot_height
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            svg_text(68, y + 4, str(tick), size=11, anchor="end", fill="#6b7280")
        )
    parts.append(
        f'<line x1="{margin_left}" y1="{chart_bottom}" x2="{width - margin_right}" y2="{chart_bottom}" stroke="#9ca3af"/>'
    )

    total_width = len(ordered) * bar_width + max(0, len(ordered) - 1) * bar_gap
    start_x = margin_left + ((width - margin_right - margin_left) - total_width) / 2
    for i, item in enumerate(ordered):
        value = mean_value(item, benchmarks)
        bar_height = (value / 100.0) * plot_height
        x = start_x + i * (bar_width + bar_gap)
        y = chart_bottom - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="4" fill="{color_for(item)}"/>'
        )
        parts.append(
            svg_text(
                x + bar_width / 2,
                y - 8,
                f"{value:.1f}",
                size=13,
                weight="700",
                anchor="middle",
                fill=color_for(item),
            )
        )
        parts.append(
            svg_text(
                x + bar_width / 2,
                chart_bottom + 24,
                short_label(item.name),
                size=12,
                anchor="middle",
                fill="#374151",
            )
        )
    parts.append(
        svg_text(
            40,
            490,
            "Scores are normalized to 0-100. Higher is better.",
            size=12,
            fill="#6b7280",
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")
    return path


def write_benchmark_svg(output_dir: Path, collected: dict[str, Any]) -> Path:
    path = output_dir / "benchmark_bars.svg"
    benchmarks = [row["benchmark"] for row in collected["benchmarks"]]
    ordered = series_order(chart_series(collected))
    cols = 3
    panel_w = 430
    panel_h = 250
    width = cols * panel_w
    rows = math.ceil(len(benchmarks) / cols)
    height = 84 + rows * panel_h + 38
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        svg_text(
            36, 38, "Router Flow EvalScope eval by benchmark", size=24, weight="700"
        ),
        svg_text(
            36,
            62,
            "Grey bars are public reference context; missing vLLM-SR bars mean no EvalScope report was found.",
            size=13,
            fill="#6b7280",
        ),
    ]
    for idx, benchmark in enumerate(benchmarks):
        col = idx % cols
        row = idx // cols
        x0 = col * panel_w + 32
        y0 = 92 + row * panel_h
        parts.extend(
            render_panel(x0, y0, panel_w - 48, panel_h - 34, benchmark, ordered)
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")
    return path


def render_panel(
    x0: float,
    y0: float,
    width: float,
    height: float,
    benchmark: str,
    series: list[ScoreSeries],
) -> list[str]:
    chart_top = y0 + 34
    chart_bottom = y0 + height - 48
    plot_height = chart_bottom - chart_top
    visible = [item for item in series if benchmark in item.values]
    bar_width = min(34, max(18, (width - 26) / max(1, len(visible)) - 8))
    gap = 8
    total_width = len(visible) * bar_width + max(0, len(visible) - 1) * gap
    start_x = x0 + (width - total_width) / 2
    parts = [svg_text(x0, y0 + 16, benchmark, size=14, weight="700")]
    for tick in (50, 75, 100):
        y = chart_bottom - (tick / 100.0) * plot_height
        parts.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#eef2f7"/>'
        )
        parts.append(
            svg_text(x0 - 6, y + 4, str(tick), size=9, anchor="end", fill="#6b7280")
        )
    for i, item in enumerate(visible):
        value = item.values[benchmark]
        bar_height = (value / 100.0) * plot_height
        x = start_x + i * (bar_width + gap)
        y = chart_bottom - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{color_for(item)}"/>'
        )
        parts.append(
            svg_text(
                x + bar_width / 2,
                y - 5,
                f"{value:.1f}",
                size=9,
                weight="700",
                anchor="middle",
                fill=color_for(item),
            )
        )
        parts.append(
            svg_text(
                x + bar_width / 2,
                chart_bottom + 14,
                tiny_label(item.name),
                size=8,
                anchor="middle",
                fill="#374151",
            )
        )
    return parts


def short_label(label: str) -> str:
    return label.replace("vllm-sr/", "")


def tiny_label(label: str) -> str:
    replacements = {
        "vllm-sr/flow": "flow",
        "vllm-sr/flow-dynamic": "flow-dyn",
        "vllm-sr/fusion": "fusion",
        "vllm-sr/remom": "remom",
        "vllm-sr/auto": "auto",
        "VSR 1.0": "VSR",
        "VSR 1.0 Pro": "Pro",
        "GLM 5.2": "GLM",
        "Kimi K2.7 Code": "Kimi",
        "Fugu Ultra": "Ultra",
        "Gemini 3.1 Pro": "Gemini",
    }
    return replacements.get(label, label.split(maxsplit=1)[0])


def main() -> int:
    args = parse_args()
    suite = load_yaml(args.suite)
    references = load_reference_scores(args.reference_json)
    collected = collect_scores(
        suite=suite,
        output_root=args.output_root,
        references=references,
        requested_models=args.model,
        requested_benchmarks=args.benchmark,
        include_heavy=args.include_heavy,
    )
    write_outputs(args.output_dir, collected)
    missing = collected["missing"]
    print(json.dumps(collected["averages"], indent=2, sort_keys=True))
    if missing:
        print(
            f"missing {len(missing)} score(s); see {args.output_dir / 'evalscope_scores.json'}"
        )
    if args.require_complete and missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
