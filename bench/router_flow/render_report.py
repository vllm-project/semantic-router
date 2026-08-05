#!/usr/bin/env python3
"""Render Router Flow eval tables and SVG charts from flow_eval outputs."""

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

DEFAULT_BENCHMARK_ORDER = [
    "SWE Bench Pro",
    "TerminalBench 2.1",
    "LiveCodeBench",
    "LiveCodeBench Pro",
    "Humanity's Last Exam",
    "CharXiv Reasoning",
    "GPQA-D",
    "SciCode",
    "Tau3 Banking",
    "Long Context Reasoning",
    "MRCRv2",
]

DEFAULT_ARM_LABELS = {
    "auto": "vllm-sr/auto",
    "fusion": "vllm-sr/fusion",
    "flow": "vllm-sr/flow",
}

SERIES_COLORS = {
    "vllm-sr/flow": "#8b0f14",
    "vllm-sr/fusion": "#d3272f",
    "vllm-sr/auto": "#111827",
    "Fugu Ultra": "#9b111e",
    "Fugu": "#cf1f2a",
}

REFERENCE_COLOR = "#9ca3af"


@dataclass(frozen=True)
class Series:
    name: str
    values: dict[str, float]
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        action="append",
        required=True,
        help="Path to a result directory or samples.jsonl file. May be repeated.",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=None,
        help="Optional public reference scores JSON for contextual chart columns.",
    )
    parser.add_argument(
        "--arm-label",
        action="append",
        default=[],
        help="Map an eval arm to a display label, for example flow=vllm-sr/flow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench/router_flow/results/report"),
    )
    return parser.parse_args()


def parse_arm_labels(raw_values: list[str]) -> dict[str, str]:
    labels = dict(DEFAULT_ARM_LABELS)
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError("--arm-label must be formatted as arm=label")
        arm, label = raw.split("=", 1)
        labels[arm.strip()] = label.strip()
    return labels


def samples_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        path = path / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_samples(paths: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for raw in paths:
        path = samples_path(raw)
        with path.open() as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    samples.append(json.loads(stripped))
    return samples


def eval_series(
    samples: list[dict[str, Any]], arm_labels: dict[str, str]
) -> list[Series]:
    by_arm_category: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        arm = str(sample["arm"])
        category = str(sample.get("category") or "")
        score = float((sample.get("grade") or {}).get("score") or 0.0) * 100.0
        by_arm_category.setdefault(arm, {}).setdefault(category, []).append(score)

    series: list[Series] = []
    for arm in sorted(by_arm_category):
        values = {
            category: round(statistics.fmean(scores), 1)
            for category, scores in by_arm_category[arm].items()
        }
        series.append(Series(arm_labels.get(arm, arm), values, "eval"))
    return series


def reference_series(path: Path | None) -> list[Series]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    by_series: dict[str, dict[str, float]] = {}
    for row in data.get("benchmarks", []):
        benchmark = str(row.get("benchmark") or "")
        for key, value in row.items():
            if key == "benchmark":
                continue
            if isinstance(value, int | float):
                by_series.setdefault(key, {})[benchmark] = float(value)
    return [Series(name, values, "reference") for name, values in by_series.items()]


def benchmark_order(series: list[Series]) -> list[str]:
    seen = {benchmark for item in series for benchmark in item.values}
    ordered = [benchmark for benchmark in DEFAULT_BENCHMARK_ORDER if benchmark in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def series_order(series: list[Series]) -> list[Series]:
    preferred = {
        "vllm-sr/flow": 0,
        "vllm-sr/fusion": 1,
        "vllm-sr/auto": 2,
        "Fugu Ultra": 3,
        "Fugu": 4,
    }
    return sorted(series, key=lambda item: (preferred.get(item.name, 20), item.name))


def mean_value(item: Series, benchmarks: list[str]) -> float:
    values = [
        item.values[benchmark] for benchmark in benchmarks if benchmark in item.values
    ]
    if not values:
        return 0.0
    return round(statistics.fmean(values), 1)


def write_csv(output_dir: Path, series: list[Series], benchmarks: list[str]) -> Path:
    path = output_dir / "benchmark_table.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Proxy family (n=1 prompt)", *[item.name for item in series]])
        for benchmark in benchmarks:
            writer.writerow(
                [
                    benchmark,
                    *[format_score(item.values.get(benchmark)) for item in series],
                ]
            )
        writer.writerow(
            ["Average", *[f"{mean_value(item, benchmarks):.1f}" for item in series]]
        )
    return path


def write_markdown(
    output_dir: Path, series: list[Series], benchmarks: list[str]
) -> Path:
    path = output_dir / "benchmark_table.md"
    lines = [
        "| Proxy family (n=1 prompt) | "
        + " | ".join(item.name for item in series)
        + " |",
        "| --- | " + " | ".join("---:" for _ in series) + " |",
    ]
    for benchmark in benchmarks:
        cells = [format_score(item.values.get(benchmark)) for item in series]
        lines.append("| " + benchmark + " | " + " | ".join(cells) + " |")
    averages = [f"{mean_value(item, benchmarks):.1f}" for item in series]
    lines.append("| Average | " + " | ".join(averages) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_metadata(
    output_dir: Path, series: list[Series], benchmarks: list[str]
) -> Path:
    path = output_dir / "metadata.json"
    payload = {
        "benchmarks": benchmarks,
        "series": [
            {
                "name": item.name,
                "kind": item.kind,
                "average": mean_value(item, benchmarks),
            }
            for item in series
        ],
        "score_unit": "0_to_100",
        "caveat": (
            "vLLM-SR eval rows are generated from a one-prompt-per-family proxy suite. "
            "A 100.0 row means the judge gave that single proxy prompt full credit, not official benchmark accuracy. "
            "Reference rows, when present, are public external benchmark scores and are included only for context."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_score(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}"


def color_for(item: Series) -> str:
    if item.name in SERIES_COLORS:
        return SERIES_COLORS[item.name]
    if item.kind == "eval":
        return "#374151"
    return REFERENCE_COLOR


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


def write_overall_svg(
    output_dir: Path, series: list[Series], benchmarks: list[str]
) -> Path:
    path = output_dir / "overall_bars.svg"
    width = 1120
    height = 520
    margin_left = 90
    margin_right = 40
    chart_top = 90
    chart_bottom = 420
    bar_gap = 18
    bar_width = 78
    max_score = 100
    plot_height = chart_bottom - chart_top
    ordered = series_order(series)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="1120" height="520" fill="#ffffff"/>',
        svg_text(
            40,
            42,
            "Router Flow one-prompt proxy eval: average score",
            size=24,
            weight="700",
        ),
        svg_text(
            40,
            66,
            "vLLM-SR rows use one judged prompt per family; public rows are contextual and not same-harness.",
            size=13,
            fill="#6b7280",
        ),
    ]
    for tick in range(0, 101, 20):
        y = chart_bottom - (tick / max_score) * plot_height
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
        bar_height = (value / max_score) * plot_height
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
            f'<text x="{x + bar_width / 2:.1f}" y="{chart_bottom + 24:.1f}" '
            'font-family="Inter, Arial, sans-serif" font-size="12" text-anchor="middle" '
            'fill="#374151">'
            f"{html.escape(short_label(item.name))}</text>"
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


def write_benchmark_svg(
    output_dir: Path, series: list[Series], benchmarks: list[str]
) -> Path:
    path = output_dir / "benchmark_bars.svg"
    ordered = series_order(series)
    cols = 4
    panel_w = 390
    panel_h = 250
    width = cols * panel_w
    rows = math.ceil(len(benchmarks) / cols)
    height = 84 + rows * panel_h + 38
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        svg_text(
            36, 38, "Router Flow one-prompt proxy eval by family", size=24, weight="700"
        ),
        svg_text(
            36,
            62,
            "100 means one proxy prompt got full judge credit; grey bars are public reference context.",
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
    series: list[Series],
) -> list[str]:
    chart_top = y0 + 34
    chart_bottom = y0 + height - 48
    chart_left = x0 + 36
    chart_right = x0 + width - 8
    plot_height = chart_bottom - chart_top
    bar_count = len(series)
    gap = 7
    bar_width = min(
        24, (chart_right - chart_left - gap * max(0, bar_count - 1)) / max(1, bar_count)
    )
    parts = [
        svg_text(x0, y0 + 16, benchmark, size=14, weight="700"),
        f'<line x1="{chart_left:.1f}" y1="{chart_bottom:.1f}" x2="{chart_right:.1f}" y2="{chart_bottom:.1f}" stroke="#d1d5db"/>',
    ]
    for tick in (50, 75, 100):
        y = chart_bottom - (tick / 100) * plot_height
        parts.append(
            f'<line x1="{chart_left:.1f}" y1="{y:.1f}" x2="{chart_right:.1f}" y2="{y:.1f}" stroke="#eef2f7"/>'
        )
        parts.append(
            svg_text(
                chart_left - 6, y + 4, str(tick), size=9, anchor="end", fill="#9ca3af"
            )
        )
    total_width = bar_count * bar_width + gap * max(0, bar_count - 1)
    start_x = chart_left + ((chart_right - chart_left) - total_width) / 2
    for i, item in enumerate(series):
        value = item.values.get(benchmark)
        if value is None:
            continue
        bar_height = (value / 100) * plot_height
        x = start_x + i * (bar_width + gap)
        y = chart_bottom - bar_height
        color = color_for(item)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{color}"/>'
        )
        parts.append(
            svg_text(
                x + bar_width / 2,
                y - 4,
                f"{value:.1f}",
                size=9,
                weight="700",
                anchor="middle",
                fill=color,
            )
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{chart_bottom + 14:.1f}" '
            'font-family="Inter, Arial, sans-serif" font-size="8" text-anchor="middle" '
            'fill="#4b5563">'
            f"{html.escape(tiny_label(item.name))}</text>"
        )
    return parts


def short_label(name: str) -> str:
    return {
        "vllm-sr/flow": "flow",
        "vllm-sr/fusion": "fusion",
        "vllm-sr/auto": "auto",
        "Fugu Ultra": "Fugu Ultra",
        "Gemini 3.1 Pro": "Gemini 3.1",
    }.get(name, name)


def tiny_label(name: str) -> str:
    return {
        "vllm-sr/flow": "flow",
        "vllm-sr/fusion": "fusion",
        "vllm-sr/auto": "auto",
        "Fugu Ultra": "Ultra",
        "Fugu": "Fugu",
        "Opus 4.8": "Opus",
        "Gemini 3.1 Pro": "Gemini",
        "GPT 5.5": "GPT",
    }.get(name, name[:8])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_arm_labels(args.arm_label)
    samples = read_samples(args.results)
    all_series = series_order(
        eval_series(samples, labels) + reference_series(args.reference_json)
    )
    benchmarks = benchmark_order(all_series)
    write_csv(output_dir, all_series, benchmarks)
    write_markdown(output_dir, all_series, benchmarks)
    write_metadata(output_dir, all_series, benchmarks)
    write_overall_svg(output_dir, all_series, benchmarks)
    write_benchmark_svg(output_dir, all_series, benchmarks)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "benchmarks": len(benchmarks),
                "series": [s.name for s in all_series],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
