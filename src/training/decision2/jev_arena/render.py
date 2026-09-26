"""Render reproducible JevBench-public and JevArena SVG model-card figures."""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path
from typing import Any

from publication.render import COLORS

WIDTH = 1080
AXES = ("typed", "transfer", "authored", "robustness")


def _svg(height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]


def _text(
    x: float,
    y: float,
    value: Any,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#202b36",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" '
        f'font-family="system-ui, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def _color(group: str) -> str:
    return COLORS.get(group, COLORS["other"])


def _context(report: dict[str, Any]) -> tuple[str, str, str]:
    schema = report.get("schema_version")
    if schema == "jevarena-jevbench-public-rank/1":
        return (
            "JevBench public-only",
            "Raw accuracy on 231 public items · same pinned panel",
            "Independent rerun · excludes private and sealed questions · not an official JevBench rank",
        )
    if schema == "jevarena-ranking/1":
        phase = report["phase"].upper()
        return (
            f"JevArena {phase}",
            "Equal-weight geometric mean of typed, transfer, authored and robustness",
            "Independent multi-panel rank · public authored items · rank among displayed models",
        )
    raise ValueError("Unsupported ranking report schema")


def _models(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Ranking report has no models")
    if any(
        type(row.get("score")) not in (float, int)
        or not math.isfinite(row["score"])
        or not 0 <= row["score"] <= 100
        for row in rows
    ):
        raise ValueError("Ranking report contains invalid score")
    return sorted(rows, key=lambda row: (row["rank"], row["key"]))


def ranking_svg(report: dict[str, Any]) -> str:
    name, subtitle, caveat = _context(report)
    rows = _models(report)
    height = 171 + 49 * len(rows)
    parts = _svg(height, f"{name} ranking", caveat)
    parts.append(_text(32, 42, f"{name} ranking", size=24, weight=700))
    parts.append(_text(32, 67, subtitle, size=13, fill="#51606b"))
    x0, span = 365, 572
    for tick in (0, 25, 50, 75, 100):
        x = x0 + tick / 100 * span
        parts.append(
            f'<line x1="{x:.1f}" y1="106" x2="{x:.1f}" '
            f'y2="{height - 56}" stroke="#e3e8ed"/>'
        )
        parts.append(_text(x, 99, f"{tick}%", size=11, anchor="middle", fill="#61707e"))
    for index, row in enumerate(rows):
        y = 119 + index * 49
        parts.append(
            f'<rect x="{x0}" y="{y}" width="{span}" height="25" '
            'rx="4" fill="#f1f4f7"/>'
        )
        parts.append(
            f'<rect x="{x0}" y="{y}" width="{span * row["score"] / 100:.2f}" '
            f'height="25" rx="4" fill="{_color(row["group"])}"/>'
        )
        parts.append(
            _text(32, y + 18, f'{row["rank"]}. {row["label"]}', size=14, weight=600)
        )
        parts.append(
            _text(
                1028, y + 18, f'{row["score"]:.2f}', size=14, weight=700, anchor="end"
            )
        )
    parts.append(_text(32, height - 24, caveat, size=11, fill="#61707e"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def pareto_svg(report: dict[str, Any]) -> str:
    name, _, caveat = _context(report)
    rows = [row for row in _models(report) if row.get("size_b") is not None]
    if len(rows) < 2:
        raise ValueError(
            "Pareto plot needs at least two models with known parameter counts"
        )
    if any(
        type(row["size_b"]) not in (float, int)
        or row["size_b"] <= 0
        or not math.isfinite(row["size_b"])
        for row in rows
    ):
        raise ValueError("Pareto plot has invalid parameter count")
    height = 670
    parts = _svg(height, f"{name} size and score", caveat)
    parts.append(_text(32, 42, f"{name}: size and score", size=24, weight=700))
    parts.append(
        _text(
            32,
            67,
            "Actual parameter count (billions, log scale) vs score · filled = Pareto frontier",
            size=13,
            fill="#51606b",
        )
    )
    left, right, top, bottom = 105, 960, 113, 562
    minimum, maximum = min(row["size_b"] for row in rows), max(
        row["size_b"] for row in rows
    )
    log_min, log_max = math.log10(minimum) - 0.14, math.log10(maximum) + 0.14

    def x(value: float) -> float:
        return left + (math.log10(value) - log_min) / (log_max - log_min) * (
            right - left
        )

    def y(value: float) -> float:
        return bottom - value / 100 * (bottom - top)

    for tick in (0, 20, 40, 60, 80, 100):
        yy = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" '
            f'y2="{yy:.1f}" stroke="#e3e8ed"/>'
        )
        parts.append(
            _text(left - 12, yy + 5, f"{tick}", size=12, anchor="end", fill="#61707e")
        )
    powers = range(math.floor(log_min), math.ceil(log_max) + 1)
    ticks = sorted(
        {
            round(10**power * multiple, 6)
            for power in powers
            for multiple in (1, 2, 5)
            if log_min <= math.log10(10**power * multiple) <= log_max
        }
    )
    for tick in ticks:
        xx = x(tick)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" '
            f'y2="{bottom}" stroke="#edf0f3"/>'
        )
        parts.append(
            _text(
                xx, bottom + 21, f"{tick:g}", size=11, anchor="middle", fill="#61707e"
            )
        )
    frontier = sorted(
        (row for row in rows if row.get("pareto_frontier")),
        key=lambda row: (row["size_b"], row["score"]),
    )
    if len(frontier) > 1:
        path = " ".join(
            f'{x(row["size_b"]):.1f},{y(row["score"]):.1f}' for row in frontier
        )
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="#718c87" '
            'stroke-width="2" stroke-dasharray="6 5"/>'
        )
    # Draw dominated points first to keep frontier markers visible.
    for row in sorted(rows, key=lambda item: bool(item.get("pareto_frontier"))):
        xx, yy = x(row["size_b"]), y(row["score"])
        color = _color(row["group"])
        fill = color if row.get("pareto_frontier") else "#fff"
        parts.append(
            f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="8" '
            f'fill="{fill}" stroke="{color}" stroke-width="2.5"/>'
        )
        # Offset labels for coincident size buckets; all values remain visible in the rank chart.
        same_size = [peer for peer in rows if peer["size_b"] == row["size_b"]]
        index = sorted(same_size, key=lambda peer: (peer["score"], peer["key"])).index(
            row
        )
        label_y = yy - 13 if index % 2 == 0 else yy + 26
        parts.append(_text(xx + 11, label_y, row["label"], size=11, fill=color))
    parts.append(
        _text(
            (left + right) / 2,
            bottom + 53,
            "Model size (B parameters)",
            size=13,
            anchor="middle",
        )
    )
    parts.append(_text(32, height - 24, caveat, size=11, fill="#61707e"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def matrix_svg(report: dict[str, Any]) -> str:
    if report.get("schema_version") != "jevarena-ranking/1":
        raise ValueError("Axis matrix is defined for JevArena only")
    name, _, caveat = _context(report)
    rows = _models(report)
    height = 161 + 47 * len(rows)
    parts = _svg(height, f"{name} component matrix", caveat)
    parts.append(_text(32, 42, f"{name} component matrix", size=24, weight=700))
    parts.append(
        _text(
            32,
            67,
            "Four separate axes in percent · one color scale across all cells",
            size=13,
            fill="#51606b",
        )
    )
    x0, cell_width, gap = 310, 161, 8
    for index, axis in enumerate(AXES):
        parts.append(
            _text(
                x0 + index * (cell_width + gap) + cell_width / 2,
                103,
                axis.title(),
                size=13,
                weight=600,
                anchor="middle",
            )
        )
    for index, row in enumerate(rows):
        yy = 120 + index * 47
        parts.append(
            _text(32, yy + 19, f'{row["rank"]}. {row["label"]}', size=13, weight=600)
        )
        for column, axis in enumerate(AXES):
            score = row["axes"][axis]
            if type(score) not in (float, int) or not 0 <= score <= 1:
                raise ValueError("Invalid axis score")
            rr = round(239 * (1 - score) + 23 * score)
            gg = round(245 * (1 - score) + 107 * score)
            bb = round(243 * (1 - score) + 91 * score)
            fill = f"#{rr:02x}{gg:02x}{bb:02x}"
            xx = x0 + column * (cell_width + gap)
            parts.append(
                f'<rect x="{xx}" y="{yy}" width="{cell_width}" '
                f'height="29" rx="4" fill="{fill}"/>'
            )
            parts.append(
                _text(
                    xx + cell_width / 2,
                    yy + 20,
                    f"{100 * score:.1f}%",
                    size=13,
                    weight=700,
                    anchor="middle",
                    fill="#fff" if score >= 0.67 else "#17312c",
                )
            )
    parts.append(_text(32, height - 24, caveat, size=11, fill="#61707e"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    products = {"rank": ranking_svg(report), "pareto": pareto_svg(report)}
    if report.get("schema_version") == "jevarena-ranking/1":
        products["matrix"] = matrix_svg(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, svg in products.items():
        destination = args.output_dir / f"{args.prefix}-{suffix}.svg"
        destination.write_text(svg, encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
