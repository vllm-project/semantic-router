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
AXES_V1 = ("typed", "transfer", "authored", "robustness")
AXES_V2 = (
    "typed",
    "transfer",
    "jevbench_public",
    "decision_bench_v4",
    "sealed_authored",
    "robustness",
)
AXIS_LABELS = {
    "typed": "Typed",
    "transfer": "Transfer",
    "authored": "Authored",
    "jevbench_public": "Public 231",
    "decision_bench_v4": "DB v4",
    "sealed_authored": "Sealed",
    "robustness": "Robustness",
}
INK = "#172542"
MUTED = "#52617c"
PAPER = "#fff9ef"
GRID = "#dce3f1"


def _svg(height: int, title: str, description: str, *, width: int = WIDTH) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        f'<rect width="100%" height="100%" fill="{PAPER}"/>',
    ]


def _text(
    x: float,
    y: float,
    value: Any,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    fill: str = INK,
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
    if schema == "jevarena-ranking/2":
        phase = report["phase"].upper()
        return (
            f"JevArena {phase}",
            "Equal-weight geometric mean of six frozen evaluation axes",
            "Same-panel rank · public subsets disclosed in methods · sealed authored quality gate required",
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
    parts.append(_text(32, 67, subtitle, size=13, fill=MUTED))
    x0, span = 365, 572
    for tick in (0, 25, 50, 75, 100):
        x = x0 + tick / 100 * span
        parts.append(
            f'<line x1="{x:.1f}" y1="106" x2="{x:.1f}" '
            f'y2="{height - 56}" stroke="{GRID}"/>'
        )
        parts.append(_text(x, 99, f"{tick}%", size=11, anchor="middle", fill=MUTED))
    for index, row in enumerate(rows):
        y = 119 + index * 49
        parts.append(
            f'<rect x="{x0}" y="{y}" width="{span}" height="25" '
            'rx="8" fill="#e8edfa"/>'
        )
        parts.append(
            f'<rect x="{x0}" y="{y}" width="{span * row["score"] / 100:.2f}" '
            f'height="25" rx="8" fill="{_color(row["group"])}"/>'
        )
        parts.append(
            _text(32, y + 18, f'{row["rank"]}. {row["label"]}', size=14, weight=600)
        )
        parts.append(
            _text(
                1028, y + 18, f'{row["score"]:.2f}', size=14, weight=700, anchor="end"
            )
        )
    parts.append(_text(32, height - 24, caveat, size=11, fill=MUTED))
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
            fill=MUTED,
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
            f'y2="{yy:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            _text(left - 12, yy + 5, f"{tick}", size=12, anchor="end", fill=MUTED)
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
            f'y2="{bottom}" stroke="{GRID}"/>'
        )
        parts.append(
            _text(xx, bottom + 21, f"{tick:g}", size=11, anchor="middle", fill=MUTED)
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
            f'<polyline points="{path}" fill="none" stroke="#f4b642" '
            'stroke-width="2" stroke-dasharray="6 5"/>'
        )
    # Draw dominated points first to keep frontier markers visible.
    for row in sorted(rows, key=lambda item: bool(item.get("pareto_frontier"))):
        xx, yy = x(row["size_b"]), y(row["score"])
        color = _color(row["group"])
        fill = color if row.get("pareto_frontier") else PAPER
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
    parts.append(_text(32, height - 24, caveat, size=11, fill=MUTED))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def matrix_svg(report: dict[str, Any]) -> str:
    schema = report.get("schema_version")
    if schema not in ("jevarena-ranking/1", "jevarena-ranking/2"):
        raise ValueError("Axis matrix is defined for JevArena only")
    axes = AXES_V2 if schema == "jevarena-ranking/2" else AXES_V1
    name, _, caveat = _context(report)
    rows = _models(report)
    height = 161 + 47 * len(rows)
    parts = _svg(height, f"{name} component matrix", caveat)
    parts.append(_text(32, 42, f"{name} component matrix", size=24, weight=700))
    parts.append(
        _text(
            32,
            67,
            f"{len(axes)} separate axes in percent · one color scale across all cells",
            size=13,
            fill=MUTED,
        )
    )
    x0, gap = (274, 7) if len(axes) == 6 else (310, 8)
    cell_width = 125 if len(axes) == 6 else 161
    for index, axis in enumerate(axes):
        parts.append(
            _text(
                x0 + index * (cell_width + gap) + cell_width / 2,
                103,
                AXIS_LABELS[axis],
                size=12 if len(axes) == 6 else 13,
                weight=600,
                anchor="middle",
            )
        )
    for index, row in enumerate(rows):
        yy = 120 + index * 47
        parts.append(
            _text(32, yy + 19, f'{row["rank"]}. {row["label"]}', size=13, weight=600)
        )
        for column, axis in enumerate(axes):
            score = row["axes"][axis]
            if type(score) not in (float, int) or not 0 <= score <= 1:
                raise ValueError("Invalid axis score")
            rr = round(255 * (1 - score) + 49 * score)
            gg = round(231 * (1 - score) + 91 * score)
            bb = round(197 * (1 - score) + 255 * score)
            fill = f"#{rr:02x}{gg:02x}{bb:02x}"
            xx = x0 + column * (cell_width + gap)
            parts.append(
                f'<rect x="{xx}" y="{yy}" width="{cell_width}" '
                f'height="29" rx="7" fill="{fill}"/>'
            )
            parts.append(
                _text(
                    xx + cell_width / 2,
                    yy + 20,
                    f"{100 * score:.1f}%",
                    size=13,
                    weight=700,
                    anchor="middle",
                    fill="#fff" if score >= 0.85 else INK,
                )
            )
    parts.append(_text(32, height - 24, caveat, size=11, fill=MUTED))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def task_matrix_svg(report: dict[str, Any]) -> str:
    """Show the three typed tasks and every transfer task on one model panel."""
    if report.get("schema_version") != "jevarena-ranking/2":
        raise ValueError("Task matrix requires the six-axis JevArena report")
    rows = _models(report)
    typed = ("choice", "noul", "score")
    first = rows[0].get("task_scores", {})
    transfer = sorted(first.get("transfer", {}))
    expected = 15 if report.get("phase") == "release" else 3
    if set(first.get("typed", {})) != set(typed) or len(transfer) != expected:
        raise ValueError("Task matrix lacks the frozen typed/transfer task panel")
    for row in rows:
        scores = row.get("task_scores", {})
        if set(scores.get("typed", {})) != set(typed) or set(
            scores.get("transfer", {})
        ) != set(transfer):
            raise ValueError("Task matrix models do not share the same task panel")
    columns = [("typed", kind, f"Typed / {kind.title()}") for kind in typed] + [
        ("transfer", task, f"Transfer / {task.replace('_', ' ')}") for task in transfer
    ]
    x0, pitch, cell_width = 275, 88, 81
    width = max(WIDTH, x0 + pitch * len(columns) + 35)
    height = 262 + 45 * len(rows)
    caveat = (
        "Typed cells: accuracy; transfer cells: macro-F1 including invalid answers. "
        "Both use the same frozen model panel."
    )
    parts = _svg(height, "JevArena model by task matrix", caveat, width=width)
    parts.append(_text(32, 42, "JevArena: model by task", size=24, weight=700))
    parts.append(
        _text(
            32,
            67,
            "Three typed tasks and individual human transfer tasks · percentages",
            size=13,
            fill=MUTED,
        )
    )
    for index, (_, _, label) in enumerate(columns):
        x = x0 + index * pitch + cell_width / 2
        parts.append(
            f'<text x="{x:.1f}" y="194" transform="rotate(-55 {x:.1f} 194)" '
            f'fill="{INK}" font-family="system-ui, sans-serif" font-size="11" '
            f'font-weight="600">{escape(label)}</text>'
        )
    for row_index, row in enumerate(rows):
        y = 207 + row_index * 45
        parts.append(
            _text(32, y + 19, f'{row["rank"]}. {row["label"]}', size=13, weight=600)
        )
        for column_index, (axis, key, _) in enumerate(columns):
            score = row["task_scores"][axis][key]
            if (
                type(score) not in (float, int)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("Invalid task score")
            red = round(255 * (1 - score) + 49 * score)
            green = round(231 * (1 - score) + 91 * score)
            blue = round(197 * (1 - score) + 255 * score)
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            x = x0 + column_index * pitch
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="29" '
                f'rx="7" fill="{fill}"/>'
            )
            parts.append(
                _text(
                    x + cell_width / 2,
                    y + 20,
                    f"{100 * score:.1f}",
                    size=12,
                    weight=700,
                    anchor="middle",
                    fill="#fff" if score >= 0.85 else INK,
                )
            )
    parts.append(_text(32, height - 24, caveat, size=11, fill=MUTED))
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
    if report.get("schema_version") in ("jevarena-ranking/1", "jevarena-ranking/2"):
        products["matrix"] = matrix_svg(report)
    if report.get("schema_version") == "jevarena-ranking/2":
        products["task-matrix"] = task_matrix_svg(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, svg in products.items():
        destination = args.output_dir / f"{args.prefix}-{suffix}.svg"
        destination.write_text(svg, encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
