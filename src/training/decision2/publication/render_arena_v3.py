"""Two-axis JevArena v3 figures in the established Decision card chart style.

Kept separate from the frozen v2 renderer: public JevBench remains a different
ranking, and neither authored questions nor public subsets enter the v3 axes.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any

from jev_arena.render import GRID, INK, MUTED, PAPER, WIDTH, _color, _svg, _text


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema_version") != "jevarena-ranking/3":
        raise ValueError("V3 figures require the JevArena v3 ranking")
    rows = report.get("models")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("V3 figures need at least two same-panel models")
    if any(
        not isinstance(row, dict)
        or type(row.get("score")) not in (int, float)
        or not math.isfinite(row["score"])
        or not 0 <= row["score"] <= 100
        or set(row.get("axes", {})) != {"typed", "transfer"}
        for row in rows
    ):
        raise ValueError("V3 figure contains a malformed two-axis row")
    return sorted(rows, key=lambda row: (row["rank"], row["key"]))


def ranking_svg(report: dict[str, Any]) -> str:
    rows = _rows(report)
    height = 171 + 49 * len(rows)
    title = "JevArena v3 sealed-core ranking"
    caveat = (
        "8,147 sealed items · typed plus human transfer · rank among displayed models"
    )
    parts = _svg(height, title, caveat)
    parts += [
        _text(32, 42, title, size=24, weight=700),
        _text(
            32, 67, "Geometric mean of typed and human transfer", size=13, fill=MUTED
        ),
    ]
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
            f'<rect x="{x0}" y="{y}" width="{span}" height="25" rx="8" fill="#e8edfa"/>'
        )
        parts.append(
            f'<rect x="{x0}" y="{y}" width="{span * row["score"] / 100:.2f}" '
            f'height="25" rx="8" fill="{_color(row["group"])}"/>'
        )
        parts.append(
            _text(32, y + 18, f"{row['rank']}. {row['label']}", size=14, weight=600)
        )
        parts.append(
            _text(
                1028, y + 18, f"{row['score']:.2f}", size=14, weight=700, anchor="end"
            )
        )
    parts += [_text(32, height - 24, caveat, size=11, fill=MUTED), "</svg>"]
    return "\n".join(parts) + "\n"


def pareto_svg(report: dict[str, Any]) -> str:
    rows = _rows(report)
    if any(
        type(row.get("size_b")) not in (int, float)
        or not math.isfinite(row["size_b"])
        or row["size_b"] <= 0
        for row in rows
    ):
        raise ValueError("V3 Pareto needs actual positive parameter counts")
    title = "JevArena v3: size and sealed-core score"
    caveat = "Filled points are Pareto-efficient only within this same-panel roster"
    parts = _svg(670, title, caveat)
    parts += [
        _text(32, 42, title, size=24, weight=700),
        _text(
            32,
            67,
            "Actual parameters (billions, log scale) vs v3 score",
            size=13,
            fill=MUTED,
        ),
    ]
    left, right, top, bottom = 105, 960, 113, 562
    log_min = math.log10(min(row["size_b"] for row in rows)) - 0.14
    log_max = math.log10(max(row["size_b"] for row in rows)) + 0.14

    def x(value: float) -> float:
        return left + (math.log10(value) - log_min) / (log_max - log_min) * (
            right - left
        )

    def y(value: float) -> float:
        return bottom - value / 100 * (bottom - top)

    for tick in (0, 20, 40, 60, 80, 100):
        yy = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            _text(left - 12, yy + 5, str(tick), size=12, anchor="end", fill=MUTED)
        )
    for power in range(math.floor(log_min), math.ceil(log_max) + 1):
        for multiple in (1, 2, 5):
            tick = 10**power * multiple
            if log_min <= math.log10(tick) <= log_max:
                xx = x(tick)
                parts.append(
                    f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" stroke="{GRID}"/>'
                )
                parts.append(
                    _text(
                        xx,
                        bottom + 21,
                        f"{tick:g}",
                        size=11,
                        anchor="middle",
                        fill=MUTED,
                    )
                )
    frontier = sorted(
        (row for row in rows if row.get("pareto_frontier")),
        key=lambda row: (row["size_b"], row["score"]),
    )
    if len(frontier) > 1:
        points = " ".join(
            f"{x(row['size_b']):.1f},{y(row['score']):.1f}" for row in frontier
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#f4b642" stroke-width="2" stroke-dasharray="6 5"/>'
        )
    for row in sorted(rows, key=lambda value: bool(value.get("pareto_frontier"))):
        xx, yy = x(row["size_b"]), y(row["score"])
        color = _color(row["group"])
        fill = color if row.get("pareto_frontier") else PAPER
        parts.append(
            f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="8" fill="{fill}" stroke="{color}" stroke-width="2.5"/>'
        )
        parts.append(_text(xx + 13, yy - 12, row["label"], size=12, weight=600))
    parts += [_text(32, 643, caveat, size=11, fill=MUTED), "</svg>"]
    return "\n".join(parts) + "\n"


def axis_matrix_svg(report: dict[str, Any]) -> str:
    rows = _rows(report)
    height = 161 + 47 * len(rows)
    caveat = "Two separately reported sealed-core axes; public benchmarks are excluded"
    parts = _svg(height, "JevArena v3 component matrix", caveat)
    parts += [
        _text(32, 42, "JevArena v3 component matrix", size=24, weight=700),
        _text(32, 67, "Typed and human transfer · percentages", size=13, fill=MUTED),
    ]
    for column, label in enumerate(("Typed", "Human transfer")):
        parts.append(
            _text(460 + 220 * column, 103, label, size=13, weight=600, anchor="middle")
        )
    for index, row in enumerate(rows):
        yy = 120 + index * 47
        parts.append(
            _text(32, yy + 19, f"{row['rank']}. {row['label']}", size=13, weight=600)
        )
        for column, axis in enumerate(("typed", "transfer")):
            score = row["axes"][axis]
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("V3 matrix has an invalid axis fraction")
            shade = f"#{round(255 * (1 - score) + 49 * score):02x}{round(231 * (1 - score) + 91 * score):02x}{round(197 * (1 - score) + 255 * score):02x}"
            xx = 380 + 220 * column
            parts.append(
                f'<rect x="{xx}" y="{yy}" width="160" height="29" rx="7" fill="{shade}"/>'
            )
            parts.append(
                _text(
                    xx + 80,
                    yy + 20,
                    f"{100 * score:.1f}%",
                    size=13,
                    weight=700,
                    anchor="middle",
                    fill="#fff" if score >= 0.85 else INK,
                )
            )
    parts += [_text(32, height - 24, caveat, size=11, fill=MUTED), "</svg>"]
    return "\n".join(parts) + "\n"


def task_matrix_svg(report: dict[str, Any]) -> str:
    rows = _rows(report)
    typed = ("choice", "noul", "score")
    transfer = sorted(rows[0].get("task_scores", {}).get("transfer", {}))
    if len(transfer) != 15:
        raise ValueError("V3 task matrix needs all 15 human transfer tasks")
    for row in rows:
        scores = row.get("task_scores", {})
        if set(scores.get("typed", {})) != set(typed) or set(
            scores.get("transfer", {})
        ) != set(transfer):
            raise ValueError("V3 task matrix models do not share frozen tasks")
    columns = [("typed", kind, f"Typed / {kind.title()}") for kind in typed] + [
        ("transfer", task, f"Transfer / {task.replace('_', ' ')}") for task in transfer
    ]
    x0, pitch, cell_width = 275, 88, 81
    width = max(WIDTH, x0 + pitch * len(columns) + 35)
    height = 262 + 45 * len(rows)
    caveat = (
        "Typed: accuracy; transfer: macro-F1 with invalid answers counted as failure"
    )
    parts = _svg(height, "JevArena v3 model by task matrix", caveat, width=width)
    parts += [
        _text(32, 42, "JevArena v3: model by task", size=24, weight=700),
        _text(
            32, 67, "Three typed tasks and 15 human transfer tasks", size=13, fill=MUTED
        ),
    ]
    for index, (_, _, label) in enumerate(columns):
        xx = x0 + index * pitch + cell_width / 2
        parts.append(
            f'<text x="{xx:.1f}" y="194" transform="rotate(-55 {xx:.1f} 194)" fill="{INK}" font-family="system-ui, sans-serif" font-size="11" font-weight="600">{escape(label)}</text>'
        )
    for index, row in enumerate(rows):
        yy = 207 + index * 45
        parts.append(
            _text(32, yy + 19, f"{row['rank']}. {row['label']}", size=13, weight=600)
        )
        for column, (axis, key, _) in enumerate(columns):
            score = row["task_scores"][axis][key]
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("V3 task matrix contains an invalid score")
            shade = f"#{round(255 * (1 - score) + 49 * score):02x}{round(231 * (1 - score) + 91 * score):02x}{round(197 * (1 - score) + 255 * score):02x}"
            xx = x0 + column * pitch
            parts.append(
                f'<rect x="{xx}" y="{yy}" width="{cell_width}" height="29" rx="7" fill="{shade}"/>'
            )
            parts.append(
                _text(
                    xx + cell_width / 2,
                    yy + 20,
                    f"{100 * score:.1f}",
                    size=12,
                    weight=700,
                    anchor="middle",
                    fill="#fff" if score >= 0.85 else INK,
                )
            )
    parts += [_text(32, height - 24, caveat, size=11, fill=MUTED), "</svg>"]
    return "\n".join(parts) + "\n"
