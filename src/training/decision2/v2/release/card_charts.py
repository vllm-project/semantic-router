"""Model-card bar charts as standalone SVG (stdlib only).

One horizontal bar per model, sorted by score: the packaged model in the accent
colour, every comparator in neutral grey, the value at the end of each bar, a
light vertical grid and an explicit white background rectangle, so the Hub's
dark theme never shows the chart on a transparent (dark) canvas. The same style
serves every size; the charts read at the Hub's card width (about 760 px).
"""

from __future__ import annotations

import math
from typing import Any
from xml.sax.saxutils import escape

WIDTH = 760
PAD = 24
LABEL_WIDTH = 196
VALUE_ROOM = 64
ROW = 34
BAR = 20
TOP = 78
FONT = "Inter, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
INK = "#1F2328"
MUTED = "#59636E"
GRID = "#ECEEF1"
ACCENT = "#2563EB"
NEUTRAL = "#C3C8CF"
BACKGROUND = "#FFFFFF"


def _ticks(maximum: float, step: float) -> list[float]:
    return [i * step for i in range(int(math.floor(maximum / step + 1e-9)) + 1)]


def bar_chart(
    *,
    title: str,
    subtitle: str,
    rows: list[dict[str, Any]],
    maximum: float,
    step: float,
    digits: int = 1,
) -> str:
    """``rows``: ``{"label", "value", "highlight"}``; drawn in descending value order."""
    if not rows:
        raise ValueError("A chart needs at least one row")
    ordered = sorted(rows, key=lambda r: (-r["value"], r["label"]))
    x0 = PAD + LABEL_WIDTH
    span = WIDTH - x0 - VALUE_ROOM - PAD
    bottom = TOP + ROW * len(ordered)
    height = bottom + 40

    def x(value: float) -> float:
        return x0 + span * max(0.0, min(value, maximum)) / maximum

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" font-family="{escape(FONT)}">',
        f"<title>{escape(title)}</title>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{height}" fill="{BACKGROUND}"/>',
        f'<text x="{PAD}" y="34" font-size="17" font-weight="600" fill="{INK}">{escape(title)}</text>',
        f'<text x="{PAD}" y="56" font-size="12.5" fill="{MUTED}">{escape(subtitle)}</text>',
    ]
    for tick in _ticks(maximum, step):
        tx = x(tick)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{TOP - 8}" x2="{tx:.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{bottom + 20}" font-size="11" fill="{MUTED}" '
            f'text-anchor="middle">{tick:g}</text>'
        )
    parts.append(
        f'<line x1="{x0}" y1="{TOP - 8}" x2="{x0}" y2="{bottom}" stroke="{NEUTRAL}" stroke-width="1"/>'
    )
    for index, row in enumerate(ordered):
        y = TOP + ROW * index
        mid = y + BAR / 2 + 4.5
        strong = bool(row.get("highlight"))
        weight = "600" if strong else "400"
        width = max(x(row["value"]) - x0, 1.0)
        parts += [
            f'<text x="{x0 - 12}" y="{mid:.1f}" font-size="13.5" font-weight="{weight}" '
            f'fill="{INK if strong else MUTED}" text-anchor="end">{escape(row["label"])}</text>',
            f'<rect x="{x0}" y="{y}" width="{width:.1f}" height="{BAR}" rx="3" '
            f'fill="{ACCENT if strong else NEUTRAL}"/>',
            f'<text x="{x0 + width + 8:.1f}" y="{mid:.1f}" font-size="13" font-weight="{weight}" '
            f'fill="{INK}">{row["value"]:.{digits}f}</text>',
        ]
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def axis_maximum(values: list[float], step: float, floor: float) -> float:
    return max(floor, step * math.ceil(max(values) / step + 1e-9))
