"""Editable SVG primitives for neural-network computational diagrams.

No architecture or model dimensions are inferred. Coordinates use SVG units;
labels and boxes are tagged for browser layout checks.
"""

# Diagram labels intentionally use Unicode mathematical glyphs.
# ruff: noqa: RUF001

from html import escape
from pathlib import Path

COLORS = {
    "embedding": "#F7DCDF",
    "attention": "#FCE1B8",
    "ffn": "#C7E5F0",
    "norm": "#FBF8CB",
    "linear": "#DEDCEE",
    "softmax": "#D8EBD8",
    "tensor": "#E6ECE8",
    "gate": "#DED2EC",
    "group": "#F4F4F4",
    "white": "#FFFFFF",
}


class Scene:
    def __init__(self, width, height, title, desc=""):
        self.width, self.height = width, height
        self.p = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{escape(title)}</title><desc>{escape(desc)}</desc>',
            """<defs><marker id="arrow" markerWidth="12" markerHeight="10" refX="11" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L12 5 L0 10 L3 5 Z" fill="#111"/></marker></defs>
        <style>text{font-family:Arial,Helvetica,sans-serif;fill:#111;font-weight:400}.math{font-family:Georgia,'Times New Roman',serif;font-style:italic}.serif{font-family:Georgia,'Times New Roman',serif}</style>""",
        ]

    def raw(self, s):
        self.p.append(s)

    def text(self, x, y, label, size=27, anchor="middle", math=False, serif=False):
        lines = label if isinstance(label, (tuple, list)) else [label]
        klass = "math" if math else "serif" if serif else ""
        for i, line in enumerate(lines):
            if line:
                self.raw(
                    f'<text x="{x}" y="{y+i*size*1.17}" font-size="{size}" text-anchor="{anchor}" class="{klass}">{escape(str(line))}</text>'
                )

    def rect(self, x, y, w, h, color="white", r=7, stroke="#111", sw=2.2, dash=None):
        fill = COLORS.get(color, color)
        self.raw(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
            + (f' stroke-dasharray="{dash}"' if dash else "")
            + "/>"
        )

    def box(self, cx, y, label, w=290, h=58, color="linear", size=27):
        self.raw(f'<g data-box="{cx-w/2} {y} {w} {h}">')
        self.rect(cx - w / 2, y, w, h, color)
        lines = label if isinstance(label, (tuple, list)) else [label]
        baseline = y + h / 2 + size * 0.34 - (len(lines) - 1) * size * 0.585
        self.text(cx, baseline, lines, size)
        self.raw("</g>")
        return {
            "x": cx,
            "top": y,
            "bottom": y + h,
            "left": cx - w / 2,
            "right": cx + w / 2,
        }

    def path(self, points, arrow=True, dash=False, color="#111", width=2.3):
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        self.raw(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'
            + (' stroke-dasharray="7 6"' if dash else "")
            + (' marker-end="url(#arrow)"' if arrow else "")
            + "/>"
        )

    def arrow(self, x1, y1, x2, y2, **kw):
        self.path([(x1, y1), (x2, y2)], **kw)

    def circle(self, cx, cy, symbol="+", r=18, color="white", size=29):
        self.raw(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{COLORS.get(color,color)}" stroke="#111" stroke-width="2.1"/>'
        )
        self.text(cx, cy + size * 0.31, symbol, size)

    def dot(self, x, y):
        self.raw(f'<circle cx="{x}" cy="{y}" r="3.5" fill="#111"/>')

    def panel(self, cx, y, label):
        self.text(cx, y, label, 31, serif=True)

    def repeat(self, x, y, w, h, label):
        self.rect(x, y, w, h, "group", r=23, sw=2.2)
        self.text(x - 24, y + h / 2 + 9, label, 30, anchor="end", math=True)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([*self.p, "</svg>"]) + "\n", encoding="utf-8")
        return path


def transformer_layer(
    s,
    cx,
    top,
    w=290,
    n="N×",
    pre_norm=True,
    norm="LayerNorm",
    attention=("Multi-Head", "Self-Attention"),
    ffn=("Feed Forward", "Linear · GELU · Linear"),
):
    """Layout starter; caller must verify these sublayers against its model.

    Input: (cx, top+570). Output: (cx, top-20), without terminal arrowhead.
    No final norm, position mechanism, or first-layer exception is implied.
    """
    s.repeat(cx - w / 2 - 47, top, w + 120, 550, n)
    side = cx + w / 2 + 44
    if pre_norm:
        s.circle(cx, top + 48)
        s.box(cx, top + 93, ffn, w, 90, "ffn", 25)
        s.box(cx, top + 219, norm, w, 48, "norm", 25)
        s.circle(cx, top + 310)
        s.box(cx, top + 354, attention, w, 78, "attention", 26)
        s.box(cx, top + 467, norm, w, 48, "norm", 25)
        s.arrow(cx, top + 570, cx, top + 516)
        for dx in (-65, 0, 65):
            s.path(
                [
                    (cx, top + 466),
                    (cx, top + 452),
                    (cx + dx, top + 452),
                    (cx + dx, top + 433),
                ]
            )
        for y1, y2 in ((353, 329), (291, 268), (218, 184), (92, 67)):
            s.arrow(cx, top + y1, cx, top + y2)
        s.arrow(cx, top + 29, cx, top - 20, arrow=False)
        for branch, join in ((534, 310), (280, 48)):
            s.path(
                [
                    (cx, top + branch),
                    (side, top + branch),
                    (side, top + join),
                    (cx + 20, top + join),
                ]
            )
            s.dot(cx, top + branch)
    else:
        s.box(cx, top + 38, f"Add & {norm}", w, 55, "norm", 25)
        s.box(cx, top + 128, ffn, w, 98, "ffn", 24)
        s.box(cx, top + 272, f"Add & {norm}", w, 55, "norm", 25)
        s.box(cx, top + 381, attention, w, 96, "attention", 27)
        s.arrow(cx, top + 570, cx, top + 512)
        for dx in (-65, 0, 65):
            s.path([(cx, top + 512), (cx + dx, top + 512), (cx + dx, top + 478)])
        for y1, y2 in ((380, 328), (271, 227), (127, 94)):
            s.arrow(cx, top + y1, cx, top + y2)
        s.arrow(cx, top + 37, cx, top - 20, arrow=False)
        for branch, join in ((529, 300), (249, 65)):
            s.path(
                [
                    (cx, top + branch),
                    (side, top + branch),
                    (side, top + join),
                    (cx + w / 2 + 1, top + join),
                ]
            )
            s.dot(cx, top + branch)
    return {"input": (cx, top + 570), "output": (cx, top - 20)}
