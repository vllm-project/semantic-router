"""Publication diagrams using the visual grammar of Vaswani et al., Figures 1-2.

Original vector artwork. The computations represent the released Vela models.
All coordinates are explicit; labels and boxes are tagged for browser QA.
"""

from html import escape
from pathlib import Path

# RUF001 exceptions retain the multiplication glyph used in the artwork.
ROOT = Path(__file__).resolve().parent
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

    def save(self, stem):
        if "LayerNorm *" in "\n".join(
            self.p
        ) and "First-layer attention" not in "\n".join(self.p):
            old_height = self.height
            self.height += 50
            self.p[0] = (
                self.p[0]
                .replace(f'height="{old_height}"', f'height="{self.height}"')
                .replace(
                    f'viewBox="0 0 {self.width} {old_height}"',
                    f'viewBox="0 0 {self.width} {self.height}"',
                )
            )
            self.text(
                self.width / 2,
                old_height + 13,
                "* First attention LayerNorm is omitted in layer 1.",
                22,
            )
        path = ROOT / f"{stem}.svg"
        path.write_text("\n".join([*self.p, "</svg>"]) + "\n")
        return path


def encoder_layer(
    s, cx, top, w=290, kind="modernbert", n="22×", label=None  # noqa: RUF001
):
    """Expanded encoder layer, input at top+570, output at top-20.

    kind modernbert: pre-norm, separate residual adds, GEGLU.
    kind bert: post-norm add+norm, conventional GELU FFN.
    kind siglip/whisper: pre-norm, separate adds, conventional GELU FFN.
    """
    s.repeat(cx - w / 2 - 47, top, w + 120, 550, n)
    if kind == "bert":
        s.box(cx, top + 38, "Add & Norm", w, 55, "norm")
        s.box(
            cx, top + 128, ["Feed Forward", "Linear · GELU · Linear"], w, 98, "ffn", 24
        )
        s.box(cx, top + 272, "Add & Norm", w, 55, "norm")
        s.box(cx, top + 381, ["Multi-Head", "Self-Attention"], w, 96, "attention", 27)
        s.arrow(cx, top + 570, cx, top + 512)
        for dx in [-65, 0, 65]:
            s.path([(cx, top + 512), (cx + dx, top + 512), (cx + dx, top + 478)])
        s.arrow(cx, top + 380, cx, top + 328)
        s.arrow(cx, top + 271, cx, top + 227)
        s.arrow(cx, top + 127, cx, top + 94)
        s.arrow(cx, top + 37, cx, top - 20, arrow=False)
        side = cx + w / 2 + 44
        s.path(
            [
                (cx, top + 529),
                (side, top + 529),
                (side, top + 300),
                (cx + w / 2 + 1, top + 300),
            ]
        )
        s.path(
            [
                (cx, top + 249),
                (side, top + 249),
                (side, top + 65),
                (cx + w / 2 + 1, top + 65),
            ]
        )
    else:
        s.circle(cx, top + 48)
        ffn = (
            ["GEGLU", "Feed Forward"]
            if kind == "modernbert"
            else ["Feed Forward", "Linear · GELU · Linear"]
        )
        s.box(cx, top + 93, ffn, w, 90, "ffn", 26)
        s.box(cx, top + 219, "LayerNorm", w, 48, "norm", 25)
        s.circle(cx, top + 310)
        s.box(cx, top + 354, ["Multi-Head", "Self-Attention"], w, 78, "attention", 26)
        s.box(
            cx,
            top + 467,
            "LayerNorm *" if kind == "modernbert" else "LayerNorm",
            w,
            48,
            "norm",
            25,
        )
        s.arrow(cx, top + 570, cx, top + 516)
        for dx in [-65, 0, 65]:
            s.path(
                [
                    (cx, top + 466),
                    (cx, top + 452),
                    (cx + dx, top + 452),
                    (cx + dx, top + 433),
                ]
            )
        s.arrow(cx, top + 353, cx, top + 329)
        s.arrow(cx, top + 291, cx, top + 268)
        s.arrow(cx, top + 218, cx, top + 184)
        s.arrow(cx, top + 92, cx, top + 67)
        s.arrow(cx, top + 29, cx, top - 20, arrow=False)
        side = cx + w / 2 + 45
        s.path(
            [
                (cx, top + 534),
                (side, top + 534),
                (side, top + 310),
                (cx + 20, top + 310),
            ]
        )
        s.path(
            [(cx, top + 280), (side, top + 280), (side, top + 48), (cx + 20, top + 48)]
        )
        s.dot(cx, top + 534)
        s.dot(cx, top + 280)
    if label:
        s.text(cx, top + 607, label, 23)
    return {
        "input": (cx, top + 570),
        "output": (cx, top - 20),
        "top": top,
        "bottom": top + 550,
    }


def modernbert_input(s, cx, stack_top, w=290, input_label="Input tokens"):
    # The embedding norm doubles as the first attention pre-norm.
    y = stack_top + 601
    s.box(cx, y, "Embedding LayerNorm", w, 48, "norm", 24)
    s.arrow(cx, y - 1, cx, stack_top + 571, arrow=False)
    s.box(cx, y + 89, "Token Embedding", w, 66, "embedding", 26)
    s.arrow(cx, y + 88, cx, y + 50)
    s.arrow(cx, y + 205, cx, y + 157)
    s.text(cx, y + 239, input_label, 26)
    return y + 239


def linear_chain(s, cx, top, items, w=290, gap=30):
    """Items ordered top→bottom, connected bottom→top."""
    last = None
    for label, color, h in items:
        node = s.box(cx, top, label, w, h, color, 25)
        if last:
            s.arrow(cx, node["top"] - 1, cx, last["bottom"] + 1)
        last = node
        top += h + gap
    return last
