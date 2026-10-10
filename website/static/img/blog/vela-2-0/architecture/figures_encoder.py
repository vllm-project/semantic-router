"""Editable Vela 2.0 0.3B operator diagrams, fixed-release source evidence.

Neural topology: vllm-sr/Vela-2.0-0.3B at
7162ab91b808bf36201cdd2bc87212f9d2c5c4db, ModernBERT SDPA in Transformers
v4.57.6. No weights or inference are required. Root renderer calls generate().
"""

# Diagram labels intentionally use Unicode mathematical glyphs.
# ruff: noqa: RUF001

from pathlib import Path

from architecture_svg import Scene


def math_box(s, cx, y, label, w=290, h=64, color="tensor", size=26):
    lines = list(label) if isinstance(label, (tuple, list)) else [label]
    s.raw(f'<g data-box="{cx-w/2} {y} {w} {h}">')
    s.rect(cx - w / 2, y, w, h, color)
    baseline = y + h / 2 + size * 0.34 - (len(lines) - 1) * size * 0.585
    s.text(cx, baseline, lines, size, math=True)
    s.raw("</g>")


def encoder_operators(path):
    s = Scene(
        1580,
        1470,
        "Vela 2.0 0.3B: attention and GEGLU operators",
        "Bottom-up detail of the 22-layer ModernBERT backbone. "
        "Twelve Q/K/V heads of width 64; YaRN on Q/K; bidirectional "
        "global or local padding mask. GEGLU width 1152. SDPA path.",
    )
    cx = 390
    math_box(s, cx, 1230, ("Attention input X", "X ∈ ℝ^(L×768)"), w=370, h=78)
    s.box(cx, 1120, ("Wqkv: 768 → 2304", "Bias-free projection"), w=370, h=78, size=26)
    s.box(
        cx,
        1015,
        ("Split Q / K / V", "12 heads · dₕ = 64"),
        w=350,
        h=74,
        color="tensor",
        size=26,
    )
    s.arrow(cx, 1230, cx, 1198)
    s.arrow(cx, 1120, cx, 1089)
    for x, name in ((180, "Q"), (390, "K"), (600, "V")):
        math_box(s, x, 900, (name, "12×L×64"), w=158, h=72)
        s.path([(cx, 1015), (cx, 991), (x, 991), (x, 972)])
    s.dot(cx, 991)
    for x in (180, 390):
        s.box(
            x,
            795,
            ("YaRN RoPE ×4", "θ = 160000"),
            w=200,
            h=70,
            color="attention",
            size=25,
        )
        s.arrow(x, 900, x, 865)
    math_box(
        s, 290, 655, ("QKᵀ / √64", "per head: L×L"), w=340, h=78, color="attention"
    )
    s.path([(180, 795), (180, 766), (230, 766), (230, 733)])
    s.path([(390, 795), (390, 766), (350, 766), (350, 733)])
    s.circle(290, 555, "+", r=20)
    s.arrow(290, 655, 290, 575)
    s.box(
        580,
        515,
        ("Padding + layer mask", "Local ±64 / global"),
        w=270,
        h=80,
        color="tensor",
        size=24,
    )
    s.arrow(445, 555, 310, 555)
    s.box(290, 450, "Softmax over keys", w=290, h=58, color="softmax", size=26)
    s.arrow(290, 535, 290, 508)
    s.box(
        cx,
        335,
        ("Attention weights × V", "12×L×64"),
        w=290,
        h=76,
        color="attention",
        size=25,
    )
    s.path([(290, 450), (290, 432), (cx, 432), (cx, 411)])
    s.path([(600, 900), (600, 878), (770, 878), (770, 373), (535, 373)])
    s.box(cx, 240, ("Concat heads", "L×768"), w=290, h=70, color="tensor", size=26)
    s.arrow(cx, 335, cx, 310)
    s.box(cx, 145, ("Wout: 768 → 768", "Bias-free projection"), w=330, h=70, size=25)
    s.arrow(cx, 240, cx, 215)
    math_box(s, cx, 55, "Attention(X) ∈ ℝ^(L×768)", w=420, h=60)
    s.arrow(cx, 145, cx, 115)

    fx = 1190
    math_box(s, fx, 1230, ("FFN input Y", "Y ∈ ℝ^(L×768)"), w=370, h=78)
    s.box(fx, 1120, ("Wi: 768 → 2304", "Bias-free projection"), w=370, h=78, size=26)
    s.arrow(fx, 1230, fx, 1198)
    s.box(
        fx,
        1015,
        ("Split into A and G", "dff = 1152"),
        w=350,
        h=74,
        color="tensor",
        size=26,
    )
    s.arrow(fx, 1120, fx, 1089)
    for x, name in ((1050, "A"), (1330, "G")):
        math_box(s, x, 900, (name, "L×1152"), w=185, h=72)
        s.path([(fx, 1015), (fx, 991), (x, 991), (x, 972)])
    s.dot(fx, 991)
    s.box(1050, 795, "GELU", w=190, h=64, color="ffn", size=27)
    s.arrow(1050, 900, 1050, 859)
    s.circle(fx, 610, "⊙", r=25, color="gate", size=30)
    s.path([(1050, 795), (1050, 610), (1165, 610)])
    s.path([(1330, 900), (1330, 610), (1215, 610)])
    s.box(fx, 385, ("Wo: 1152 → 768", "Bias-free projection"), w=370, h=78, size=26)
    s.arrow(fx, 585, fx, 463)
    math_box(s, fx, 165, "GEGLU(Y) ∈ ℝ^(L×768)", w=420, h=64)
    s.arrow(fx, 385, fx, 229)
    s.panel(cx, 1350, "(a) Multi-head attention")
    s.text(cx, 1388, "Global layers: 1, 4, 7, 10, 13, 16, 19, 22", 24)
    s.text(cx, 1422, "Other layers: local |i − j| ≤ 64", 24)
    s.panel(fx, 1350, "(b) GEGLU feed-forward")
    s.text(fx, 1388, "Activation on A; unactivated gate G", 24)
    s.text(fx, 1422, "Backbone dropout = 0; residuals shown in main figure", 24)
    s.save(path)


def encoder_readouts(path):
    s = Scene(
        1600,
        1470,
        "Vela 2.0 0.3B: schema-conditioned readouts",
        "Bottom-up option cosine plus raw-option MLP readout and "
        "word-first-subtoken versus label-marker cosine span readout. "
        "Five 768-to-256 biased projections reuse one shared LayerNorm. "
        "Learned tau and inference calibration temperatures are distinct.",
    )
    # The five selectors all originate in the same encoder-state tensor.
    math_box(
        s, 800, 1280, "Final-normalized encoder states H ∈ ℝ^(L×768)", w=1480, h=70
    )
    xs = (160, 420, 690, 1080, 1370)
    labels = (
        ("Option marker hₒ", "[O] or [ABS]"),
        ("Question marker hq", "[Q]"),
        ("Target-range mean p", "[SEG] + part tokens"),
        ("First word subtokens", "W×768"),
        ("[E] label markers", "E×768"),
    )
    widths = (250, 235, 250, 250, 250)
    for x, label, width in zip(xs, labels, widths, strict=True):
        s.box(x, 1150, label, w=width, h=76, color="tensor", size=24)
        s.arrow(x, 1280, x, 1226)
        s.box(x, 1035, "LayerNorm*", w=170, h=55, color="norm", size=25)
        s.arrow(x, 1150, x, 1090)
    names = ("Wₒ", "Wq", "Wp", "Wt", "We")
    for x, name in zip(xs, names, strict=True):
        s.box(x, 925, (name, "768 → 256"), w=210, h=76, color="linear", size=26)
        s.arrow(x, 1035, x, 1001)

    # Decision cosine branch: sum of option and question projections, then L2.
    s.circle(300, 825, "+", r=23)
    s.path([(160, 925), (160, 825), (277, 825)])
    s.path([(420, 925), (420, 825), (323, 825)])
    s.box(300, 730, "L2 normalize u", w=220, h=60, color="softmax", size=25)
    s.arrow(300, 802, 300, 790)
    s.box(690, 805, "L2 normalize v", w=220, h=60, color="softmax", size=25)
    s.arrow(690, 925, 690, 865)
    math_box(s, 500, 605, "dot(u, v) / τ", w=290, h=76, color="attention", size=28)
    s.path([(300, 730), (300, 705), (440, 705), (440, 681)])
    s.path([(690, 805), (690, 715), (560, 715), (560, 681)])
    # Distinct raw h_o branch: the custom readout LayerNorm is bypassed.
    s.box(
        190,
        535,
        ("MLP: 768 → 1536 → 1", "ReLU · Dropout(0.1)"),
        w=300,
        h=92,
        color="ffn",
        size=24,
    )
    s.path([(160, 1130), (25, 1130), (25, 581), (40, 581)])
    s.dot(160, 1130)
    # Join logits, then choose softmax / independent sigmoid.
    s.circle(340, 430, "+", r=23)
    s.path([(190, 535), (190, 430), (317, 430)])
    s.path([(500, 605), (500, 430), (363, 430)])
    s.text(590, 385, "z = cosine + MLP", 26, math=True)
    s.box(
        340,
        275,
        ("z / Ttype: softmax or sigmoid", "Choice · Noul · Score / Set"),
        w=600,
        h=82,
        color="softmax",
        size=25,
    )
    s.arrow(340, 407, 340, 357)
    s.box(
        340,
        125,
        ("Choice: argmax · Noul: P(yes)", "Score: E[level] · Set: p > θ"),
        w=600,
        h=88,
        color="tensor",
        size=25,
    )
    s.arrow(340, 275, 340, 213)

    # Span head: one projection pair, no broad span head and no separate Q input.
    for x in (1080, 1370):
        s.box(x, 805, "L2 normalize", w=205, h=60, color="softmax", size=25)
        s.arrow(x, 925, x, 865)
    math_box(
        s,
        1230,
        670,
        ("Z = t eᵀ / τspan", "Z ∈ ℝ^(W×E)"),
        w=490,
        h=84,
        color="attention",
        size=27,
    )
    s.path([(1080, 805), (1080, 780), (1150, 780), (1150, 754)])
    s.path([(1370, 805), (1370, 780), (1310, 780), (1310, 754)])
    s.box(
        1230,
        530,
        ("Sigmoid(Z / Tspan)", "Tspan = 0.29268"),
        w=430,
        h=78,
        color="softmax",
        size=26,
    )
    s.arrow(1230, 670, 1230, 608)
    # Noul view bypasses hard thresholding and decoding.
    s.path(
        [(1230, 530), (1230, 500), (1490, 500), (1490, 220), (1370, 220), (1370, 195)]
    )
    s.dot(1230, 500)
    s.box(
        1370,
        115,
        ("Noul view", "max word-label probability"),
        w=380,
        h=80,
        color="tensor",
        size=24,
    )
    s.box(
        1080,
        365,
        ("Max label per word · p > θ", "Merge neighbors · trim edges"),
        w=420,
        h=88,
        color="tensor",
        size=24,
    )
    s.path([(1230, 500), (1080, 500), (1080, 453)])
    s.box(
        1080,
        230,
        ("Labeled spans", "Unicode character offsets"),
        w=350,
        h=84,
        color="tensor",
        size=24,
    )
    s.arrow(1080, 365, 1080, 314)

    s.panel(420, 1385, "(a) Option cosine + MLP")
    s.panel(1230, 1385, "(b) Word–label cosine")
    s.text(
        800,
        1422,
        "* Shared readout LayerNorm parameters; learned τ; external calibration temperatures T.",
        24,
    )
    s.text(
        800,
        1454,
        "Dropout is inactive in inference. Additional span questions run as separate encoder sequences.",
        24,
    )
    s.save(path)


def generate(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        output_dir / "05-encoder-attention-geglu.svg",
        output_dir / "06-encoder-readouts.svg",
    ]
    encoder_operators(files[0])
    encoder_readouts(files[1])
    return files


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("output_dir", nargs="?", default=str(Path(__file__).parent))
    generate(p.parse_args().output_dir)
