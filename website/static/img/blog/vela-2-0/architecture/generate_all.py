"""Rebuild the verified Vela 2.0 diagrams: python generate_all.py.
No network access or model execution. See README.md and architecture-spec.json.
"""

# Diagram labels intentionally use Unicode mathematical glyphs.
# ruff: noqa: RUF001

import json
from pathlib import Path

from architecture_svg import Scene

ROOT = Path(__file__).resolve().parent


def conn(s, x, a, b):
    s.arrow(x, a, x, b)


def model_figure(size, p, index):
    enc = size == "0.3B"
    s = Scene(
        1240,
        1570,
        f"Vela 2.0 {size}: model architecture",
        "Verified inference graph. Read bottom to top. Independent checkpoint.",
    )
    c, decision_x, r = 620, 320, 920
    s.text(decision_x, 58, "Decision outputs", 28)
    s.text(r, 58, "Labeled text spans", 28)
    s.box(
        decision_x,
        100,
        ["Calibration + softmax / sigmoid", "Choice · Noul · Score · Set"],
        w=460,
        h=88,
        color="softmax",
        size=24,
    )
    s.box(
        r,
        100,
        ["Sigmoid + threshold + merge", "Unicode character offsets"],
        w=460,
        h=88,
        color="softmax",
        size=24,
    )
    conn(s, decision_x, 99, 72)
    conn(s, r, 99, 72)
    if enc:
        dl = ["Cosine + option MLP", "projection width 256"]
        sl = ["Word–label cosine / τ_span", "projection width 256"]
        di = ["[Q] and [O] marker states", "+ target-part mean"]
        si = ["First sub-token of each word", "+ [E] label-marker states"]
    else:
        dl = ["CandidateHead: LayerNorm", "scaled bilinear + GELU MLP"]
        sl = ["SpanHeadV2: router OR broad", "separate weights; same topology"]
        di = ["Option endpoint states", "+ question-block endpoint"]
        si = ["Repeated target: first sub-tokens", "+ label-block mean states"]
    for x, head, features in [(decision_x, dl, di), (r, sl, si)]:
        s.box(x, 235, head, w=460, h=86, color="linear", size=24)
        s.box(x, 370, features, w=460, h=86, color="tensor", size=24)
        conn(s, x, 234, 189)
        conn(s, x, 369, 322)
    s.box(
        c,
        530,
        "Final LayerNorm" if enc else "Final RMSNorm",
        w=350,
        h=54,
        color="norm",
        size=26,
    )
    s.path([(c, 529), (c, 492), (decision_x, 492), (decision_x, 457)])
    s.path([(c, 492), (r, 492), (r, 457)])
    s.dot(c, 492)
    s.text(c, 480, f'H: sequence length × {p["hidden"]:,}', 22, math=True)
    s.repeat(338, 625, 568, 595, f'{p["layers"]}×')
    s.circle(c, 675)
    s.box(
        c,
        723,
        [
            "GEGLU feed forward" if enc else "SwiGLU feed forward",
            f'{p["hidden"]:,} → {p["ffn"]:,} → {p["hidden"]:,}',
        ],
        w=350,
        h=80,
        color="ffn",
        size=25,
    )
    s.box(c, 841, "LayerNorm" if enc else "RMSNorm", w=350, h=50, color="norm", size=25)
    s.circle(c, 935)
    mixer = (
        ["Bidirectional self-attention", "local / global layers"]
        if enc
        else ["Gated-DeltaNet (L)", "or gated causal GQA (F)"]
    )
    s.box(c, 983, mixer, w=350, h=85, color="attention", size=24)
    s.box(
        c, 1106, "LayerNorm*" if enc else "RMSNorm", w=350, h=50, color="norm", size=25
    )
    for a, b in [
        (656, 585),
        (722, 694),
        (840, 804),
        (916, 892),
        (982, 954),
        (1105, 1069),
    ]:
        conn(s, c, a, b)
    s.path([(c, 1188), (859, 1188), (859, 935), (639, 935)])
    s.path([(c, 907), (859, 907), (859, 675), (639, 675)])
    s.dot(c, 1188)
    s.dot(c, 907)
    if enc:
        s.box(c, 1254, "Embedding LayerNorm", w=350, h=54, color="norm", size=25)
        s.box(
            c,
            1350,
            ["Token embedding", "256,008 × 768"],
            w=350,
            h=68,
            color="embedding",
            size=25,
        )
        conn(s, c, 1253, 1157)
        conn(s, c, 1349, 1309)
        s.text(1015, 1014, ["12 heads × 64", "RoPE · YaRN ×4"], 23)
        s.text(1015, 1128, "* Identity in layer 1", 22)
        s.text(170, 774, ["Global layers", "1, 4, 7, …, 22"], 23)
        s.text(170, 893, ["Other layers", "|i − j| ≤ 64"], 23)
        s.text(c, 1461, "Schema + typed state · input limit 8,192 tokens", 25)
        conn(s, c, 1433, 1419)
        foot = "Bidirectional SDPA encoder with schema-conditioned decision and span readouts."
    else:
        s.box(
            c,
            1280,
            ["Token embedding", f'248,320 × {p["hidden"]:,}'],
            w=350,
            h=74,
            color="embedding",
            size=25,
        )
        conn(s, c, 1279, 1157)
        s.text(
            170,
            778,
            ["Layer schedule", "L → L → L → F", f'{p["layers"]//4} cycles'],
            23,
        )
        s.text(
            170,
            862,
            [
                f'{p["layers"]*3//4} linear layers',
                f'{p["layers"]//4} full-attention layers',
            ],
            22,
        )
        s.text(
            1022,
            1015,
            [f'GQA: {p["q_heads"]} Q / {p["kv_heads"]} KV', "head width 256"],
            23,
        )
        s.text(
            1022,
            1125,
            ["Linear heads:", f'{p["linear_k"]} Q,K / {p["linear_v"]} V', "width 128"],
            22,
        )
        s.text(
            c,
            1412,
            [
                "Typed state prefix + isolated question blocks",
                "input limit 16,384 tokens",
            ],
            25,
        )
        conn(s, c, 1382, 1355)
        foot = "Causal tree execution; the state prefix is reused per layer. Head computations use FP32."
    s.panel(c, 1510, f"Vela 2.0 {size}")
    s.text(c, 1547, foot, 21)
    s.save(ROOT / f"{index:02d}-vela-2.0-{size.lower()}-architecture.svg")


def tree_figure():
    s = Scene(
        1500,
        1190,
        "Vela 2.0 hybrid models: state-prefix and question isolation",
        "Runtime tree forward for 0.8B, 4B, and 9B. Activation and cache flow run upward.",
    )
    xs = [280, 750, 1220]
    for x, out, lab in zip(
        xs,
        ["Decision logits", "Word × label logits", "Decision logits"],
        ["Choice / Noul / Score", "Span", "Set"],
        strict=True,
    ):
        s.text(x, 68, out, 27)
        s.box(
            x,
            112,
            [lab + " readout", "from final-normalized states"],
            w=370,
            h=80,
            color="linear",
            size=24,
        )
        conn(s, x, 111, 83)
        s.box(
            x,
            245,
            ["Continue own question block", "through every backbone layer"],
            w=370,
            h=88,
            color="attention",
            size=24,
        )
        conn(s, x, 244, 193)
        s.box(
            x,
            390,
            ["Own block tokens", "position starts after the prefix"],
            w=370,
            h=80,
            color="embedding",
            size=23,
        )
        conn(s, x, 389, 334)
    s.box(
        750,
        620,
        ["Shared state-prefix pass", "F layers: KV; L layers: state + conv tail"],
        w=740,
        h=88,
        color="tensor",
        size=25,
    )
    for x in xs:
        side = x - 212
        s.path([(750, 619), (750, 550), (side, 550), (side, 289), (x - 186, 289)])
    s.dot(750, 550)
    s.text(715, 586, "Independent prefix forks", 23, anchor="end")
    s.box(
        750,
        780,
        [
            "Typed state: request / context / answer",
            "role-tagged text → prefix token embeddings",
        ],
        w=740,
        h=88,
        color="embedding",
        size=25,
    )
    conn(s, 750, 779, 709)
    s.text(750, 929, "Runtime request: one state and multiple questions", 27)
    conn(s, 750, 902, 869)
    for i, x in enumerate(xs, 1):
        s.text(x, 512, f"Question {i}", 24)
    s.text(
        750,
        1011,
        [
            "For a fixed rendered state, each question sees its prefix and its own causal block.",
            "Each extra span question uses another rendered sequence.",
            "Span targets > 2,048 tokens: 1,800-token windows, stride 1,536; mean word logits.",
        ],
        23,
    )
    s.panel(750, 1144, "State sharing and isolated question continuations")
    s.save(ROOT / "11-state-prefix-and-question-isolation.svg")


def main():
    specs = json.loads((ROOT / "architecture-spec.json").read_text())
    for i, size in enumerate(["0.3B", "0.8B", "4B", "9B"], 1):
        model_figure(size, specs["models"][size], i)
    tree_figure()
    for module in ["figures_encoder", "figures_hybrid", "figures_heads"]:
        m = __import__(module)
        m.generate(ROOT)


if __name__ == "__main__":
    main()
