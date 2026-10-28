# ruff: noqa: RUF001  (figure labels use typographic characters on purpose)
"""Vela 3.0 architecture figures, in the Vela 2.0 kit's style (architecture_svg.Scene). python3 vela3_figures.py

Every dimension is read from vela3-spec.json, which is built by build_vela3_spec.py from the release packages' own config.json /
vela3_config.json / vela3v_config.json and adapter safetensors headers (pkg_src/, sources in README.md (README-vela3.md in the research kit)).
Static source inspection only: no weights are loaded and no model is run. Read every figure bottom to top.
"""

import json
from pathlib import Path

from architecture_svg import Scene

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "vela3_out"
SPEC = json.loads((ROOT / "vela3-spec.json").read_text())


def conn(s, x, a, b):
    s.arrow(x, a, x, b)


def group(s, x, y, w, h, label=None, dash=None):
    s.rect(x, y, w, h, "group", r=23, sw=2.2, dash=dash)
    if label:
        s.text(x + w / 2, y + 36, label, 25, serif=True)


# =====================================================================================================================
# 01 vela3-fast
# =====================================================================================================================
def fast_figure():
    p = SPEC["vela3-fast"]
    s = Scene(
        1560,
        1700,
        "vela3-fast: model architecture",
        "UltraBERT-380M bidirectional encoder. Frozen decision backbone with its decision readout; one routed adapter per "
        "non-decision question. Read bottom to top.",
    )
    c, lx, r, ax = 600, 330, 860, 1320
    s.text(lx, 58, "Decisions", 28)
    s.text(r, 58, "Router answers · labeled spans", 28)
    s.box(
        lx,
        100,
        ["Softmax over the options", "Choice · Noul = p(yes)"],
        w=440,
        h=88,
        color="softmax",
        size=24,
    )
    s.box(
        r,
        100,
        [
            "Softmax / sigmoid · threshold + merge",
            "options, sets, character-offset spans",
        ],
        w=500,
        h=88,
        color="softmax",
        size=23,
    )
    conn(s, lx, 99, 72)
    conn(s, r, 99, 72)
    s.box(
        lx,
        235,
        ["Cosine / τ + option MLP", f"projection width {p['proj_dim']}"],
        w=440,
        h=86,
        color="linear",
        size=24,
    )
    s.box(
        r,
        235,
        ["Adapter head: cosine + MLP", "or word–label cosine / τ_span"],
        w=500,
        h=86,
        color="linear",
        size=24,
    )
    s.box(
        lx,
        370,
        ["[Q] and [O] marker states", "+ state-part mean"],
        w=440,
        h=86,
        color="tensor",
        size=24,
    )
    s.box(
        r,
        370,
        ["[O] markers · first sub-token", "of each word · [E] label markers"],
        w=500,
        h=86,
        color="tensor",
        size=24,
    )
    for x in (lx, r):
        conn(s, x, 234, 189)
        conn(s, x, 369, 322)
    s.box(c, 530, "Final RMSNorm", w=350, h=54, color="norm", size=26)
    s.path([(c, 529), (c, 492), (lx, 492), (lx, 457)])
    s.path([(c, 492), (r, 492), (r, 457)])
    s.dot(c, 492)
    s.text(c, 480, f"H: sequence length × {p['hidden']:,}", 22, math=True)
    s.text(lx - 10, 520, "frozen pass", 21, anchor="end")
    s.text(r + 10, 520, "adapter pass", 21, anchor="start")
    s.repeat(c - 282, 625, 568, 595, f"{p['layers']}×")
    s.circle(c, 675)
    s.box(
        c,
        723,
        ["SwiGLU feed forward", f"{p['hidden']:,} → {p['ffn']:,} → {p['hidden']:,}"],
        w=350,
        h=80,
        color="ffn",
        size=25,
    )
    s.box(c, 841, "RMSNorm", w=350, h=50, color="norm", size=25)
    s.circle(c, 935)
    s.box(
        c,
        983,
        ["Bidirectional self-attention", "local / global layers"],
        w=350,
        h=85,
        color="attention",
        size=24,
    )
    s.box(c, 1106, "RMSNorm", w=350, h=50, color="norm", size=25)
    for a, b in [
        (656, 585),
        (722, 694),
        (840, 804),
        (916, 892),
        (982, 954),
        (1105, 1069),
    ]:
        conn(s, c, a, b)
    s.path([(c, 1188), (c + 239, 1188), (c + 239, 935), (c + 19, 935)])
    s.path([(c, 907), (c + 239, 907), (c + 239, 675), (c + 19, 675)])
    s.dot(c, 1188)
    s.dot(c, 907)
    s.box(
        c,
        1290,
        ["Token embedding", f"{p['vocab']:,} × {p['hidden']:,}"],
        w=350,
        h=74,
        color="embedding",
        size=25,
    )
    conn(s, c, 1289, 1157)
    s.text(
        c + 195,
        1318,
        [
            f"{p['vocab_base']:,} UltraBERT tokens",
            f"+ {p['markers']} markers: [Q] [O] [ABS] [E]",
            "[SEP_SCHEMA] [SEG_user / context / answer]",
        ],
        21,
        anchor="start",
    )
    s.text(
        c,
        1470,
        f"Question schema + typed state · one input of up to {p['max_len']:,} tokens",
        25,
    )
    conn(s, c, 1445, 1366)
    s.text(
        130,
        760,
        ["Global layers", "3, 6, 9, …, 24", f"RoPE θ = {p['rope_global']:,}"],
        22,
    )
    s.text(
        130,
        880,
        ["Other layers", f"|i − j| ≤ {p['window']}", f"RoPE θ = {p['rope_local']:,}"],
        22,
    )
    s.text(130, 1010, [f"{p['heads']} heads × {p['head_dim']}", "no bias"], 22)
    # the adapter column
    group(s, 1100, 625, 440, 660, "Routed adapter (one per pass)", dash="9 7")
    s.box(
        ax,
        700,
        ["LoRA ΔW = B A · α / r", "q, k, v, o, gate, up, down", "all 24 layers"],
        w=390,
        h=110,
        color="gate",
        size=22,
    )
    s.box(
        ax,
        845,
        ["Own copies of the top layers", "spans 20–23 · router 16–23", "domain 12–23"],
        w=390,
        h=110,
        color="linear",
        size=22,
    )
    s.box(
        ax,
        990,
        ["Own readout head", "norm, projections, τ"],
        w=390,
        h=80,
        color="linear",
        size=22,
    )
    s.box(
        ax,
        1105,
        ["Every other weight: the frozen", "backbone's, shared by reference"],
        w=390,
        h=80,
        color="tensor",
        size=21,
    )
    s.path([(1100, 920), (c + 240, 920)], dash=True, arrow=True)
    s.text(
        ax, 1222, ["Rank 64 (α 128); NER and", "relation adapters rank 128 (α 256)"], 22
    )
    s.panel(c, 1580, "vela3-fast (0.4B)")
    s.text(
        c,
        1620,
        "Decisions run on the frozen backbone with no adapter; every other question runs one routed adapter pass.",
        21,
    )
    s.text(
        c,
        1652,
        "Bidirectional SDPA encoder. Decisions run under bf16 autocast on GPU; adapter passes run in fp32, as evaluated.",
        21,
    )
    s.save(OUT / "01-vela3-fast-architecture.svg")


# =====================================================================================================================
# 02 vela3-fast-vision
# =====================================================================================================================
def vision_figure():
    p = SPEC["vela3-fast-vision"]
    t, v = p["text"], p["vision"]
    s = Scene(
        1640,
        1560,
        "vela3-fast-vision: model architecture",
        "Three UltraVBERT backbone files on the vision side, each with its own task head, plus the frozen text encoder with "
        "the text adapters. Each question runs on one network. Read bottom to top.",
    )
    xs = [250, 640, 1030]
    tops = [
        ("Image + text decisions", "Choice · Noul"),
        ("Region pointing", "box + point"),
        ("Image hallucination spans", "· OCR check"),
    ]
    heads = [
        ("Mean of each option's tokens", "→ head → softmax"),
        ("One logit per image token", "64 × 64 px cells"),
        ("One logit per answer word", "first sub-token"),
    ]
    backs = [
        ("Decision-trained backbone", "backbone_h2 · head only"),
        ("Decision-trained backbone", "backbone_v2 + LoRA r64"),
        ("Original UltraVBERT", "model + LoRA r64"),
    ]
    for x, (a, b), hd, bk in zip(xs, tops, heads, backs, strict=True):
        s.text(x, 58, a, 25)
        s.text(x, 88, b, 22)
        s.box(x, 130, hd, w=350, h=80, color="softmax", size=22)
        s.box(
            x,
            260,
            ["Head: Linear · GELU", "· LayerNorm · Linear → 1"],
            w=350,
            h=80,
            color="linear",
            size=22,
        )
        s.box(
            x,
            390,
            bk,
            w=350,
            h=80,
            color="gate" if "LoRA" in bk[1] else "tensor",
            size=22,
        )
        conn(s, x, 129, 100)
        conn(s, x, 259, 211)
        conn(s, x, 389, 341)
    # shared UltraVBERT topology (drawn once)
    group(
        s,
        60,
        520,
        1160,
        690,
        "UltraVBERT topology of each backbone file (weights differ per file)",
    )
    s.box(
        640,
        590,
        [
            "UltraBERT-150M text encoder, bidirectional",
            f"{t['layers']} layers · hidden {t['hidden']} · SwiGLU {t['hidden']} → {t['ffn']:,} → {t['hidden']}",
            f"{t['heads']} heads × {t['head_dim']} · local |i − j| ≤ {t['window']} · every third layer global",
        ],
        w=800,
        h=120,
        color="attention",
        size=22,
    )
    s.box(
        330,
        850,
        ["Image tokens", f"{v['tokens_1024']} per image at {v['edge']} px"],
        w=380,
        h=80,
        color="tensor",
        size=22,
    )
    s.box(
        950,
        850,
        ["Text tokens", "question, options, state"],
        w=380,
        h=80,
        color="embedding",
        size=22,
    )
    s.path([(330, 849), (330, 760), (560, 760), (560, 711)])
    s.path([(950, 849), (950, 760), (720, 760), (720, 711)])
    s.box(
        330,
        980,
        ["Pixel shuffle ×4", "+ connector (LayerNorm)"],
        w=380,
        h=80,
        color="linear",
        size=22,
    )
    s.box(
        330,
        1105,
        [
            "SigLIP2-B/16 vision encoder",
            f"{v['layers']} layers · hidden {v['hidden']} · {v['heads']} heads",
        ],
        w=380,
        h=80,
        color="ffn",
        size=22,
    )
    conn(s, 330, 979, 931)
    conn(s, 330, 1104, 1061)
    s.text(330, 1240, [f"2 × 2 tiles of {v['tile']} px", "+ one global view"], 22)
    conn(s, 330, 1218, 1186)
    s.text(
        950,
        1240,
        ["state up to", f"{p['decision_state_tokens']:,} tokens (decisions)"],
        22,
    )
    conn(s, 950, 1218, 931)
    for x in xs:
        s.path([(x, 520), (x, 471)])
    # text side
    tx = 1430
    group(s, 1260, 120, 340, 1090, "Text side (frozen)")
    s.box(
        tx,
        230,
        ["Router answers", "· text spans · NER"],
        w=290,
        h=80,
        color="softmax",
        size=22,
    )
    s.box(
        tx,
        380,
        ["vela3-fast runtime", "heads per adapter"],
        w=290,
        h=80,
        color="linear",
        size=22,
    )
    s.box(
        tx,
        530,
        [
            "Text adapters",
            "router (r64) · span (r64)",
            "NER (r128) · domain",
            "· feedback",
        ],
        w=290,
        h=150,
        color="gate",
        size=21,
    )
    s.box(
        tx,
        770,
        ["Original UltraVBERT", "text encoder", "UltraBERT-150M, frozen"],
        w=290,
        h=110,
        color="attention",
        size=21,
    )
    conn(s, tx, 229, 178)
    conn(s, tx, 379, 311)
    conn(s, tx, 529, 461)
    conn(s, tx, 769, 681)
    s.text(
        tx,
        960,
        [
            "Non-router text",
            "decisions go to the",
            "decision-trained",
            "backbone (left)",
        ],
        22,
    )
    s.text(
        820,
        1300,
        "Request: state, questions, optional images (decisions up to 3; spans and boxes 1)",
        25,
    )
    s.path([(1300, 1300), (tx, 1300), (tx, 1211)])
    s.panel(820, 1400, "vela3-fast-vision (0.3B)")
    s.text(
        820,
        1440,
        "Each question is answered by one network: decision, box or span backbone, or the text side.",
        21,
    )
    s.text(
        820,
        1472,
        "The decision head has no LoRA: the whole decision-trained backbone is its network.",
        21,
    )
    s.save(OUT / "02-vela3-fast-vision-architecture.svg")


# =====================================================================================================================
# 03-06 the d3 sizes
# =====================================================================================================================
def d3_figure(name, index):
    p = SPEC[name]
    s = Scene(
        1560,
        1700,
        f"{name}: model architecture",
        f"The frozen {p['base']} decoder (Qwen3.5 hybrid layers) with row-gated LoRA adapters and Vela span heads. Read bottom to top.",
    )
    c, lx, r, ax = 600, 330, 860, 1320
    s.text(lx, 58, "Decisions", 28)
    s.text(r, 58, "Labeled text spans", 28)
    s.box(
        lx,
        100,
        ["Softmax over option codes", "Choice · Noul · Score"],
        w=440,
        h=88,
        color="softmax",
        size=24,
    )
    s.box(
        r,
        100,
        ["Sigmoid + threshold + merge", "character-offset spans"],
        w=500,
        h=88,
        color="softmax",
        size=24,
    )
    conn(s, lx, 99, 72)
    conn(s, r, 99, 72)
    s.box(
        lx,
        235,
        ["d3 code readout (unchanged)", "option-code logits, last position"],
        w=440,
        h=86,
        color="linear",
        size=23,
    )
    s.box(
        r,
        235,
        ["SpanHeadV2 (Vela 2.0 topology)", "span · NER · relation · image halu"],
        w=500,
        h=86,
        color="linear",
        size=23,
    )
    s.box(
        lx,
        370,
        ["d3 prompt: State · Question", "· Options under answer codes"],
        w=440,
        h=86,
        color="tensor",
        size=23,
    )
    s.box(
        r,
        370,
        ["First sub-token of each word", "+ mean of each option line"],
        w=500,
        h=86,
        color="tensor",
        size=23,
    )
    for x in (lx, r):
        conn(s, x, 234, 189)
        conn(s, x, 369, 322)
    s.box(c, 530, "Final RMSNorm", w=350, h=54, color="norm", size=26)
    s.path([(c, 529), (c, 492), (lx, 492), (lx, 457)])
    s.path([(c, 492), (r, 492), (r, 457)])
    s.dot(c, 492)
    s.text(c, 480, f"H: sequence length × {p['hidden']:,}", 22, math=True)
    s.repeat(c - 282, 625, 568, 595, f"{p['layers']}×")
    s.circle(c, 675)
    s.box(
        c,
        723,
        ["SwiGLU feed forward", f"{p['hidden']:,} → {p['ffn']:,} → {p['hidden']:,}"],
        w=350,
        h=80,
        color="ffn",
        size=25,
    )
    s.box(c, 841, "RMSNorm", w=350, h=50, color="norm", size=25)
    s.circle(c, 935)
    s.box(
        c,
        983,
        ["Gated-DeltaNet (L)", "or gated GQA (F)"],
        w=350,
        h=85,
        color="attention",
        size=24,
    )
    s.box(c, 1106, "RMSNorm", w=350, h=50, color="norm", size=25)
    for a, b in [
        (656, 585),
        (722, 694),
        (840, 804),
        (916, 892),
        (982, 954),
        (1105, 1069),
    ]:
        conn(s, c, a, b)
    s.path([(c, 1188), (c + 239, 1188), (c + 239, 935), (c + 19, 935)])
    s.path([(c, 907), (c + 239, 907), (c + 239, 675), (c + 19, 675)])
    s.dot(c, 1188)
    s.dot(c, 907)
    s.box(
        c,
        1290,
        ["Token embedding", f"{p['vocab']:,} × {p['hidden']:,}"],
        w=350,
        h=74,
        color="embedding",
        size=25,
    )
    conn(s, c, 1289, 1157)
    s.box(
        c - 400,
        1290,
        [
            "Vision tower (images)",
            f"{p['vision_depth']} layers · {p['vision_hidden']:,} → {p['hidden']:,}",
        ],
        w=330,
        h=74,
        color="ffn",
        size=21,
    )
    s.path([(c - 400, 1289), (c - 400, 1240), (c - 2, 1240)])
    s.text(c - 400, 1410, ["image tokens", "join the sequence"], 21)
    s.text(
        c,
        1525,
        (
            "d3 prompt; the state is read once, every question branches from it"
            if name == "vela3-turbo"
            else "d3 prompt, read with noncausal full attention (left-padded)"
        ),
        25,
    )
    conn(s, c, 1500, 1366)
    s.text(
        130, 760, ["Layer schedule", "L → L → L → F", f"{p['layers'] // 4} cycles"], 22
    )
    s.text(
        130,
        870,
        [f"GQA: {p['q_heads']} Q / {p['kv_heads']} KV", f"head width {p['head_dim']}"],
        22,
    )
    s.text(
        130,
        960,
        [
            "Linear heads:",
            f"{p['lin_k']} Q,K / {p['lin_v']} V",
            f"width {p['lin_dim']}",
        ],
        22,
    )
    group(s, 1100, 625, 440, 760, "Row-gated LoRA adapters", dash="9 7")
    y = 690
    for a in p["adapters"]:
        s.box(
            ax,
            y,
            [a["name"], f"r {a['r']} · layers {a['first']}–{p['layers'] - 1}"],
            w=390,
            h=70,
            color="gate",
            size=21,
        )
        y += 82
    s.path([(1100, 920), (c + 240, 920)], dash=True, arrow=True)
    s.text(
        ax,
        y + 30,
        ["At most one adapter per pass;", "off rows keep base(x) bit for bit"],
        21,
    )
    s.panel(c, 1600, f"{name} ({p['size']})")
    s.text(
        c,
        1640,
        f"Base {p['base']} at {p['base_rev'][:8]}, frozen. Ordinary questions run with every adapter off; spans add one pass per question.",
        21,
    )
    s.save(OUT / f"{index:02d}-{name}-architecture.svg")


# =====================================================================================================================
# 07 question routing (both families)
# =====================================================================================================================
def routing_figure():
    s = Scene(
        1640,
        1170,
        "Vela 3.0: one request, every question",
        "How system_one sends each question to a network, in vela3-fast (left) and the d3 sizes (right). Read top to bottom.",
    )

    def tree(cx, title, rows, foot):
        s.text(cx, 60, title, 28, serif=True)
        s.box(
            cx,
            100,
            ["system_one(state, questions)", "each question routed on its own"],
            w=640,
            h=84,
            color="tensor",
            size=24,
        )
        y = 250
        for cond, dest, col in rows:
            s.box(cx - 165, y, cond, w=320, h=84, color="norm", size=21)
            s.box(cx + 170, y, dest, w=330, h=84, color=col, size=21)
            s.arrow(cx - 4, y + 42, cx + 3, y + 42)
            y += 112
        s.path([(cx, 184), (cx, 212), (cx - 345, 212), (cx - 345, y - 70)], arrow=False)
        for k in range(len(rows)):
            yy = 250 + k * 112 + 42
            s.arrow(cx - 345, yy, cx - 327, yy)
        s.text(cx, y + 20, foot, 21)

    tree(
        420,
        "(a) vela3-fast",
        [
            (
                ["choice / noul,", "not a router text"],
                ["Decision backbone", "no adapter"],
                "softmax",
            ),
            (
                ["choice / set with one", "of the router texts"],
                ["Router adapter or", "domain / fact-check / feedback"],
                "gate",
            ),
            (
                ["span: PII or", "hallucination label set"],
                ["Span-set adapter", "(own thresholds)"],
                "gate",
            ),
            (
                ["span: evidence", "label or wording"],
                ["Evidence adapter", "one window ≤ 32,768"],
                "gate",
            ),
            (["span: relation", "wording"], ["Relation adapter", "rank 128"], "gate"),
            (
                ["span: names-only", "entity labels"],
                ["NER adapter", "rank 128"],
                "gate",
            ),
            (
                ["any other span", "question"],
                ["Span adapter", "8,192-token windows"],
                "gate",
            ),
        ],
        [
            "Rules are checked top to bottom. Adapter passes share the",
            "frozen weights; the decision pass never loads an adapter.",
        ],
    )
    tree(
        1220,
        "(b) vela3-lite · -mini · -flash · -turbo",
        [
            (
                ["choice / noul / score,", "not a router id"],
                ["d3 base", "every adapter off"],
                "softmax",
            ),
            (
                ["router question id", "(attack, domain, …)"],
                ["Decision LoRA (some ids:", "their own LoRA)"],
                "gate",
            ),
            (
                ["span: PII / halu / toxic", "label set"],
                ["Span LoRA + span head", "(halu + image: image-halu†)"],
                "gate",
            ),
            (
                ["span: relation", "wording"],
                ["Relation LoRA", "+ relation head"],
                "gate",
            ),
            (
                ["span: evidence or", "answer wording"],
                ["Span LoRA", "+ span head"],
                "gate",
            ),
            (
                ["span: entity question,", "open labels"],
                ["NER LoRA + NER head", "(+ label definitions*)"],
                "gate",
            ),
            (["any other span", "question"], ["Span LoRA", "+ span head"], "gate"),
        ],
        [
            "* vela3-mini and -flash write a one-line definition per",
            "named label (adapters off, causal), cached per label set.",
            "† not on vela3-turbo. Rules are checked top to bottom.",
        ],
    )
    s.save(OUT / "07-question-routing.svg")


# =====================================================================================================================
# 08 gated LoRA operator
# =====================================================================================================================
def gated_lora_figure():
    s = Scene(
        1500,
        1040,
        "Vela 3.0 d3 sizes: the row-gated LoRA linear layer",
        "MultiLoRA (vela3_adapters.py): every adapted Linear layer of the frozen base. Read bottom to top.",
    )
    s.raw('<g transform="translate(0,-230)">')
    cx = 560
    s.box(
        cx,
        1080,
        ["Layer input x", "rows of a batch"],
        w=380,
        h=80,
        color="tensor",
        size=25,
    )
    s.box(
        330,
        900,
        ["Frozen base Linear", "y = W x (+ b)"],
        w=360,
        h=84,
        color="linear",
        size=25,
    )
    s.box(
        800,
        900,
        ["A: in → r (fp32)", "of the active adapter"],
        w=360,
        h=84,
        color="gate",
        size=24,
    )
    s.path([(cx, 1079), (cx, 1020), (330, 1020), (330, 985)])
    s.path([(cx, 1020), (800, 1020), (800, 985)])
    s.dot(cx, 1020)
    s.box(
        800, 770, ["B: r → out (fp32)", "× α / r"], w=360, h=84, color="gate", size=24
    )
    conn(s, 800, 899, 855)
    s.circle(560, 620, "+", r=22)
    s.path([(330, 899), (330, 620), (538, 620)])
    s.path([(800, 769), (800, 620), (582, 620)])
    s.box(
        560,
        470,
        ["Row gate: torch.where(rows,", "y + Δ, y)"],
        w=520,
        h=84,
        color="norm",
        size=24,
    )
    conn(s, 560, 597, 555)
    s.path([(330, 700), (160, 700), (160, 512), (300, 512)], dash=True)
    s.dot(330, 700)
    s.text(150, 690, ["y for rows", "off the gate"], 21, anchor="end")
    s.box(
        560,
        320,
        ["Layer output", "base(x) bit for bit on off rows"],
        w=520,
        h=84,
        color="tensor",
        size=24,
    )
    conn(s, 560, 469, 405)
    s.box(
        1220,
        560,
        ["Gate state", "one adapter name or none", "+ optional row mask"],
        w=420,
        h=110,
        color="gate",
        size=23,
    )
    s.path([(1010, 560), (822, 512)], dash=True)
    s.box(
        1220,
        760,
        ["No adapter active:", "the layer returns base(x)", "itself"],
        w=420,
        h=110,
        color="tensor",
        size=23,
    )
    s.text(
        1220,
        960,
        [
            "Decision LoRAs: every layer",
            "Span / NER / relation LoRAs:",
            "the upper layers only",
        ],
        22,
    )
    s.panel(
        750,
        1240,
        "Row-gated LoRA: an adapter changes only the rows it is switched on for",
    )
    s.raw("</g>")
    s.save(OUT / "08-row-gated-lora.svg")


# =====================================================================================================================
# 09 vela3-fast decision readout, distilled from d3
# =====================================================================================================================
def readout_figure():
    p = SPEC["vela3-fast"]
    s = Scene(
        1600,
        1380,
        "vela3-fast: the decision readout and its distillation target",
        "Decision readout of the frozen backbone (vela3_encoder.DecisionEncoder) and the soft target it was trained on. Read bottom to top.",
    )
    s.box(
        700,
        1240,
        f"Final-normalized encoder states H ∈ ℝ^(L×{p['hidden']:,})",
        w=1300,
        h=70,
        color="tensor",
        size=26,
    )
    xs = (220, 520, 860)
    labels = (
        ("Option marker hₒ", "[O]"),
        ("Question marker hq", "[Q]"),
        ("State mean p", "[SEG_user] … state"),
    )
    for x, lab, nm in zip(xs, labels, ("Wₒ", "Wq", "Wp"), strict=True):
        s.box(x, 1110, lab, w=270, h=76, color="tensor", size=24)
        conn(s, x, 1240, 1186)
        s.box(x, 995, "LayerNorm*", w=190, h=55, color="norm", size=24)
        conn(s, x, 1110, 1050)
        s.box(
            x,
            885,
            (nm, f"{p['hidden']:,} → {p['proj_dim']}"),
            w=230,
            h=76,
            color="linear",
            size=25,
        )
        conn(s, x, 995, 961)
    s.circle(370, 785, "+", r=23)
    s.path([(220, 885), (220, 785), (347, 785)])
    s.path([(520, 885), (520, 785), (393, 785)])
    s.box(370, 690, "L2 normalize u", w=240, h=60, color="softmax", size=24)
    conn(s, 370, 762, 750)
    s.box(860, 765, "L2 normalize v", w=240, h=60, color="softmax", size=24)
    conn(s, 860, 885, 825)
    s.box(600, 570, "dot(u, v) / τ", w=290, h=74, color="attention", size=27)
    s.path([(370, 690), (370, 668), (530, 668), (530, 644)])
    s.path([(860, 765), (860, 668), (670, 668), (670, 644)])
    s.box(
        170,
        520,
        [f"MLP: {p['hidden']:,} → {2 * p['hidden']:,} → 1", "ReLU · Dropout(0.1)"],
        w=280,
        h=92,
        color="ffn",
        size=23,
    )
    s.path([(220, 1090), (60, 1090), (60, 613)])
    s.dot(220, 1090)
    s.text(70, 1075, "raw hₒ", 20, anchor="start")
    s.circle(420, 420, "+", r=23)
    s.path([(170, 520), (170, 420), (397, 420)])
    s.path([(600, 570), (600, 420), (443, 420)])
    s.box(
        420,
        290,
        ["z: one logit per option", "softmax over the question's options"],
        w=560,
        h=82,
        color="softmax",
        size=24,
    )
    conn(s, 420, 397, 372)
    s.box(
        420,
        150,
        ["Choice: argmax + probabilities", "Noul: 2 options {no, yes}, p(yes)"],
        w=560,
        h=84,
        color="tensor",
        size=24,
    )
    conn(s, 420, 290, 234)
    # training target, dashed
    group(s, 1080, 200, 470, 640, "Training target (not at inference)", dash="9 7")
    s.box(
        1315,
        300,
        ["Soft-target cross-entropy", "on the softmax over options"],
        w=400,
        h=84,
        color="softmax",
        size=22,
    )
    s.box(
        1315,
        450,
        ["Target: d3's answer", "distribution over the options"],
        w=400,
        h=84,
        color="tensor",
        size=22,
    )
    s.box(
        1315,
        600,
        ["With a gold label:", "½ gold + ½ d3"],
        w=400,
        h=84,
        color="tensor",
        size=22,
    )
    s.text(1315, 760, ["rows written in the same", "format as inference"], 22)
    conn(s, 1315, 449, 385)
    conn(s, 1315, 599, 535)
    s.path([(700, 331), (1112, 331)], dash=True, arrow=False)
    s.text(
        800,
        1340,
        "* one shared LayerNorm; τ learned (exp of log τ). Dropout is inactive at inference.",
        22,
    )
    s.save(OUT / "09-vela3-fast-decision-readout.svg")


# =====================================================================================================================
# 10 the span pass on the d3 sizes
# =====================================================================================================================
def span_figure():
    s = Scene(
        1600,
        1300,
        "Vela 3.0 d3 sizes: the span pass",
        "One span question = one forward of the frozen d3 base with its span-family LoRA on, read by a SpanHeadV2. Read bottom to top.",
    )
    s.raw('<g transform="translate(0,-250)">')
    s.box(
        800,
        1230,
        [
            "d3 prompt (chat template): State: <parts> · Question: <span question>",
            "Options: one line per label  <code>: label[: description or definition]",
        ],
        w=1400,
        h=90,
        color="embedding",
        size=24,
    )
    s.box(
        800,
        1090,
        [
            "Frozen d3 decoder, noncausal full attention",
            "span / NER / relation / image-halu LoRA switched on",
        ],
        w=900,
        h=84,
        color="attention",
        size=24,
    )
    conn(s, 800, 1229, 1175)
    s.box(800, 970, "Final-norm states H", w=500, h=60, color="norm", size=25)
    conn(s, 800, 1089, 1031)
    s.box(
        420,
        850,
        ["First sub-token of each", "word of the target part"],
        w=470,
        h=80,
        color="tensor",
        size=24,
    )
    s.box(
        1180,
        850,
        ["Mean over each option line", "label vector, one per label"],
        w=470,
        h=80,
        color="tensor",
        size=24,
    )
    s.path([(800, 969), (800, 940), (420, 940), (420, 931)])
    s.path([(800, 940), (1180, 940), (1180, 931)])
    s.dot(800, 940)
    s.box(
        800,
        700,
        [
            "SpanHeadV2: LayerNorm + centring, 64 label slots",
            "scaled bilinear K Qᵀ / √256 + GELU MLP",
        ],
        w=1000,
        h=90,
        color="linear",
        size=24,
    )
    conn(s, 420, 849, 791)
    conn(s, 1180, 849, 791)
    s.box(
        800,
        560,
        ["Word × label logits", "Z ∈ ℝ^(W×L)"],
        w=440,
        h=78,
        color="tensor",
        size=25,
    )
    conn(s, 800, 699, 639)
    s.box(
        800,
        430,
        "Sigmoid · threshold per label set",
        w=520,
        h=64,
        color="softmax",
        size=25,
    )
    conn(s, 800, 559, 495)
    s.box(
        800,
        310,
        ["Merge neighbouring words · trim", "character-offset spans with scores"],
        w=640,
        h=84,
        color="tensor",
        size=24,
    )
    conn(s, 800, 429, 395)
    s.text(
        800,
        1385,
        [
            "Targets over 2,048 tokens: word-aligned windows ≤ 1,800 tokens, stride 1,536; word logits averaged.",
            "With images, the image tokens enter the same pass in d3's image layout.",
        ],
        22,
    )
    s.panel(800, 1500, "No second copy of the text: every word already sees the labels")
    s.raw("</g>")
    s.save(OUT / "10-d3-span-pass.svg")


def main():
    OUT.mkdir(exist_ok=True)
    fast_figure()
    vision_figure()
    for i, n in enumerate(
        ["vela3-lite", "vela3-mini", "vela3-flash", "vela3-turbo"], 3
    ):
        d3_figure(n, i)
    routing_figure()
    gated_lora_figure()
    readout_figure()
    span_figure()


if __name__ == "__main__":
    main()
