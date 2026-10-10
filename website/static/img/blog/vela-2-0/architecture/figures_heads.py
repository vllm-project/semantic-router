"""Verified Vela 2.0 decoder-family task-head operator diagrams.

Sources and derivation: README.md.
The model revision and public forward are fixed there; no weights are loaded.
Use generate(output_dir) to reproduce both editable SVGs.
"""

# Diagram labels intentionally use Unicode mathematical glyphs.
# ruff: noqa: RUF001

from pathlib import Path

from architecture_svg import Scene

CAPTIONS = {
    "09-decoder-candidate-head": (
        "Vela 2.0 decoder-family CandidateHead. Each option endpoint and the "
        "question-block endpoint use separate LayerNorms. A scaled bilinear "
        "score and an additive GELU MLP score are summed. The same learned head "
        "serves Choice, Noul, Score and Set; Set adds one scalar bias before "
        "temperature scaling and sigmoid. Noul is a two-option Choice; Score "
        "returns the expectation of its ordered distribution. H is 1024, "
        "2560 or 4096 for the 0.8B, 4B and 9B checkpoints; d is 256. These are "
        "independent checkpoints with the same topology, not shared weights."
    ),
    "10-decoder-span-heads": (
        "Vela 2.0 decoder-family SpanHeadV2. Word vectors select the first "
        "subword state of each word in the target repeated after the labels; "
        "label vectors average each label block. "
        "Word LayerNorm is followed by centering over words. Label LayerNorm "
        "is followed by a learned 64-slot embedding and centering over labels "
        "when there are at least two labels. Scaled bilinear and additive "
        "GELU MLP branches produce word-by-label logits. Router and broad "
        "heads have independent parameters and are selected per question. "
        "Calibration and span decoding operate after this selection. The "
        "registered but unused Linear(H,1) module is omitted. H is "
        "1024/2560/4096, d is 256; slots above 63 reuse slot 63. Long targets "
        "use overlapping windows and average logits before sigmoid."
    ),
}


def _projection(s, cx, y, label, bias=False, w=255):
    suffix = "H → 256, bias" if bias else "H → 256, no bias"
    return s.box(cx, y, [label, suffix], w=w, h=76, color="linear", size=25)


def _wire_with_bridge(s, points, cross=None, arrow=True):
    """A small white underlay makes an unconnected line crossing explicit."""
    if cross:
        x, y = cross
        s.path([(x - 9, y), (x + 9, y)], arrow=False, color="#FFF", width=8)
    s.path(points, arrow=arrow)


def candidate_head(output_dir):
    s = Scene(
        1500,
        1330,
        "Vela 2.0 decoder CandidateHead",
        CAPTIONS["09-decoder-candidate-head"],
    )

    # Diagram flows from the shared question hidden states at the bottom up.
    s.box(
        740,
        1180,
        ["Question-block hidden states", "H = 1024 / 2560 / 4096"],
        w=680,
        h=83,
        color="tensor",
        size=27,
    )
    s.box(
        390,
        1060,
        ["Select option endpoint", "cᵢ ∈ ℝᴴ"],
        w=450,
        h=78,
        color="tensor",
        size=26,
    )
    s.box(
        1080,
        1060,
        ["Select block-last endpoint", "q ∈ ℝᴴ"],
        w=450,
        h=78,
        color="tensor",
        size=26,
    )
    s.path([(740, 1180), (740, 1160), (390, 1160), (390, 1138)])
    s.path([(740, 1160), (1080, 1160), (1080, 1138)])
    s.dot(740, 1160)

    s.box(390, 956, "LayerNorm candidate", w=340, h=58, color="norm", size=26)
    s.box(1080, 956, "LayerNorm query", w=340, h=58, color="norm", size=26)
    s.arrow(390, 1060, 390, 1014)
    s.arrow(1080, 1060, 1080, 1014)

    _projection(s, 220, 790, "Key Wₖ")
    _projection(s, 535, 790, "Query Wq")
    _projection(s, 930, 790, "Candidate MLP Wₘ", bias=True, w=290)
    _projection(s, 1250, 790, "Query MLP Wₙ", w=280)

    # Candidate and query are each fanned out to the two independent branches.
    s.path([(390, 956), (390, 926), (220, 926), (220, 866)])
    s.path([(390, 926), (930, 926), (930, 866)])
    s.dot(390, 926)
    s.path([(1080, 956), (1080, 900), (1250, 900), (1250, 866)])
    _wire_with_bridge(s, [(1080, 900), (535, 900), (535, 866)], cross=(930, 900))
    s.dot(1080, 900)

    s.box(
        378,
        598,
        ["Scaled dot product", "(Wₖ ĉᵢ)ᵀ (Wq q̂) / √256"],
        w=525,
        h=82,
        color="attention",
        size=25,
    )
    s.path([(220, 790), (220, 735), (285, 735), (285, 680)])
    s.path([(535, 790), (535, 735), (470, 735), (470, 680)])

    s.circle(1090, 697, "+", r=20)
    s.path([(930, 790), (930, 697), (1070, 697)])
    s.path([(1250, 790), (1250, 697), (1110, 697)])
    s.box(1090, 580, "GELU", w=280, h=60, color="ffn", size=27)
    s.arrow(1090, 677, 1090, 640)
    s.box(
        1090,
        467,
        ["Scalar projection vᵀ", "256 → 1, no bias"],
        w=360,
        h=76,
        color="linear",
        size=26,
    )
    s.arrow(1090, 580, 1090, 543)

    s.circle(740, 383, "+", r=21)
    s.path([(378, 598), (378, 383), (719, 383)])
    s.path([(1090, 467), (1090, 383), (761, 383)])
    s.box(
        740,
        266,
        ["Option logits zᵢ", "one shared head"],
        w=400,
        h=76,
        color="tensor",
        size=27,
    )
    s.arrow(740, 362, 740, 342)

    s.box(
        390,
        100,
        ["Choice / Noul / Score", "Softmax(z / Ttype)"],
        w=640,
        h=88,
        color="softmax",
        size=27,
    )
    s.box(
        1120,
        100,
        ["Set", "Sigmoid((z + bset) / Tset)"],
        w=595,
        h=88,
        color="softmax",
        size=26,
    )
    s.path([(740, 266), (740, 226), (390, 226), (390, 188)])
    s.path([(740, 226), (1120, 226), (1120, 188)])
    s.dot(740, 226)
    s.text(390, 45, "Exclusive probability distribution", 27)
    s.text(1120, 45, "Independent label probabilities", 27)
    s.arrow(390, 100, 390, 62)
    s.arrow(1120, 100, 1120, 62)
    s.panel(750, 1300, "(i) CandidateHead: bilinear + additive MLP")
    return s.save(Path(output_dir) / "09-decoder-candidate-head.svg")


def span_heads(output_dir):
    s = Scene(
        1460,
        1410,
        "Vela 2.0 decoder router and broad SpanHeadV2",
        CAPTIONS["10-decoder-span-heads"],
    )

    # A single expanded topology uses the selected independent parameter set.
    # The dashed parameter link is deliberately separate from activation flow.
    s.rect(35, 305, 1365, 985, "group", r=23, sw=2.2)
    s.text(730, 1270, "Selected SpanHeadV2", 27, serif=True)
    s.box(
        730,
        1166,
        ["Span-question hidden states", "H = 1024 / 2560 / 4096"],
        w=700,
        h=80,
        color="tensor",
        size=27,
    )
    s.box(
        305,
        1050,
        ["First subword per word", "X ∈ ℝᵂˣᴴ"],
        w=475,
        h=78,
        color="tensor",
        size=27,
    )
    s.box(
        1055,
        1050,
        ["Mean per label block", "A ∈ ℝᴸˣᴴ"],
        w=475,
        h=78,
        color="tensor",
        size=27,
    )
    s.path([(730, 1166), (730, 1147), (305, 1147), (305, 1128)])
    s.path([(730, 1147), (1055, 1147), (1055, 1128)])
    s.dot(730, 1147)

    s.box(305, 951, "Word LayerNorm", w=340, h=58, color="norm", size=26)
    s.box(1055, 951, "Label LayerNorm", w=340, h=58, color="norm", size=26)
    s.arrow(305, 1050, 305, 1009)
    s.arrow(1055, 1050, 1055, 1009)
    s.box(
        305,
        857,
        ["Subtract word mean", "over W words"],
        w=340,
        h=70,
        color="norm",
        size=26,
    )
    s.arrow(305, 951, 305, 927)

    s.circle(1055, 888, "+", r=20)
    s.arrow(1055, 951, 1055, 908)
    s.box(
        1268, 854, ["Slot embedding", "64 × H"], w=244, h=68, color="embedding", size=25
    )
    s.arrow(1146, 888, 1075, 888)
    s.box(
        1055,
        776,
        ["Subtract label mean", "only when L ≥ 2"],
        w=340,
        h=70,
        color="norm",
        size=26,
    )
    s.arrow(1055, 868, 1055, 846)

    _projection(s, 194, 636, "Word K", w=262)
    _projection(s, 501, 636, "Label Q", w=262)
    _projection(s, 850, 636, "Word M", bias=True, w=268)
    _projection(s, 1259, 636, "Label N", w=262)
    s.path([(305, 857), (305, 825), (194, 825), (194, 712)])
    s.path([(305, 825), (850, 825), (850, 712)])
    s.dot(305, 825)
    s.path([(1055, 776), (1055, 750), (1259, 750), (1259, 712)])
    _wire_with_bridge(s, [(1055, 750), (501, 750), (501, 712)], cross=(850, 750))
    s.dot(1055, 750)

    s.box(
        347,
        465,
        ["Scaled word × label product", "K(Xc) Q(Ac)ᵀ / √256"],
        w=548,
        h=78,
        color="attention",
        size=25,
    )
    s.path([(194, 636), (194, 590), (255, 590), (255, 543)])
    s.path([(501, 636), (501, 590), (439, 590), (439, 543)])
    s.circle(1105, 563, "+", r=19)
    s.path([(850, 636), (850, 563), (1086, 563)])
    s.path([(1259, 636), (1259, 563), (1124, 563)])
    s.box(1105, 463, "GELU", w=240, h=58, color="ffn", size=27)
    s.arrow(1105, 544, 1105, 521)
    s.box(
        1105,
        371,
        ["Scalar projection vᵀ", "256 → 1"],
        w=335,
        h=76,
        color="linear",
        size=25,
    )
    s.arrow(1105, 463, 1105, 447)

    s.circle(730, 332, "+", r=21)
    s.path([(347, 465), (347, 332), (709, 332)])
    s.path([(1105, 371), (1105, 332), (751, 332)])
    s.box(
        448,
        220,
        ["Word × label logits", "Z ∈ ℝᵂˣᴸ"],
        w=430,
        h=76,
        color="tensor",
        size=27,
    )
    s.path([(730, 311), (730, 309), (448, 309), (448, 296)])
    s.box(448, 123, "Sigmoid(Z / Tselected)", w=430, h=64, color="softmax", size=27)
    s.arrow(448, 220, 448, 187)
    s.box(448, 27, "Threshold + span decoding", w=475, h=64, color="tensor", size=26)
    s.arrow(448, 123, 448, 91)

    # Parameter sources do not carry activation tensors; no router/broad fusion.
    s.box(
        945,
        225,
        ["Router weights θr", "PII / Halu / Toxic"],
        w=282,
        h=70,
        color="linear",
        size=24,
    )
    s.box(
        1262,
        225,
        ["Broad weights θb", "Open extraction"],
        w=282,
        h=70,
        color="linear",
        size=24,
    )
    s.box(
        1105,
        118,
        ["Choose one parameter set", "head override or label dispatch"],
        w=555,
        h=82,
        color="gate",
        size=24,
    )
    s.path([(945, 225), (945, 211), (1020, 211), (1020, 200)], dash=True)
    s.path([(1262, 225), (1262, 211), (1190, 211), (1190, 200)], dash=True)
    s.path(
        [(1382.5, 159), (1430, 159), (1430, 595), (1400, 595)], dash=True, arrow=False
    )
    s.text(1110, 52, "Independent parameters; one head per question", 25)
    s.panel(730, 1370, "(j) SpanHeadV2: word–label scoring with head selection")
    return s.save(Path(output_dir) / "10-decoder-span-heads.svg")


def generate(output_dir):
    """Generate both SVGs and return their absolute output paths."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return [candidate_head(output_dir), span_heads(output_dir)]


if __name__ == "__main__":
    for path in generate(Path(__file__).resolve().parent):
        print(path)
