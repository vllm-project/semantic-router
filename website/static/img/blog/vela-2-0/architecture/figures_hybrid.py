"""Editable Vela 2.0 hybrid token-mixer details, pinned to exported public code.

Uses architecture_svg.Scene without modifying the shared primitives. Dimensions
come from the independent 0.8B / 4B / 9B checkpoints; the figures share operator
topology, not weights. Logical operator order is verified at Transformers 5.17.0
and the checkpoint revisions recorded in the research evidence report.
"""

# Diagram labels intentionally use Unicode mathematical glyphs.
# ruff: noqa: RUF001

from pathlib import Path

from architecture_svg import Scene


def _param_table(s, y, headers, rows, centers):
    """Plain large-type table, without an enclosing decorative card."""
    s.path([(70, y - 28), (1530, y - 28)], arrow=False, width=1.7)
    for x, name in zip(centers, headers, strict=True):
        s.text(x, y, name, size=25)
    s.path([(70, y + 15), (1530, y + 15)], arrow=False, width=1.3)
    for i, row in enumerate(rows):
        for x, value in zip(centers, row, strict=True):
            s.text(x, y + 53 + 38 * i, value, size=25)
    s.path([(70, y + 145), (1530, y + 145)], arrow=False, width=1.7)


def gated_deltanet(output_dir):
    s = Scene(
        1600,
        1840,
        "Vela 2.0 Gated-DeltaNet: exact operator detail",
        "Bottom-to-top activation flow: pre-normalized hidden states; fused QKV "
        "projection, causal depthwise convolution and SiLU, Q/K head repetition "
        "and L2 normalization, chunk gated delta update with independent prefix "
        "state, per-value-head RMSNorm before SiLU output gating, and output "
        "projection. The three model sizes have independent parameters.",
    )

    # Main output and its norm-before-gate branch.
    s.text(480, 120, "Mixer output  ·  L × d", size=27, math=True)
    s.box(480, 165, "Output Linear: Dv → d", w=390, h=68, size=26)
    s.arrow(480, 165, 480, 135)
    s.box(480, 305, "Concatenate V heads", w=390, h=64, color="tensor", size=26)
    s.arrow(480, 305, 480, 233)
    s.circle(480, 425, "⊙", r=21, size=31)
    s.arrow(480, 404, 480, 369)
    s.box(480, 505, "RMSNorm: 128 / V head", w=390, h=65, color="norm", size=26)
    s.arrow(480, 505, 480, 446)
    s.box(
        480,
        635,
        (
            "Chunk Gated Delta Rule",
            "decay g · update β",
            "prefix: S₀ = 0; question: S₀ = Sₚ",
        ),
        w=660,
        h=116,
        color="attention",
        size=25,
    )
    s.arrow(480, 635, 480, 570)

    # The equivalent recurrence is an annotation, not another execution path.
    s.text(1180, 122, "Per-head Delta state", size=27)
    s.text(
        1180,
        165,
        (
            "Dₜ = exp(gₜ) Sₜ₋₁",
            "δₜ = βₜ (vₜ − Dₜᵀ kₜ)",
            "Sₜ = Dₜ + kₜ δₜᵀ",
            "oₜ = Sₜᵀ q̄ₜ",
        ),
        size=27,
        math=True,
    )
    s.text(1180, 315, "q̄ = L2Norm(q) / √128", size=25, math=True)
    s.text(1180, 350, "State per V head: 128 × 128", size=25)
    s.text(1180, 385, "Prefix starts at zero; questions fork Sₚ", size=24)

    # Fused QKV trunk, read bottom-to-top.
    s.box(330, 1270, ("Fused QKV Linear", "d → 2Dk + Dv"), w=440, h=82, size=26)
    s.box(
        330,
        1145,
        ("Causal depthwise Conv1D", "k = 4 · prefix tail + own block"),
        w=510,
        h=84,
        color="attention",
        size=25,
    )
    s.arrow(330, 1270, 330, 1229)
    s.box(330, 1048, "SiLU", w=290, h=58, color="gate", size=27)
    s.arrow(330, 1145, 330, 1106)
    s.box(330, 945, "Split Q / K / V into heads", w=510, h=65, color="tensor", size=25)
    s.arrow(330, 1048, 330, 1010)
    s.box(245, 870, "Repeat Q/K heads 1× or 2×", w=345, h=58, color="tensor", size=24)
    s.box(
        245,
        770,
        ("L2Norm(Q, K)", "Q scale: 1 / √128"),
        w=330,
        h=80,
        color="norm",
        size=25,
    )
    # Distinct Q/K and V outputs from the split; V bypasses Q/K norm.
    s.path([(245, 945), (245, 928)])
    s.path([(510, 945), (510, 780), (630, 780), (630, 751)])
    s.text(542, 858, "V", size=26, math=True)
    s.arrow(245, 870, 245, 850)
    s.arrow(245, 770, 245, 751)

    # Decay branch: Linear a -> additive learned time bias -> softplus ->
    # multiplication by a learned negative rate. Update branch is separate.
    s.box(800, 1270, "Linear a: d → hv", w=290, h=82, size=26)
    s.circle(800, 1214, "+", r=20, size=30)
    s.arrow(800, 1270, 800, 1234)
    s.box(680, 1182, "dt bias", w=160, h=63, color="tensor", size=25)
    s.arrow(760, 1214, 780, 1214)
    s.box(800, 1085, "Softplus", w=290, h=65, color="softmax", size=26)
    s.arrow(800, 1194, 800, 1150)
    s.circle(800, 1007, "⊙", r=20, size=30)
    s.arrow(800, 1085, 800, 1027)
    s.box(1035, 975, "−exp(A log)", w=280, h=64, color="tensor", size=26)
    s.arrow(895, 1007, 820, 1007)
    s.path([(800, 987), (850, 987), (850, 725), (810, 725)])
    s.text(884, 825, "g", size=27, math=True)

    s.box(1120, 1270, "Linear b: d → hv", w=285, h=82, size=26)
    s.box(1120, 1135, "Sigmoid", w=245, h=66, color="softmax", size=26)
    s.arrow(1120, 1270, 1120, 1201)
    s.path([(1120, 1135), (1120, 1095), (1238, 1095), (1238, 681), (810, 681)])
    s.text(1270, 860, "β", size=27, math=True)

    # Output gate is projected directly from H and applied AFTER the norm.
    s.box(1430, 1270, "Linear z: d → Dv", w=270, h=82, size=26)
    s.box(1430, 1090, "SiLU(z)", w=255, h=66, color="gate", size=27)
    s.arrow(1430, 1270, 1430, 1156)
    s.path([(1430, 1090), (1430, 425), (501, 425)])
    s.text(1458, 845, "z gate", size=25, anchor="start")

    # One input, with separate learned projections. Fan-out is explicit.
    input_x = 800
    s.box(
        input_x, 1450, "H = RMSNorm(X)  ·  L × d", w=1130, h=70, color="tensor", size=27
    )
    s.arrow(input_x, 1450, input_x, 1390, arrow=False)
    s.path([(330, 1390), (1430, 1390)], arrow=False)
    s.dot(input_x, 1390)
    for x in (330, input_x, 1120, 1430):
        s.arrow(x, 1390, x, 1352)
        if x != input_x:
            s.dot(x, 1390)

    _param_table(
        s,
        1570,
        ["Model", "d", "Q/K heads / V heads", "Q/K repetition", "Dv"],
        [
            ("0.8B", "1024", "16 / 16", "1×", "2048"),
            ("4B", "2560", "16 / 32", "2×", "4096"),
            ("9B", "4096", "16 / 32", "2×", "4096"),
        ],
        [160, 430, 740, 1115, 1420],
    )
    s.text(
        800,
        1747,
        "Dk = 2048; dk = dv = 128. Independent checkpoints; shared operator topology.",
        size=24,
    )
    s.panel(
        800, 1800, "(g) Gated-DeltaNet: convolution, delta update and gated readout"
    )
    return s.save(Path(output_dir) / "07-gated-deltanet.svg")


def gated_gqa_swiglu(output_dir):
    s = Scene(
        1600,
        1840,
        "Vela 2.0 gated causal GQA and SwiGLU: exact operator detail",
        "Bottom-to-top GQA with a joint query/gate projection, Q/K RMSNorm, "
        "partial 64-of-256-dimension RoPE, four-way KV head repetition, scaled "
        "dot product, causal visibility mask, softmax, value sum, concatenation "
        "and sigmoid output gating. A separate SwiGLU panel expands its two "
        "parallel projections, SiLU, elementwise product and down projection.",
    )

    # GQA output, with its sigmoid gate separate from normalized Q.
    s.text(550, 120, "Mixer output  ·  L × d", size=27, math=True)
    s.box(550, 165, "Output Linear: hq × 256 → d", w=480, h=68, size=26)
    s.arrow(550, 165, 550, 136)
    s.circle(550, 305, "⊙", r=21, size=31)
    s.arrow(550, 284, 550, 233)
    s.box(550, 400, "Concatenate query heads", w=410, h=66, color="tensor", size=26)
    s.arrow(550, 400, 550, 326)
    s.box(550, 510, "Attention weights × V", w=435, h=66, color="attention", size=26)
    s.arrow(550, 510, 550, 466)
    s.box(
        450, 620, "Softmax over visible tokens", w=515, h=66, color="softmax", size=26
    )
    s.path([(450, 620), (450, 595), (550, 595), (550, 576)])
    s.box(
        450,
        728,
        ("Add causal visibility mask", "shared prefix + own question"),
        w=555,
        h=83,
        color="tensor",
        size=25,
    )
    s.arrow(450, 728, 450, 686)
    s.box(450, 850, "QKᵀ / √256", w=515, h=66, color="attention", size=27)
    s.arrow(450, 850, 450, 811)

    # Q+gate are one projection, unlike the separate K and V projections.
    s.box(280, 1270, ("Linear, split Q / G", "d → 2 hq × 256"), w=385, h=88, size=25)
    s.box(620, 1270, ("Linear K", "d → hkv × 256"), w=240, h=88, size=25)
    s.box(890, 1270, ("Linear V", "d → hkv × 256"), w=240, h=88, size=25)
    s.box(280, 1150, "RMSNorm: Q, 256 / head", w=320, h=65, color="norm", size=25)
    s.box(620, 1150, "RMSNorm: K, 256 / head", w=320, h=65, color="norm", size=25)
    s.path([(330, 1270), (330, 1240), (280, 1240), (280, 1215)])
    s.arrow(620, 1270, 620, 1215)
    s.box(
        280,
        1040,
        ("Partial RoPE: Q", "first 64 / 256 dimensions"),
        w=320,
        h=82,
        color="attention",
        size=25,
    )
    s.box(
        620,
        1040,
        ("Partial RoPE: K", "first 64 / 256 dimensions"),
        w=320,
        h=82,
        color="attention",
        size=25,
    )
    s.arrow(280, 1150, 280, 1122)
    s.arrow(620, 1150, 620, 1122)
    s.box(620, 940, "Repeat K heads 4×", w=275, h=64, color="tensor", size=25)
    s.box(905, 940, "Repeat V heads 4×", w=275, h=64, color="tensor", size=25)
    s.arrow(620, 1040, 620, 1004)
    s.path([(890, 1270), (890, 1025), (905, 1025), (905, 1004)])
    s.path([(280, 1040), (280, 975), (345, 975), (345, 916)])
    s.path([(620, 940), (620, 927), (560, 927), (560, 916)])
    # The V path is outside the QK -> mask -> softmax stack.
    s.path([(905, 940), (905, 920), (980, 920), (980, 543), (767.5, 543)])
    s.text(1005, 745, "V", size=27, math=True)

    s.box(130, 368, "Sigmoid(G)", w=210, h=68, color="gate", size=25)
    s.path([(150, 1270), (150, 1240), (55, 1240), (55, 458), (130, 458), (130, 436)])
    s.path([(130, 368), (130, 305), (529, 305)])
    s.text(81, 855, "G", size=27, math=True)

    # Pre-normalized input and fan-out.
    s.box(550, 1450, "H = RMSNorm(X)  ·  L × d", w=940, h=70, color="tensor", size=26)
    s.arrow(550, 1450, 550, 1400, arrow=False)
    s.path([(280, 1400), (890, 1400)], arrow=False)
    s.dot(550, 1400)
    for x in (280, 620, 890):
        s.dot(x, 1400)
        s.arrow(x, 1400, x, 1358)

    # SwiGLU, with a separate input stream because it is the next pre-norm
    # sublayer, not a parallel token-mixer branch.
    s.path([(1075, 220), (1075, 1520)], arrow=False, color="#B7B7B7", width=1.4)
    s.text(1340, 730, "FFN output  ·  L × d", size=27, math=True)
    s.box(1340, 790, "Down Linear: dff → d", w=420, h=75, color="ffn", size=26)
    s.arrow(1340, 790, 1340, 745)
    s.circle(1340, 985, "⊙", r=21, size=31)
    s.arrow(1340, 964, 1340, 865)
    s.box(1200, 1090, "SiLU", w=220, h=66, color="gate", size=27)
    s.path([(1200, 1090), (1200, 985), (1319, 985)])
    s.path([(1470, 1270), (1470, 985), (1361, 985)])
    s.box(1200, 1270, ("Gate Linear", "d → dff"), w=240, h=88, color="ffn", size=25)
    s.box(1470, 1270, ("Up Linear", "d → dff"), w=230, h=88, color="ffn", size=25)
    s.arrow(1200, 1270, 1200, 1156)
    s.box(
        1340,
        1450,
        ("U = RMSNorm(Y)", "Y = mixer residual add"),
        w=480,
        h=70,
        color="tensor",
        size=25,
    )
    s.arrow(1340, 1450, 1340, 1400, arrow=False)
    s.path([(1200, 1400), (1470, 1400)], arrow=False)
    s.dot(1340, 1400)
    for x in (1200, 1470):
        s.dot(x, 1400)
        s.arrow(x, 1400, x, 1358)

    _param_table(
        s,
        1570,
        ["Model", "d", "Query / KV heads", "Head dimension", "dff"],
        [
            ("0.8B", "1024", "8 / 2", "256", "3584"),
            ("4B", "2560", "16 / 4", "256", "9216"),
            ("9B", "4096", "16 / 4", "256", "12288"),
        ],
        [160, 425, 750, 1120, 1420],
    )
    s.text(
        800,
        1747,
        "RoPE θ = 10⁷; RMSNorm ε = 10⁻⁶. Independent checkpoints; shared operator topology.",
        size=24,
    )
    s.panel(800, 1800, "(h) Gated causal grouped-query attention and SwiGLU")
    return s.save(Path(output_dir) / "08-gated-gqa-swiglu.svg")


def generate(output_dir):
    """Generate the two SVG details; callers own rendering and visual QA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [gated_deltanet(output_dir), gated_gqa_swiglu(output_dir)]


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
