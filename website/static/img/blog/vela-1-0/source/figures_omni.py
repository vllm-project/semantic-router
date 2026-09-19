"""Detailed public Vela Omni embedding graphs, in Transformer figure grammar.

Release evidence: Nano 0496b39a51c8199592e58cbff81c250f056bd94b;
Mini f7fafd36abf49adf88b1b2ec0186c68b008eeb07. Pinned source URLs:
https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/omni_components/single_modality.py
https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/omni_components/qwen_text_backbone.py
https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/omni_components/mini.py
Transformer internals follow Hugging Face Transformers v4.57.6.
The shared destination contains three independent vectors, not their sum.
Run: python3 source/figures_omni.py [--output-dir PATH]
Default SVG output is the parent figure directory when run from source/.
"""

import argparse
from pathlib import Path

import paper_svg
from paper_svg import Scene

# RUF001 exceptions retain the multiplication glyph used in the artwork.
W = 290
STACK = 900
CENTERS = (350, 900, 1725)


def encoder(s, cx, *, kind, layers, width, heads, inner):
    """Expand one repeated layer, with actual normalization and residual order."""
    top = STACK
    s.repeat(cx - W / 2 - 47, top, W + 120, 550, f"{layers}×")  # noqa: RUF001
    dh = width // heads
    attention = ["Multi-Head Self-Attention", f"{heads} heads · {dh} / head"]
    if kind == "modernbert":
        attention = [
            "RoPE Self-Attention",
            f"{heads} heads · {dh} / head",
            "local ±64 / global",
        ]
        ffn = [
            "GEGLU Feed Forward",
            "GELU(gate) × value",  # noqa: RUF001
            f"{width} → {inner} → {width}",
        ]
    else:
        act = "GELU(tanh)" if kind == "siglip" else "GELU"
        ffn = [
            "Feed Forward",
            f"Linear · {act} · Linear",
            f"{width} → {inner} → {width}",
        ]
    if kind == "bert":
        s.box(cx, top + 38, "Add & Norm", W, 55, "norm", 25)
        s.box(cx, top + 128, ffn, W, 98, "ffn", 21)
        s.box(cx, top + 272, "Add & Norm", W, 55, "norm", 25)
        s.box(cx, top + 381, attention, W, 96, "attention", 22)
        s.arrow(cx, top + 570, cx, top + 512)
        for dx in (-65, 0, 65):
            s.path([(cx, top + 512), (cx + dx, top + 512), (cx + dx, top + 478)])
        s.arrow(cx, top + 380, cx, top + 328)
        s.arrow(cx, top + 271, cx, top + 227)
        s.arrow(cx, top + 127, cx, top + 94)
        s.arrow(cx, top + 37, cx, top - 20, arrow=False)
        side = cx + W / 2 + 44
        s.path(
            [
                (cx, top + 529),
                (side, top + 529),
                (side, top + 300),
                (cx + W / 2 + 1, top + 300),
            ]
        )
        s.path(
            [
                (cx, top + 249),
                (side, top + 249),
                (side, top + 65),
                (cx + W / 2 + 1, top + 65),
            ]
        )
        s.dot(cx, top + 529)
        s.dot(cx, top + 249)
    else:
        s.circle(cx, top + 48)
        s.box(cx, top + 93, ffn, W, 98, "ffn", 21)
        s.box(cx, top + 219, "LayerNorm", W, 48, "norm", 25)
        s.circle(cx, top + 310)
        s.box(cx, top + 354, attention, W, 92, "attention", 22)
        s.box(
            cx,
            top + 475,
            "LayerNorm *" if kind == "modernbert" else "LayerNorm",
            W,
            46,
            "norm",
            25,
        )
        s.arrow(cx, top + 570, cx, top + 522)
        for dx in (-65, 0, 65):
            s.path(
                [
                    (cx, top + 474),
                    (cx, top + 461),
                    (cx + dx, top + 461),
                    (cx + dx, top + 447),
                ]
            )
        s.arrow(cx, top + 353, cx, top + 329)
        s.arrow(cx, top + 291, cx, top + 268)
        s.arrow(cx, top + 218, cx, top + 192)
        s.arrow(cx, top + 92, cx, top + 67)
        s.arrow(cx, top + 29, cx, top - 20, arrow=False)
        side = cx + W / 2 + 45
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


def common_output(s, dimension):
    # Three separate endpoints within one semantic vector space.
    s.rect(170, 115, 2010, 130, "white", r=9)
    s.text(1175, 158, f"Shared {dimension}-dimensional embedding space", 29, serif=True)
    for cx, label in zip(CENTERS, ("text", "image", "audio"), strict=True):
        s.raw(
            f'<text x="{cx}" y="211" font-size="30" text-anchor="middle" class="math">'
            f'z<tspan baseline-shift="sub" font-size="20" font-style="normal">{label}</tspan></text>'
        )
        s.box(cx, 300, "L2 Normalize", W, 58, "softmax", 26)
        s.arrow(cx, 299, cx, 246)


def final_norm(s, cx):
    s.box(cx, 812, "Final LayerNorm", W, 48, "norm", 25)
    s.arrow(cx, STACK - 20, cx, 861)


def text_nano(s):
    cx = CENTERS[0]
    encoder(s, cx, kind="bert", layers=12, width=384, heads=12, inner=1536)
    s.box(cx, 705, ["CLS Readout", "first token · 384"], W, 75, "tensor", 25)
    s.arrow(cx, STACK - 20, cx, 781)
    s.box(cx, 475, ["Identity Projection", "no learned text map"], W, 75, "linear", 22)
    s.arrow(cx, 704, cx, 551)
    s.arrow(cx, 474, cx, 359)
    s.box(cx, 1505, "Embedding LayerNorm", W, 48, "norm", 24)
    s.arrow(cx, 1504, cx, STACK + 571, arrow=False)
    s.circle(cx, 1605)
    s.arrow(cx, 1586, cx, 1554)
    for px, labels in [
        (
            200,
            ["Position", "Embedding", "512 × 384"],  # noqa: RUF001
        ),
        (
            350,
            ["Token", "Embedding", "30,522 × 384"],  # noqa: RUF001
        ),
        (
            500,
            ["Type", "Embedding", "2 × 384"],  # noqa: RUF001
        ),
    ]:
        s.box(px, 1700, labels, 138, 100, "embedding", 20)
    s.path([(200, 1699), (200, 1605), (331, 1605)])
    s.arrow(cx, 1699, cx, 1624)
    s.path([(500, 1699), (500, 1605), (369, 1605)])
    s.arrow(cx, 1945, cx, 1801)
    s.text(cx, 1982, ["Text tokens", "n ≤ 512"], 27)
    s.text(cx, 2070, "GIST-small / BERT", 29, serif=True)


def qwen_encoder(s, cx):
    """Qwen3: causal GQA, per-head Q/K norms, pre-RMSNorm and SwiGLU."""
    top = STACK
    s.repeat(cx - W / 2 - 47, top, W + 120, 550, "28×")  # noqa: RUF001
    s.circle(cx, top + 48)
    s.box(
        cx,
        top + 93,
        [
            "SwiGLU Feed Forward",
            "SiLU(gate) × value",  # noqa: RUF001
            "1024 → 3072 → 1024",
        ],
        W,
        98,
        "ffn",
        21,
    )
    s.box(cx, top + 219, "RMSNorm", W, 48, "norm", 25)
    s.circle(cx, top + 310)
    s.box(
        cx,
        top + 344,
        [
            "Causal GQA + RoPE",
            "16 Q / 8 KV · 128 / head",
            "Q, K: RMSNorm → RoPE",
            "output: 2048 → 1024",
        ],
        W,
        110,
        "attention",
        20,
    )
    s.box(cx, top + 482, "RMSNorm", W, 42, "norm", 25)
    s.arrow(cx, top + 570, cx, top + 525)
    for dx in (-65, 0, 65):
        s.path(
            [
                (cx, top + 481),
                (cx, top + 468),
                (cx + dx, top + 468),
                (cx + dx, top + 455),
            ]
        )
    s.arrow(cx, top + 343, cx, top + 329)
    s.arrow(cx, top + 291, cx, top + 268)
    s.arrow(cx, top + 218, cx, top + 192)
    s.arrow(cx, top + 92, cx, top + 67)
    s.arrow(cx, top + 29, cx, top - 20, arrow=False)
    side = cx + W / 2 + 45
    s.path(
        [(cx, top + 539), (side, top + 539), (side, top + 310), (cx + 20, top + 310)]
    )
    s.path([(cx, top + 280), (side, top + 280), (side, top + 48), (cx + 20, top + 48)])
    s.dot(cx, top + 539)
    s.dot(cx, top + 280)


def text_mini(s):
    cx = CENTERS[0]
    qwen_encoder(s, cx)
    s.box(cx, 812, "Final RMSNorm", W, 48, "norm", 25)
    s.arrow(cx, STACK - 20, cx, 861)
    s.box(
        cx,
        690,
        ["Last Nonpadding Token", "1024-dimensional state"],
        W,
        75,
        "tensor",
        22,
    )
    s.arrow(cx, 811, cx, 766)
    s.box(cx, 565, ["L2 Normalize", "full 1024 dimensions"], W, 75, "softmax", 23)
    s.arrow(cx, 689, cx, 641)
    s.box(cx, 435, ["Matryoshka Prefix", "first 768 dimensions"], W, 75, "tensor", 23)
    s.arrow(cx, 564, cx, 511)
    s.arrow(cx, 434, cx, 359)
    s.box(
        cx,
        1680,
        ["Token Embedding", "151,669 × 1024"],  # noqa: RUF001
        W,
        80,
        "embedding",
        25,
    )
    s.arrow(cx, 1679, cx, STACK + 571, arrow=False)
    s.box(
        cx,
        1830,
        ["Native Tokenizer", "default: shared text", "optional: instruction + text"],
        W,
        95,
        "tensor",
        22,
    )
    s.arrow(cx, 1829, cx, 1761)
    s.arrow(cx, 1945, cx, 1926)
    s.text(cx, 1982, ["Text + optional instruction", "n ≤ 32,768 total tokens"], 25)
    s.text(cx, 2070, "Qwen3-Embedding-0.6B", 27, serif=True)


def image_frontend(s, mini):
    cx = CENTERS[1]
    width, patch, image, tokens = (1152, 14, 384, 729) if mini else (768, 16, 512, 1024)
    s.circle(cx, 1530)
    s.arrow(cx, 1511, cx, STACK + 571, arrow=False)
    s.box(
        cx + 218,
        1484,
        [
            "Learned Position",
            f"{tokens} × {width}",  # noqa: RUF001
        ],
        182,
        92,
        "embedding",
        21,
    )
    s.arrow(cx + 126, 1530, cx + 19, 1530)
    s.box(
        cx,
        1670,
        [
            "Patch Embedding",
            f"Conv2D: {patch} × {patch}, stride {patch}",  # noqa: RUF001
            f"{tokens} tokens × {width}",  # noqa: RUF001
        ],
        W,
        106,
        "embedding",
        22,
    )
    s.arrow(cx, 1669, cx, 1549)
    s.arrow(cx, 1945, cx, 1777)
    s.text(cx, 1982, ["RGB image", f"{image} × {image}"], 27)  # noqa: RUF001
    s.text(cx, 2070, "SigLIP ViT", 29, serif=True)


def map_pool(s, mini=False):
    cx = CENTERS[1]
    width, inner, dim = (1152, 4304, 768) if mini else (768, 3072, 384)
    # Mini v4 normalizes SigLIP's native pooled state before its projection.
    # Q is learned; K,V are all patch states after the final LayerNorm.
    # The learned probe has no residual shortcut into the attention output.
    attn_y, attn_h = (721, 65) if mini else (704, 72)
    norm_y = 652 if mini else 633
    ffn_y, ffn_h = (568, 60) if mini else (541, 68)
    sum_y = 531 if mini else 495
    s.box(
        cx,
        attn_y,
        ["Multi-Head", "Attention Pool"],
        W,
        attn_h,
        "attention",
        23 if mini else 25,
    )
    s.box(
        cx - 265,
        attn_y + (0.5 if mini else 4),
        ["Learned", "probe q"],
        146,
        64,
        "embedding",
        22,
    )
    s.arrow(cx - 191, attn_y + attn_h / 2, cx - W / 2 - 1, attn_y + attn_h / 2)
    branch_y = attn_y + attn_h + 15
    for dx, label in ((-55, "K"), (55, "V")):
        s.path(
            [
                (cx, 811),
                (cx, branch_y),
                (cx + dx, branch_y),
                (cx + dx, attn_y + attn_h + 1),
            ]
        )
        if label == "V":
            s.text(
                cx + 75,
                branch_y + 6,
                label,
                17 if mini else 18,
                anchor="start",
                math=True,
            )
        else:
            s.text(
                cx + dx - 19,
                branch_y + 6,
                label,
                17 if mini else 18,
                anchor="end",
                math=True,
            )
    s.box(cx, norm_y, "LayerNorm", W, 45, "norm", 24)
    s.arrow(cx, attn_y - 1, cx, norm_y + 46)
    s.box(
        cx,
        ffn_y,
        ["Feed Forward", f"{width} → {inner} → {width}"],
        W,
        ffn_h,
        "ffn",
        22 if mini else 23,
    )
    s.arrow(cx, norm_y - 1, cx, ffn_y + ffn_h + 1)
    s.circle(cx, sum_y)
    s.arrow(cx, ffn_y - 1, cx, sum_y + 19)
    residual_y = attn_y - 12
    s.path(
        [(cx, residual_y), (cx + 194, residual_y), (cx + 194, sum_y), (cx + 20, sum_y)]
    )
    s.dot(cx, residual_y)
    if mini:
        s.box(cx, 452, "L2 Normalize", W, 46, "softmax", 24)
        s.arrow(cx, sum_y - 19, cx, 499)
        s.box(cx, 378, ["Linear Projection", f"{width} → {dim}"], W, 60, "linear", 21)
        s.arrow(cx, 451, cx, 439)
    else:
        s.box(cx, 380, ["Linear Projection", f"{width} → {dim}"], W, 70, "linear", 25)
        s.arrow(cx, sum_y - 19, cx, 451)
    s.arrow(cx, 377 if mini else 379, cx, 359)


def image_encoder(s, mini):
    cx = CENTERS[1]
    width, layers, heads, inner = (1152, 27, 16, 4304) if mini else (768, 12, 12, 3072)
    encoder(s, cx, kind="siglip", layers=layers, width=width, heads=heads, inner=inner)
    final_norm(s, cx)
    map_pool(s, mini)
    image_frontend(s, mini)


def whisper_encoder(s, mini):
    cx = 1450
    width, layers, heads, inner, dim = (
        (1024, 24, 16, 4096, 768) if mini else (384, 4, 6, 1536, 384)
    )
    encoder(s, cx, kind="whisper", layers=layers, width=width, heads=heads, inner=inner)
    final_norm(s, cx)
    s.box(cx, 704, ["Mean over Frames", "all 1500 states"], W, 75, "tensor", 25)
    s.arrow(cx, 811, cx, 780)
    s.box(cx, 470, ["Retained Affine", f"{width} → {dim}"], W, 70, "linear", 25)
    s.arrow(cx, 703, cx, 541)
    s.path([(cx, 469), (cx, 408), (1705, 408)])
    s.circle(cx, 1530)
    s.arrow(cx, 1511, cx, STACK + 571, arrow=False)
    s.box(
        cx + 207,
        1484,
        ["Sinusoidal Position", f"1500 × {width}"],  # noqa: RUF001
        188,
        92,
        "embedding",
        19,
    )
    s.arrow(cx + 112, 1530, cx + 19, 1530)
    s.box(
        cx,
        1620,
        ["Conv1D + GELU", f"k = 3, s = 2; {width} → {width}"],
        W,
        75,
        "embedding",
        23,
    )
    s.arrow(cx, 1619, cx, 1549)
    s.box(
        cx,
        1732,
        ["Conv1D + GELU", f"k = 3, s = 1; 80 → {width}"],
        W,
        75,
        "embedding",
        23,
    )
    s.arrow(cx, 1731, cx, 1696)
    s.box(
        cx,
        1844,
        [
            "Log-Mel Spectrogram",
            "80 bins × 3000 frames",  # noqa: RUF001
        ],
        W,
        75,
        "tensor",
        23,
    )
    s.arrow(cx, 1843, cx, 1808)
    s.box(
        cx, 1980, ["Independent Resampling", "PCM → 16 kHz mono"], W, 85, "tensor", 21
    )
    s.arrow(cx, 1979, cx, 1920)
    s.text(
        cx - 45,
        2140,
        "Whisper medium" if mini else "Whisper tiny",
        27,
        anchor="end",
        serif=True,
    )


def clap_block(s):
    """Expanded pre-LN Swin block; stage boxes separately show patch merging."""
    cx, top = 2550, 1160
    s.text(cx, 1100, "Inside each Swin block", 29, serif=True)
    s.repeat(cx - W / 2 - 35, top, W + 105, 550, "")
    s.circle(cx, top + 48)
    s.box(
        cx,
        top + 93,
        ["Feed Forward", "Linear · GELU · Linear", "D → 4D → D"],
        W,
        98,
        "ffn",
        21,
    )
    s.box(cx, top + 219, "LayerNorm", W, 48, "norm", 25)
    s.circle(cx, top + 310)
    s.box(
        cx,
        top + 344,
        [
            "Window Self-Attention",
            "8 × 8; relative position bias",  # noqa: RUF001
            "alternating cyclic shifts",
        ],
        W,
        110,
        "attention",
        20,
    )
    s.box(cx, top + 482, "LayerNorm", W, 42, "norm", 25)
    s.arrow(cx, top + 570, cx, top + 525)
    for dx in (-65, 0, 65):
        s.path(
            [
                (cx, top + 481),
                (cx, top + 468),
                (cx + dx, top + 468),
                (cx + dx, top + 455),
            ]
        )
    s.arrow(cx, top + 343, cx, top + 329)
    s.arrow(cx, top + 291, cx, top + 268)
    s.arrow(cx, top + 218, cx, top + 192)
    s.arrow(cx, top + 92, cx, top + 67)
    s.arrow(cx, top + 29, cx, top - 20, arrow=False)
    side = cx + W / 2 + 45
    s.path(
        [(cx, top + 539), (side, top + 539), (side, top + 310), (cx + 20, top + 310)]
    )
    s.path([(cx, top + 280), (side, top + 280), (side, top + 48), (cx + 20, top + 48)])
    s.dot(cx, top + 539)
    s.dot(cx, top + 280)
    s.text(
        cx,
        1790,
        [
            "D = 96, 192, 384, 768",
            "24 dimensions per head",
            "No shift when grid ≤ window",
        ],
        22,
    )
    s.text(
        cx,
        1900,
        [
            "Between stages 1–3:",  # noqa: RUF001
            "2 × 2 patch concatenation",  # noqa: RUF001
            "LayerNorm → Linear 4D → 2D",
        ],
        22,
    )


def clap_encoder(s, mini):
    cx, dim = 2000, 768 if mini else 384
    s.box(
        cx, 470, ["Learned Residual Map", f"bias-free 512 → {dim}"], W, 70, "linear", 23
    )
    s.path([(cx, 469), (cx, 408), (1745, 408)])
    s.box(
        cx,
        580,
        [
            "Frozen TRAIN Statistics",
            "(c − mean) / scale",  # noqa: RUF001
        ],
        W,
        75,
        "norm",
        23,
    )
    s.arrow(cx, 579, cx, 541)
    s.box(
        cx,
        690,
        ["Window Aggregation", "L2 each → mean → L2", "one window: its unit vector"],
        W,
        95,
        "softmax",
        21,
    )
    s.arrow(cx, 689, cx, 656)
    s.box(cx, 825, ["CLAP Projection", "768 → 512 → 512; ReLU"], W, 80, "linear", 22)
    s.arrow(cx, 824, cx, 786)
    s.box(cx, 945, ["Global Average Pool", "768 dimensions"], W, 65, "tensor", 23)
    s.arrow(cx, 944, cx, 906)
    s.box(cx, 1040, "Final LayerNorm", W, 48, "norm", 24)
    s.arrow(cx, 1039, cx, 1011)
    stages = [(4, 2, 768, 32), (3, 6, 384, 16), (2, 2, 192, 8), (1, 2, 96, 4)]
    for i, (stage, depth, width, heads) in enumerate(stages):
        y = 1130 + i * 140
        lines = [
            f"Stage {stage}: {depth} Swin blocks",
            f"width {width} · {heads} heads",
        ]
        lines += (
            [f"patch merge: {4*width} → {2*width}"]
            if stage < len(stages)
            else ["final 8 × 8 grid"]  # noqa: RUF001
        )
        s.box(cx, y, lines, W, 100, "attention", 20)
        s.arrow(cx, y - 1, cx, 1089 if i == 0 else y - 39)
    s.box(
        cx,
        1730,
        [
            "Patch Embedding + LN",
            "Conv2D 4 × 4, stride 4",  # noqa: RUF001
            "64 × 64 grid · width 96",  # noqa: RUF001
        ],
        W,
        95,
        "embedding",
        21,
    )
    s.arrow(cx, 1729, cx, 1651)
    s.box(
        cx,
        1870,
        [
            "Log-Mel → BatchNorm",
            "64 mel bins",
            "resize + reshape 256 × 256",  # noqa: RUF001
        ],
        W,
        95,
        "tensor",
        20,
    )
    s.arrow(cx, 1869, cx, 1826)
    s.box(
        cx,
        2010,
        [
            "Endpoint-spaced Windows",
            "≤ 10 s each; repeat-pad short",
            "1–3 windows per waveform",  # noqa: RUF001
        ],
        W,
        95,
        "tensor",
        20,
    )
    s.arrow(cx, 2009, cx, 1966)
    s.box(
        cx, 2180, ["Independent Resampling", "PCM → 48 kHz mono"], W, 85, "tensor", 21
    )
    s.arrow(cx, 2179, cx, 2106)
    s.text(cx + 210, 2220, "Frozen CLAP / HTS-AT", 27, anchor="start", serif=True)
    clap_block(s)


def audio_encoder(s, mini):
    whisper_encoder(s, mini)
    clap_encoder(s, mini)
    s.circle(1725, 408)
    s.arrow(1725, 389, 1725, 359)
    s.text(1725, 459, "speech affine + CLAP residual", 21)
    s.box(
        1725,
        2370,
        [
            "Original-rate PCM · ≤ 30 s",
            "mono or channels-first; preserve original bandwidth",
        ],
        880,
        92,
        "tensor",
        24,
    )
    s.path([(1725, 2369), (1725, 2310), (1450, 2310), (1450, 2066)])
    s.path([(1725, 2310), (2000, 2310), (2000, 2266)])
    s.dot(1725, 2310)


def draw(mini=False):
    variant = "Mini" if mini else "Nano"
    dimension = 768 if mini else 384
    s = Scene(
        2850,
        2530,
        f"Vela Omni {variant}: three modality paths",
        "Bottom-to-top public encode_text, encode_image and encode_audio graphs. "
        "Expanded Transformer layers show real normalization order, residuals, attention and feed-forward widths. "
        "The common vector space contains three independently computed vectors. Audio combines the retained unnormalized Whisper affine with a learned residual of frozen CLAP features, followed by L2 normalization. Both audio branches independently resample original PCM. "
        "Dropout, inactive during inference, is omitted. "
        + (
            "Mini vision uses native learned-probe attention pooling over all 729 patch states, "
            "then L2 normalization, linear projection to 768, and final L2 normalization. "
            "Mini text uses Qwen3-Embedding-0.6B, causal grouped-query attention with 16 query "
            "heads and 8 key/value heads of width 128, per-head Q/K RMSNorm, RoPE theta 1000000, "
            "pre-RMSNorm and SwiGLU. Its last nonpadding token is normalized over all 1024 dimensions, "
            "then truncated to the first 768, then normalized again. No learned text projection is used."
            if mini
            else "Nano vision uses learned-probe attention pooling followed by a pre-normalized GELU-tanh residual MLP. "
            "The frozen GIST-small text tower has twelve BERT layers and uses CLS readout with identity projection. "
            "The single-modality package has no BERT pooler, multimodal fusion, or intermediate exit heads."
        ),
    )
    s.text(1175, 64, f"Vela Omni {variant}", 39, serif=True)
    s.text(
        1175, 96, "1.36B total parameters" if mini else "163.8M total parameters", 22
    )
    common_output(s, dimension)
    (text_mini if mini else text_nano)(s)
    image_encoder(s, mini)
    audio_encoder(s, mini)
    if mini:
        s.text(
            625,
            2220,
            "Text: causal attention; RoPE θ = 1,000,000; no language-model head.",
            21,
        )
    return s.save("08-omni-mini" if mini else "07-omni-nano")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir.parent if script_dir.name == "source" else script_dir,
    )
    args = parser.parse_args()
    paper_svg.ROOT = args.output_dir.resolve()
    paper_svg.ROOT.mkdir(parents=True, exist_ok=True)
    for mini in (False, True):
        print(draw(mini))


if __name__ == "__main__":
    main()
