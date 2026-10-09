"""Editable system diagrams: verified request flow, never neural internals."""

from __future__ import annotations

import base64
import html
from pathlib import Path

INK = "#211b35"
MUTED = "#686477"
PURPLE = "#7051bb"
PURPLE_PALE = "#f0ebfa"
GOLD = "#bc711a"
GOLD_PALE = "#fff3df"
GREEN = "#27775d"


def text(x, y, value, size=28, color=INK, weight=400, anchor="start", box=None):
    bounds = f' data-box="{box}"' if box else ""
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-family="DejaVu Sans,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{bounds}>'
        f"{html.escape(value)}</text>"
    )


def rect(x, y, w, h, fill="white", stroke="#ddd7e7", radius=16, extra=""):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2" {extra}/>'
    )


def arrow(path, color=INK, width=3, extra="", head=True):
    marker = ' marker-end="url(#arrow)"' if head else ""
    return (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round"{marker} {extra}/>'
    )


def start(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="context-stroke"/></marker></defs>',
    ]


def badge(x, y, width, label, color=PURPLE, fill=PURPLE_PALE):
    return rect(x, y, width, 42, fill, fill, 21) + text(
        x + width / 2, y + 29, label, 21, color, 600, "middle", f"{x} {y} {width} 42"
    )


def banner(logo: Path) -> str:
    encoded = base64.b64encode(logo.read_bytes()).decode()
    parts = start(1920, 1080)
    parts += [
        '<defs><linearGradient id="night" x2=".95" y2="1"><stop stop-color="#171226"/><stop offset=".65" stop-color="#30203f"/><stop offset="1" stop-color="#513548"/></linearGradient><radialGradient id="glow"><stop stop-color="#eeb16c" stop-opacity=".27"/><stop offset="1" stop-color="#d19168" stop-opacity="0"/></radialGradient><filter id="white-logo"><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 1 0"/></filter></defs>',
        '<rect width="1920" height="1080" fill="url(#night)"/>',
        '<ellipse cx="1670" cy="975" rx="850" ry="700" fill="url(#glow)"/>',
        '<path d="M1070 -130 C1720 85 1240 330 1920 610 M1190 -165 C1870 45 1370 390 2080 645" fill="none" stroke="#a48dc5" stroke-width="2" opacity=".14"/>',
        f'<image x="115" y="64" width="350" height="145" xlink:href="data:image/png;base64,{encoded}" filter="url(#white-logo)"/>',
        text(1795, 150, "SYSTEM ONE AUTO", 28, "#d0bfdc", 500, "end"),
        text(120, 365, "DECISION MODELS.", 118, "#ffffff", 700),
        text(120, 510, "ONE AUTO ROUTER.", 118, "#f1b665", 700),
        text(
            126, 600, "Start small. Escalate when the answer needs more.", 39, "#e4d9ed"
        ),
        rect(125, 698, 1670, 245, "#241c32", "#685372", 24),
        rect(169, 743, 337, 146, "#433148", "#ad8158", 14),
        text(202, 799, "Kai", 39, "#f8d09b", 700),
        text(202, 849, "0.6B · first answer", 24, "#e1d2e2"),
        arrow("M506 816 H606", "#e9c18c", 4),
        text(760, 799, "Accept or upgrade", 30, "#ffffff", 600, "middle"),
        text(760, 844, "Check answer probabilities", 22, "#cdbcd9", 400, "middle"),
        arrow("M914 816 H1010", "#b9a1da", 4),
        rect(1018, 743, 337, 146, "#342942", "#9b80bd", 14),
        text(1051, 799, "Vega", 39, "#d6c0f5", 700),
        text(1051, 849, "27B · when needed", 24, "#e1d2e2"),
        '<path d="M1405 750 V887" stroke="#64526f" stroke-width="2"/>',
        text(1460, 792, "CHOICE", 23, "#ffffff", 600),
        text(1460, 835, "SCORE", 23, "#ffffff", 600),
        text(1460, 878, "NOUL", 23, "#ffffff", 600),
        text(128, 1007, "DECISION 2.0", 22, "#bdabc9", 500),
        text(1794, 1007, "Same questions. One native API.", 25, "#e8d4b7", 500, "end"),
        "</svg>",
    ]
    return "\n".join(parts)


def cascade() -> str:
    parts = start(1800, 1010)
    parts += [
        '<rect width="1800" height="1010" fill="white"/>',
        text(65, 82, "Small first. Upgrade only when needed.", 46, weight=700),
        text(
            65,
            132,
            "Kai answers the original questions. The cascade checks those answers.",
            27,
            MUTED,
        ),
        rect(398, 189, 1337, 710, "#faf9fc", "#ded9e7", 22),
        badge(434, 215, 254, "vllm-sr/auto"),
        text(1700, 245, "SELECTED CASCADE", 21, MUTED, 600, "end"),
        rect(65, 388, 270, 190),
        text(200, 438, "Your request", 29, weight=700, anchor="middle"),
        text(200, 481, "Input + questions", 23, MUTED, anchor="middle"),
        badge(88, 511, 71, "C", INK, "#eeedf0"),
        badge(165, 511, 71, "S", INK, "#eeedf0"),
        badge(242, 511, 71, "N", INK, "#eeedf0"),
        arrow("M335 480 H443"),
        rect(450, 365, 350, 232, GOLD_PALE, "#dca04e"),
        text(480, 409, "01  FIRST ANSWER", 21, GOLD, 600),
        text(480, 463, "Kai 0.6B", 38, GOLD, 700),
        text(480, 504, "Original input + questions", 23),
        text(480, 552, "Answers + probabilities", 23, MUTED),
        arrow("M800 480 H886"),
        '<path d="M1030 363 L1170 480 L1030 597 L890 480Z" fill="#f0ebfa" stroke="#7051bb" stroke-width="2.5"/>',
        text(1030, 452, "Check", 27, PURPLE, 600, "middle"),
        text(1030, 489, "probabilities", 25, PURPLE, 600, "middle"),
        text(1030, 520, "against thresholds", 18, MUTED, 400, "middle"),
        arrow("M1030 363 V306 H1545 V367", GREEN),
        text(1288, 290, "PASS · return Kai", 24, GREEN, 600, "middle"),
        rect(1370, 372, 350, 218, "white", "#8f829f"),
        text(1545, 425, "Native response", 30, weight=700, anchor="middle"),
        text(1545, 469, "Same answer schema", 23, MUTED, anchor="middle"),
        badge(1387, 511, 103, "choice"),
        badge(1495, 511, 103, "score"),
        badge(1603, 511, 99, "noul"),
        arrow("M1030 597 V674", PURPLE),
        text(1061, 641, "OTHERWISE · upgrade", 23, PURPLE, 600),
        rect(852, 681, 356, 175, PURPLE_PALE, "#a68dcb"),
        text(883, 722, "02  STRONGER ANSWER", 21, PURPLE, 600),
        text(883, 774, "Vega 27B", 37, PURPLE, 700),
        text(883, 819, "Same original request", 24),
        arrow("M1208 770 H1545 V596", PURPLE),
        text(1252, 745, "Valid response", 23, PURPLE, 600),
        text(480, 649, "One model call so far", 23, GOLD),
        text(65, 944, "C = choice   S = score   N = noul", 22, MUTED),
        text(
            1735,
            944,
            "Up to 2 model calls · no valid final answer → unresolved",
            22,
            MUTED,
            anchor="end",
        ),
        "</svg>",
    ]
    return "\n".join(parts)


def ecosystem() -> str:
    parts = start(1800, 1160)
    parts += [
        '<rect width="1800" height="1160" fill="white"/>',
        text(65, 82, "Decision models need a router, too.", 46, weight=700),
        text(
            65,
            132,
            "The same orchestration idea. A different kind of model and answer.",
            27,
            MUTED,
        ),
        rect(60, 188, 810, 809, "#faf9fc", "#ded9e7", 22),
        rect(930, 188, 810, 809, "#fffcf7", "#e5d4b8", 22),
        text(100, 243, "01", 30, MUTED, 700),
        text(161, 243, "LLM ROUTING", 28, INK, 700),
        text(970, 243, "02", 30, GOLD, 700),
        text(1031, 243, "DECISION-MODEL ROUTING", 28, GOLD, 700),
        rect(240, 284, 450, 105),
        text(465, 326, "Messages", 30, weight=600, anchor="middle"),
        text(465, 363, "Chat Completions / Responses", 22, MUTED, anchor="middle"),
        rect(1110, 284, 450, 105),
        text(1335, 326, "Input + typed questions", 29, weight=600, anchor="middle"),
        text(1335, 363, "System One API", 23, MUTED, anchor="middle"),
        arrow("M465 389 V445"),
        arrow("M1335 389 V445", GOLD),
        rect(240, 451, 450, 116, PURPLE_PALE, "#a48dc5"),
        text(465, 498, "vLLM Semantic Router", 28, PURPLE, 700, "middle"),
        text(465, 540, "Select an LLM backend", 24, MUTED, anchor="middle"),
        rect(1110, 451, 450, 116, GOLD_PALE, "#dca04e"),
        text(1335, 498, "System One Auto", 30, GOLD, 700, "middle"),
        text(1335, 540, "Select a decision / cascade", 24, MUTED, anchor="middle"),
    ]
    parts += [
        arrow("M465 567 V615 M215 615 H715", "#8b7f9c", head=False),
        arrow("M215 833 H715", "#8b7f9c", head=False),
        arrow("M465 833 V863", "#8b7f9c"),
        arrow("M1335 567 V615 M1130 615 H1540", "#8b7f9c", head=False),
        arrow("M1130 833 H1540", "#8b7f9c", head=False),
        arrow("M1335 833 V863", "#8b7f9c"),
    ]
    for x, name in ((215, "Small LLM"), (465, "Mid-size LLM"), (715, "Large LLM")):
        parts += [
            arrow(f"M{x} 615 V674", "#8b7f9c"),
            rect(x - 107, 680, 214, 99),
            text(x, 737, name, 25, weight=600, anchor="middle"),
            arrow(f"M{x} 779 V833", "#8b7f9c", head=False),
        ]
    # Aligned stages are alternatives or authored cascade stages, not shared weights.
    for x, name, scale, fill, color in (
        (1130, "Kai", "0.6B", GOLD_PALE, GOLD),
        (1540, "Vega", "27B", PURPLE_PALE, PURPLE),
    ):
        parts += [
            arrow(f"M{x} 615 V674", color),
            rect(x - 128, 680, 256, 99, fill, color),
            text(x - 95, 738, name, 29, color, 700),
            text(x + 94, 738, scale, 26, color, 400, "end"),
            arrow(f"M{x} 779 V833", color, head=False),
        ]
    parts += [
        text(1335, 659, "DECISION 2.0", 21, GOLD, 600, "middle"),
        rect(240, 869, 450, 86),
        text(465, 922, "Generated text", 29, weight=600, anchor="middle"),
        rect(1110, 869, 450, 86),
        badge(1134, 891, 123, "choice"),
        badge(1273, 891, 123, "score"),
        badge(1412, 891, 123, "noul"),
        text(465, 1050, "Models generate language.", 25, MUTED, anchor="middle"),
        text(
            1335,
            1050,
            "Models answer structured questions.",
            25,
            MUTED,
            anchor="middle",
        ),
        text(
            900,
            1123,
            "Measured here: Kai → Vega. Broader provider evaluation is next.",
            24,
            MUTED,
            anchor="middle",
        ),
        "</svg>",
    ]
    return "\n".join(parts)
