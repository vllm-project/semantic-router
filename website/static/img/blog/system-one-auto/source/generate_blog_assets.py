"""Render the frozen System One Auto article assets; no model or network calls.

Charts: matplotlib, editable SVG + embedded-font vector PDF + 300-DPI PNG.
Cover/call flow: repository-native SVG, rendered by a local Chromium binary.
The cover raster is a high-quality JPEG; scientific chart rasters remain PNG.
The cover uses the original logo's alpha geometry with a white SVG filter;
the source logo file is never modified and no logo is redrawn.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

DATA_SHA = "b099ef540ab140c87ad50596cd3bd5d321941d623aeac4244b2bdcc84324d5be"
INK, MUTED, GRID = "#172430", "#5f6b76", "#e7ebef"
AUTO, VEGA, KAI = "#D55E00", "#0072B2", "#63798b"
SIZE = (12.24, 7.48)
TEXT_OVERLAP_TOLERANCE_PX = 2
MAX_COVER_BYTES = 500 * 1024
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "svg.hashsalt": "system-one-auto-20261010",
    }
)


def frame(title, subtitle):
    fig = plt.figure(figsize=SIZE, dpi=300, facecolor="white")
    fig.text(0.06, 0.925, title, fontsize=29, weight="bold", color=INK)
    fig.text(0.06, 0.86, subtitle, fontsize=16, color=MUTED)
    return fig


def footer(fig, text):
    fig.text(0.06, 0.035, text, fontsize=11, color=MUTED)


def chart_axes(fig, rect):
    ax = fig.add_axes(rect)
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)
    ax.tick_params(length=0, pad=9)
    ax.set_axisbelow(True)
    return ax


def inspect_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts)
        if ax.axison:
            texts += ax.get_xticklabels() + ax.get_yticklabels()
            texts += [ax.xaxis.label, ax.yaxis.label, ax.title]
        legend = ax.get_legend()
        if legend:
            texts += legend.get_texts()
    rectangles = []
    for text in texts:
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            rectangles.append((text.get_text(), box))
    width, height = fig.canvas.get_width_height()
    outside = [
        text
        for text, box in rectangles
        if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height
    ]
    overlaps = []
    for i, (left, a) in enumerate(rectangles):
        for right, b in rectangles[i + 1 :]:
            if (
                min(a.x1, b.x1) - max(a.x0, b.x0) > TEXT_OVERLAP_TOLERANCE_PX
                and min(a.y1, b.y1) - max(a.y0, b.y0) > TEXT_OVERLAP_TOLERANCE_PX
            ):
                overlaps.append([left, right])
    return {
        "text_objects": len(rectangles),
        "outside_canvas": outside,
        "text_overlaps": overlaps,
    }


def save_chart(fig, out, name, reports):
    reports[name] = inspect_text(fig)
    fig.savefig(
        out / f"{name}.svg", metadata={"Date": None, "Creator": "vLLM Semantic Router"}
    )
    fig.savefig(
        out / f"{name}.pdf",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "vLLM Semantic Router",
        },
    )
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def quality(data, out, reports):
    fig = frame(
        "More accuracy. Selective escalation.",
        "The same 231 public JevBench requests · exact-label accuracy",
    )
    ax = chart_axes(fig, [0.25, 0.30, 0.69, 0.46])
    order = ["direct_kai", "auto", "direct_vega"]
    colors = [KAI, AUTO, VEGA]
    for y, arm, color in zip([2, 1, 0], order, colors, strict=True):
        result = data["quality"][arm]
        value = 100 * result["accuracy"]
        ax.barh(y, value, color=color, height=0.63, zorder=3)
        ax.text(
            value - 2,
            y,
            f"{value:.2f}%",
            ha="right",
            va="center",
            color="white",
            fontsize=22,
            weight="bold",
        )
        ax.text(
            2, y, f"{result['correct']} / 231", va="center", color="white", fontsize=14
        )
    ax.set(
        yticks=[2, 1, 0],
        yticklabels=[
            "Direct Kai\n0.6B",
            "System One Auto\nKai → Vega",
            "Direct Vega\n27B",
        ],
        xlim=(0, 100),
        ylim=(-0.6, 2.6),
        xticks=[0, 25, 50, 75, 100],
    )
    ax.set_xlabel("Exact-label accuracy (%)", fontsize=14, labelpad=10)
    ax.grid(axis="x", color=GRID)
    fig.text(0.25, 0.155, "+16.45 pp", color=AUTO, fontsize=25, weight="bold")
    fig.text(0.25, 0.105, "vs Kai · 95% CI 11.34\u201321.72", fontsize=12, color=MUTED)
    fig.text(0.65, 0.155, "\u22125.63 pp", color=VEGA, fontsize=25, weight="bold")
    fig.text(
        0.65,
        0.105,
        "vs Vega · 95% CI \u22128.89 to \u22122.59",
        fontsize=12,
        color=MUTED,
    )
    footer(
        fig,
        "195 source groups · paired source-group bootstrap · public suite, not the sealed v1.6.1 composite",
    )
    save_chart(fig, out, "quality", reports)


def latency(data, out, reports):
    fig = frame(
        "Lower mean. A longer tail.",
        "Real frontend latency · both warm passes · lower is better",
    )
    for left, metric, title, limit in (
        (0.09, "mean", "Mean latency", 140),
        (0.57, "p95", "p95 latency", 460),
    ):
        ax = chart_axes(fig, [left, 0.28, 0.36, 0.46])
        for arm, color, offset, label in (
            ("direct_vega", VEGA, -0.18, "Direct Vega"),
            ("auto", AUTO, 0.18, "Auto"),
        ):
            values = [
                data["passes"][str(p)]["arms"][arm]["elapsed_ms"][metric]
                for p in (1, 2)
            ]
            bars = ax.bar(
                [offset, 1 + offset],
                values,
                width=0.31,
                color=color,
                label=label,
                zorder=3,
            )
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + limit * 0.018,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color=color,
                    weight="bold",
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Pass 1", "Pass 2"],
            ylim=(0, limit),
            xlim=(-0.6, 1.6),
        )
        ax.set_ylabel("Milliseconds", fontsize=13, labelpad=8)
        ax.set_title(title, fontsize=20, loc="left", pad=18, weight="bold", color=INK)
        ax.grid(axis="y", color=GRID)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.02, -0.18),
            ncol=2,
            frameon=False,
            fontsize=12,
            handlelength=1,
            columnspacing=1.4,
        )
    fig.text(
        0.09,
        0.115,
        "20.2\u201320.6% lower mean",
        fontsize=18,
        weight="bold",
        color=AUTO,
    )
    fig.text(
        0.57, 0.115, "8.4\u20138.7% higher p95", fontsize=18, weight="bold", color=VEGA
    )
    footer(
        fig,
        "231 requests / arm / pass · concurrency 1 · two resident GPUs · first-pass measurements retained separately",
    )
    save_chart(fig, out, "latency", reports)


def paths(data, out, reports):
    fig = frame(
        "Most answers stop with Kai.",
        "Request outcomes and model calls are different counts.",
    )
    outcomes = data["passes"]["0"]["arms"]["auto"]["served_by"]
    calls = data["passes"]["0"]["arms"]["auto"]["physical_calls"]
    fig.text(
        0.08, 0.75, "231 APPLICATION REQUESTS", fontsize=13, color=MUTED, weight="bold"
    )
    ax = fig.add_axes([0.08, 0.54, 0.84, 0.17])
    ax.barh(0, outcomes["kai"], height=0.95, color=AUTO)
    ax.barh(0, outcomes["vega"], left=outcomes["kai"], height=0.95, color=VEGA)
    for model, center in (
        ("kai", outcomes["kai"] / 2),
        ("vega", outcomes["kai"] + outcomes["vega"] / 2),
    ):
        label = "Return Kai" if model == "kai" else "Upgrade to Vega"
        ax.text(
            center,
            0.09,
            f"{outcomes[model]}",
            ha="center",
            va="center",
            color="white",
            fontsize=30,
            weight="bold",
        )
        ax.text(
            center,
            -0.20,
            f"{label} · {outcomes[model]/231:.2%}",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
        )
    ax.set(xlim=(0, 231), ylim=(-0.52, 0.52))
    ax.axis("off")
    fig.text(
        0.08, 0.44, "ACTUAL NATIVE API CALLS", fontsize=13, color=MUTED, weight="bold"
    )
    for left, name, color, note in (
        (0.08, "kai", KAI, "Every request starts here"),
        (0.52, "vega", VEGA, "Only escalated requests"),
    ):
        card = FancyBboxPatch(
            (left, 0.15),
            0.40,
            0.24,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor="#f6f8fa",
            edgecolor=GRID,
            linewidth=1,
        )
        fig.patches.append(card)
        fig.text(
            left + 0.025,
            0.325,
            name.capitalize(),
            fontsize=17,
            color=color,
            weight="bold",
        )
        fig.text(
            left + 0.365,
            0.265,
            f"{int(calls[name])}",
            fontsize=40,
            color=color,
            weight="bold",
            ha="right",
        )
        fig.text(left + 0.025, 0.20, note, fontsize=13, color=MUTED)
    footer(
        fig,
        "337 native calls in total · at most 2 per request · no claim of fewer provisioned GPUs or lower dollar cost",
    )
    save_chart(fig, out, "paths", reports)


def svg_text(x, y, text, *, size=28, color=INK, weight=400, anchor="start", extra=""):
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" {extra}>{html.escape(text)}</text>'


def banner(logo: Path) -> str:
    encoded = base64.b64encode(logo.read_bytes()).decode()
    parts = [
        """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1920" height="1080" viewBox="0 0 1920 1080">
<defs>
 <linearGradient id="night" x2="0.75" y2="1"><stop stop-color="#111727"/><stop offset=".55" stop-color="#29263b"/><stop offset="1" stop-color="#54404b"/></linearGradient>
 <radialGradient id="horizon"><stop stop-color="#e9a25f" stop-opacity=".68"/><stop offset=".42" stop-color="#b76748" stop-opacity=".28"/><stop offset="1" stop-color="#151827" stop-opacity="0"/></radialGradient>
 <radialGradient id="planet" cx=".18" cy=".5" r=".9"><stop stop-color="#f8cf9a"/><stop offset=".08" stop-color="#d89564"/><stop offset=".4" stop-color="#75515d"/><stop offset=".8" stop-color="#30263a"/></radialGradient>
 <linearGradient id="title" x2="1" y2="1"><stop stop-color="#f5d7ae"/><stop offset="1" stop-color="#f29a53"/></linearGradient>
 <filter id="white-logo"><feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 1 0"/></filter>
</defs>
<rect width="1920" height="1080" fill="url(#night)"/>
<ellipse cx="710" cy="1200" rx="1550" ry="860" fill="url(#horizon)"/>
<circle cx="2150" cy="90" r="590" fill="url(#planet)" opacity=".62"/>
<circle cx="2150" cy="90" r="590" fill="none" stroke="#e9ad79" stroke-opacity=".34" stroke-width="2"/>
<path d="M-120 950 C500 820 1380 950 2020 770" fill="none" stroke="#f7c99e" stroke-opacity=".12" stroke-width="2"/>
<path d="M-120 990 C550 820 1440 990 2020 810" fill="none" stroke="#f7c99e" stroke-opacity=".08" stroke-width="1"/>
"""
    ]
    for x, y, radius, opacity in (
        (160, 150, 2, 0.25),
        (398, 95, 1.5, 0.3),
        (1220, 110, 2, 0.4),
        (1510, 250, 3, 0.7),
        (1660, 650, 1.5, 0.3),
        (232, 626, 2, 0.3),
        (630, 188, 1, 0.4),
        (1100, 766, 1.5, 0.25),
        (810, 79, 2, 0.3),
        (145, 860, 1.5, 0.25),
    ):
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="white" opacity="{opacity}"/>'
        )
    parts += [
        svg_text(
            960,
            285,
            "SYSTEM ONE",
            size=34,
            color="#cfccd9",
            weight=500,
            anchor="middle",
            extra='letter-spacing="9"',
        ),
        svg_text(
            960, 465, "Auto", size=182, color="url(#title)", weight=700, anchor="middle"
        ),
        svg_text(
            960,
            584,
            "Small first. Strong when it counts.",
            size=49,
            color="#ffffff",
            weight=500,
            anchor="middle",
        ),
        svg_text(
            960,
            666,
            "DECISION 2.0  ·  KAI → VEGA",
            size=25,
            color="#e8c5a3",
            anchor="middle",
            extra='letter-spacing="3"',
        ),
        f'<image x="745" y="745" width="430" height="177" xlink:href="data:image/png;base64,{encoded}" filter="url(#white-logo)"/>',
        "</svg>",
    ]
    return "\n".join(parts)


def cascade() -> str:
    parts = [
        """<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="920" viewBox="0 0 1800 920">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="context-stroke"/></marker></defs>
<rect width="1800" height="920" fill="white"/>
<rect x="325" y="145" width="1420" height="675" rx="24" fill="#f7f9fb" stroke="#dde4eb" stroke-width="2"/>
""",
        svg_text(
            55, 85, "One request. An explicit second opinion.", size=46, weight=700
        ),
        svg_text(
            55,
            130,
            "Kai answers the original bundle. Upgrade only when the gate does not pass.",
            size=24,
            color=MUTED,
        ),
        svg_text(
            365,
            198,
            "vllm-sr/auto   ·   SELECTED DECISION / CASCADE",
            size=23,
            color=MUTED,
            weight=600,
        ),
    ]
    boxes = [
        (45, 385, 230, 150, "#ffffff", INK),
        (370, 385, 265, 150, "#fff0e6", AUTO),
        (1010, 640, 280, 155, "#e8f3fa", VEGA),
        (1455, 375, 255, 180, "#ffffff", INK),
    ]
    for x, y, w, h, fill, stroke in boxes:
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
        )
    parts += [
        svg_text(160, 435, "Application", size=31, weight=700, anchor="middle"),
        svg_text(160, 480, "State + questions", size=22, anchor="middle", color=MUTED),
        svg_text(
            502, 435, "Kai 0.6B", size=35, weight=700, anchor="middle", color=AUTO
        ),
        svg_text(502, 481, "Answers once", size=24, anchor="middle"),
        svg_text(
            1150, 693, "Vega 27B", size=35, weight=700, anchor="middle", color=VEGA
        ),
        svg_text(1150, 740, "Original request", size=24, anchor="middle"),
        svg_text(1582, 425, "One complete", size=27, weight=700, anchor="middle"),
        svg_text(1582, 463, "native response", size=27, weight=700, anchor="middle"),
        svg_text(
            1582, 511, "Answers + probabilities", size=18, anchor="middle", color=MUTED
        ),
        '<path d="M800 370 L910 460 L800 550 L690 460Z" fill="#fff9ef" stroke="#8e7353" stroke-width="2.5"/>',
        svg_text(800, 450, "All answers", size=25, weight=600, anchor="middle"),
        svg_text(800, 486, "pass?", size=25, weight=600, anchor="middle"),
    ]
    paths = [
        ("M275 460 H365", INK),
        ("M635 460 H685", INK),
        ("M800 370 V285 H1582 V370", AUTO),
        ("M800 550 V718 H1005", VEGA),
        ("M1290 718 H1582 V560", VEGA),
    ]
    for d, color in paths:
        parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" marker-end="url(#arrow)"/>'
        )
    parts += [
        svg_text(
            1170,
            262,
            "PASS · keep Kai\u2019s answer",
            size=26,
            weight=600,
            anchor="middle",
            color=AUTO,
        ),
        svg_text(820, 613, "OTHERWISE", size=23, weight=600, color=VEGA),
        svg_text(
            502, 590, "No extra classifier call", size=21, color=MUTED, anchor="middle"
        ),
        svg_text(55, 867, "Choice · Score · Noul", size=24, color=MUTED),
        svg_text(
            1745,
            867,
            "At most two native model calls in this example",
            size=24,
            color=MUTED,
            anchor="end",
        ),
        "</svg>",
    ]
    return "\n".join(parts)


def render_svg(svg: Path, chromium: Path, width: int, height: int):
    wrapper = svg.with_suffix(".render.html")
    wrapper.write_text(
        f"<html><head><style>@page{{size:{width}px {height}px;margin:0}}html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden}}svg{{display:block}}</style></head><body>{svg.read_text()}</body></html>"
    )
    common = [
        str(chromium),
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=1500",
    ]
    subprocess.run(
        [
            *common,
            f"--window-size={width},{height}",
            f"--screenshot={svg.with_suffix('.png').resolve()}",
            wrapper.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            *common,
            f"--print-to-pdf={svg.with_suffix('.pdf').resolve()}",
            wrapper.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    wrapper.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--logo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromium", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("figure data differs from frozen evidence")
    data = json.loads(args.data.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for draw in (quality, latency, paths):
        draw(data, args.output, reports)
    for name, source, size in (
        ("hero", banner(args.logo), (1920, 1080)),
        ("cascade", cascade(), (1800, 920)),
    ):
        svg = args.output / f"{name}.svg"
        svg.write_text(source)
        render_svg(svg, args.chromium, *size)
        if name == "hero":
            with Image.open(svg.with_suffix(".png")) as raster:
                raster.convert("RGB").save(
                    svg.with_suffix(".jpg"),
                    quality=98,
                    subsampling=0,
                    optimize=True,
                    progressive=True,
                )
            if svg.with_suffix(".jpg").stat().st_size >= MAX_COVER_BYTES:
                raise ValueError("cover JPEG exceeds the repository asset limit")
            svg.with_suffix(".png").unlink()
    files = {
        path.name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(args.output.iterdir())
        if path.suffix in {".svg", ".pdf", ".png", ".jpg"}
    }
    for name, record in files.items():
        if name.endswith((".png", ".jpg")):
            record["pixels"] = list(Image.open(args.output / name).size)
    report = {
        "data_sha256": DATA_SHA,
        "logo_sha256": hashlib.sha256(args.logo.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "logo_treatment": "Original alpha geometry rendered in white by SVG filter; source PNG unchanged; no KR Labs mark",
        "cover_raster": {
            "format": "JPEG",
            "quality": 98,
            "chroma_subsampling": "4:4:4",
            "progressive": True,
        },
        "matplotlib_text_checks": reports,
        "files": files,
        "requires_visual_review": True,
        "no_model_or_network_calls": True,
    }
    (args.output / "generation-receipt.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
