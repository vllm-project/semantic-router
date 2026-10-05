"""Render the Vela 2.0 launch-post charts.

Every number below is the one published in the first Vela 2.0 launch post
(PR #4557); the comment beside each block names that post's section.
Run from this directory: python3 make_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, BG = "#172430", "#5f6b76", "#e7ebef", "#ffffff"
AMBER, ORANGE, ROSE, SLATE, PALE = "#e9a25f", "#e07a4f", "#cf5f78", "#3f566b", "#c9d1d8"
SIZE = (12.24, 7.48)  # 2448 x 1496 px at 200 dpi, the Vela 1.0 chart size

for name in ("Arial", "Helvetica", "DejaVu Sans"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update(
    {
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "font.size": 18,
    }
)


def frame(title, subtitle, footer, top=0.76):
    fig = plt.figure(figsize=SIZE, dpi=200, facecolor=BG)
    ax = fig.add_axes([0.3, 0.15, 0.64, top - 0.15])
    fig.text(0.06, 0.92, title, fontsize=34, fontweight="bold", color=INK)
    fig.text(0.06, 0.865, subtitle, fontsize=19, color=MUTED)
    fig.text(
        0.94,
        0.925,
        "VELA 2.0",
        fontsize=21,
        fontweight="bold",
        color=ORANGE,
        ha="right",
    )
    fig.text(0.06, 0.04, footer, fontsize=14, color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor=BG)
    plt.close(fig)


def router_tasks():
    # Section "Against the Vela 1.0 specialists": specialist test rows, 9B vs Vela 1.0.
    rows = [
        ("Prompt attacks, unseen families", "AUC", 0.792, 0.989, "Guard"),
        ("Multilingual HateCheck", "AUC", 0.646, 0.855, "Safety"),
        ("RTP-LX request harm", "AUC", 0.761, 0.801, "Safety"),
        ("PII, 8K-token documents", "F1", 0.908, 0.940, "PII"),
        ("Hallucination, 10,698 examples", "example-F1", 0.875, 0.885, "Halu"),
        ("Domain", "macro-F1", 0.831, 0.844, "Domain"),
        ("PII, short texts", "F1", 0.976, 0.985, "PII"),
    ]
    fig, ax = frame(
        "One model, every router signal",
        "Vela 2.0 9B against each Vela 1.0 specialist, on the specialist's own test rows",
        "Paired comparison on identical rows. Measured for the Vela 2.0 release.",
    )
    ax.set_position([0.35, 0.15, 0.59, 0.61])
    for i, (task, metric, old, new, spec) in enumerate(reversed(rows)):
        ax.plot([old, new], [i, i], color=PALE, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter(old, i, s=110, color=SLATE, zorder=2)
        ax.scatter(new, i, s=170, color=ORANGE, zorder=3)
        ax.text(
            old - 0.008,
            i,
            f"{old:.3f}",
            ha="right",
            va="center",
            color=SLATE,
            fontsize=17,
        )
        ax.text(
            new + 0.008,
            i,
            f"{new:.3f}",
            ha="left",
            va="center",
            color=ORANGE,
            fontsize=17,
            fontweight="bold",
        )
        ax.text(
            0.0,
            i,
            f"{task}  ",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=INK,
            fontsize=18,
        )
        ax.text(
            0.0,
            i - 0.36,
            f"{metric} · vs {spec}  ",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=MUTED,
            fontsize=14,
        )
    ax.set_yticks([])
    ax.set_xlim(0.6, 1.03)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.scatter([], [], s=110, color=SLATE, label="Vela 1.0 specialist")
    ax.scatter([], [], s=170, color=ORANGE, label="Vela 2.0 9B")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.0),
        ncol=2,
        frameon=False,
        fontsize=17,
    )
    save(fig, "router-tasks.png")


def safety_family():
    # Section "Safety against a generic decision model": macro AUC over 14 public router safety sets.
    rows = [
        ("GLiNER2.5-Decide", 0.704, SLATE),
        ("Vela 2.0 0.3B", 0.871, AMBER),
        ("Vela 2.0 0.8B", 0.875, AMBER),
        ("Vela 2.0 4B", 0.921, ORANGE),
        ("Vela 2.0 9B", 0.921, ROSE),
    ]
    fig, ax = frame(
        "Safety across the family",
        "Macro AUC over 14 public safety and prompt-attack sets",
        "RTP-LX, HateCheck, XSTest, CultureGuard, AEGIS 2.0, PolyGuard, Do-Not-Answer and "
        "prompt-attack sets. Measured for the Vela 2.0 release.",
        top=0.8,
    )
    for i, (_name, v, c) in enumerate(reversed(rows)):
        ax.barh(i, v - 0.5, left=0.5, height=0.58, color=c)
        ax.text(
            v + 0.006,
            i,
            f"{v:.3f}",
            va="center",
            color=INK,
            fontsize=18,
            fontweight="bold",
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=18)
    ax.set_xlim(0.5, 1.0)
    save(fig, "safety-family.png")


def evidence():
    # Section "Open extraction with the broad head": ACL-Verbatim word-F1, held out of training, one harness for all models.
    rows = [
        ("Vela 2.0 9B", 24.5, ROSE),
        ("Vela 2.0 4B", 24.4, ORANGE),
        ("Vela 2.0 0.8B", 23.6, AMBER),
        ("GLiFormer-large", 7.0, SLATE),
        ("GLiNER-large-v2.5", 4.6, SLATE),
        ("GLiNER2.5-small", 4.6, SLATE),
        ("GLiNER2.5-Decide", 2.3, SLATE),
    ]
    fig, ax = frame(
        "The exact words that answer",
        "Extractive evidence on ACL-Verbatim (word-F1), a set held out of training",
        "All models scored by one harness. Vela uses the broad span head. Measured for the Vela 2.0 release.",
        top=0.8,
    )
    for i, (_name, v, c) in enumerate(reversed(rows)):
        ax.barh(i, v, height=0.58, color=c)
        ax.text(
            v + 0.3,
            i,
            f"{v:.1f}",
            va="center",
            color=INK,
            fontsize=18,
            fontweight="bold",
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=18)
    ax.set_xlim(0, 28)
    save(fig, "evidence.png")


def latency():
    # Section "Cost": one router request with seven questions, PII and hallucination spans included, one A40.
    rows = [
        ("Vela 2.0 0.3B", 0.09, 0.09, AMBER),
        ("Vela 2.0 0.8B", 0.13, 0.13, AMBER),
        ("Vela 2.0 4B", 0.40, 0.49, ORANGE),
        ("Vela 2.0 9B", 0.60, 0.71, ROSE),
    ]
    fig, ax = frame(
        "Every signal, one call",
        "Seconds per router request: seven questions, PII and hallucination spans included",
        "One NVIDIA A40. Ranges cover the demo requests. Measured for the Vela 2.0 release.",
        top=0.8,
    )
    for i, (_name, lo, hi, c) in enumerate(reversed(rows)):
        ax.barh(i, hi, height=0.58, color=c)
        label = f"{lo:.2f} s" if lo == hi else f"{lo:.2f}\u2013{hi:.2f} s"
        ax.text(
            hi + 0.01, i, label, va="center", color=INK, fontsize=18, fontweight="bold"
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=18)
    ax.set_xlim(0, 0.85)
    save(fig, "latency.png")


def jev():
    # Section "Where it gives ground": Jev Decision Index 0.2.1, 38 benchmarks, base vs Vela 2.0.
    rows = [
        ("0.8B", "Eos-0.8B", 20.14, 16.01),
        ("4B", "Nox-4B", 42.55, 31.63),
        ("9B", "Lux-9B", 46.23, 41.09),
    ]
    fig, ax = frame(
        "General decisions, kept",
        "Jev Decision Index 0.2.1, 38 benchmarks: Vela 2.0 against the Decision 2.0 model it starts from",
        "Our harness reproduces the published Decision 2.0 index scores within 0.1 points. Measured for the Vela 2.0 release.",
        top=0.8,
    )
    ax.set_position([0.12, 0.15, 0.82, 0.65])
    for i, (size, base, b, v) in enumerate(reversed(rows)):
        ax.barh(i + 0.2, b, height=0.36, color=PALE)
        ax.barh(i - 0.2, v, height=0.36, color=ORANGE)
        ax.text(
            b + 0.5,
            i + 0.2,
            f"{b:.2f}  Decision 2.0 {base}",
            va="center",
            color=MUTED,
            fontsize=16,
        )
        ax.text(
            v + 0.5,
            i - 0.2,
            f"{v:.2f}  Vela 2.0 {size} · {v / b:.0%} kept",
            va="center",
            color=INK,
            fontsize=16,
            fontweight="bold",
        )
    ax.set_yticks(range(len(rows)), [f"{r[0]}" for r in reversed(rows)], fontsize=20)
    ax.set_xlim(0, 75)
    save(fig, "jev-index.png")


if __name__ == "__main__":
    for chart in (router_tasks, safety_family, evidence, latency, jev):
        chart()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
