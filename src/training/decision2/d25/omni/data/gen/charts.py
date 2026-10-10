"""CharXiv-style reasoning over scientific charts rendered with matplotlib from random data.

Each item is one figure and one reasoning question with four options drawn from the same figure
(series names, subplot titles, tick values or nearby numbers). Questions are only asked when the
answer is visually unambiguous (top-two gaps of at least 8% of the axis range).
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image

from d25.omni.data import render
from d25.omni.data.rows import Item, fmt_number

METHODS = (
    "Ours",
    "Baseline",
    "ResNet-50",
    "ViT-B/16",
    "LSTM",
    "GCN",
    "Transformer",
    "Random Forest",
    "SVM",
    "XGBoost",
    "U-Net",
    "BERT",
    "MLP",
    "CNN",
    "Kalman filter",
    "PPO",
    "SAC",
    "DQN",
    "Adam",
    "SGD",
    "AdamW",
    "LoRA",
    "Full FT",
    "Greedy",
    "Beam search",
    "k-NN",
    "Linear",
    "Model A",
    "Model B",
    "Model C",
    "Model D",
    "Sample 1",
    "Sample 2",
    "Sample 3",
    "Sample 4",
    "Control",
    "Treatment",
    "Group A",
    "Group B",
    "Group C",
    "Group D",
    "Site 1",
    "Site 2",
    "Site 3",
)
X_AXES = (
    ("Epoch", "epoch"),
    ("Training steps (k)", "step"),
    ("Year", "year"),
    ("Temperature (K)", "temp"),
    ("Time (s)", "time"),
    ("Number of samples", "n"),
    ("Frequency (Hz)", "freq"),
    ("Batch size", "batch"),
)
Y_AXES = (
    "Accuracy (%)",
    "Loss",
    "F1 score",
    "Throughput (img/s)",
    "Error rate (%)",
    "Reward",
    "Voltage (V)",
    "Energy (meV)",
    "Yield (%)",
    "Latency (ms)",
    "Precision",
    "Normalized intensity",
    "Speedup",
    "Concentration (mg/L)",
    "Population (millions)",
    "Revenue ($M)",
)
TITLES = (
    "Validation performance",
    "Scaling behaviour",
    "Ablation study",
    "Training curves",
    "Measured response",
    "Comparison across settings",
    "Sensitivity analysis",
    "Results by condition",
)
CATEGORIES = (
    "CIFAR-10",
    "ImageNet",
    "COCO",
    "SQuAD",
    "MNIST",
    "Sim",
    "Real",
    "Low",
    "Medium",
    "High",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "North",
    "South",
    "East",
    "West",
    "Small",
    "Base",
    "Large",
    "XL",
    "Task 1",
    "Task 2",
    "Task 3",
    "Task 4",
    "Task 5",
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
)
STYLES = (
    "default",
    "seaborn-v0_8-whitegrid",
    "ggplot",
    "bmh",
    "seaborn-v0_8-paper",
    "seaborn-v0_8-ticks",
    "classic",
    "seaborn-v0_8-darkgrid",
)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if not getattr(_plt, "fonts", False):
        for path in render.FONT_DIR.glob("*.ttf"):
            try:
                font_manager.fontManager.addfont(str(path))
            except Exception:
                pass
        _plt.fonts = True
    return plt


def _family(rng: random.Random) -> str:
    from matplotlib import font_manager

    path = render.font_path(rng.choice(["sans", "sans", "serif"]), rng)
    return font_manager.FontProperties(fname=path).get_name()


def _figure(rng: random.Random, plt, shape=(1, 1)):
    style = rng.choice(STYLES)
    plt.style.use(style)
    plt.rcParams["font.family"] = _family(rng)
    plt.rcParams["font.size"] = rng.choice([9, 10, 11, 12])
    w = rng.uniform(5.5, 9.5) * (1.4 if shape[1] > 1 else 1)
    h = rng.uniform(3.6, 6.0) * (1.25 if shape[0] > 1 else 1)
    fig, axes = plt.subplots(*shape, figsize=(w, h), squeeze=False)
    return fig, axes


def _to_image(fig, plt, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=rng.choice([100, 120, 140, 160]))
    plt.close(fig)
    return render.fit_pixels(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))


def _x_values(rng: random.Random, kind: str, n: int) -> np.ndarray:
    start = {
        "year": rng.randint(1990, 2015),
        "epoch": 0,
        "step": 0,
        "temp": 100,
        "time": 0,
        "n": 0,
        "freq": 10,
        "batch": 0,
    }[kind]
    step = {
        "year": 1,
        "epoch": rng.choice([5, 10, 20]),
        "step": rng.choice([10, 25, 50]),
        "temp": rng.choice([25, 50]),
        "time": rng.choice([1, 2, 5]),
        "n": rng.choice([100, 500, 1000]),
        "freq": rng.choice([10, 20, 50]),
        "batch": rng.choice([16, 32, 64]),
    }[kind]
    return np.array([start + step * (i + (kind in ("n", "batch"))) for i in range(n)])


def _series(rng: random.Random, n: int, k: int) -> np.ndarray:
    out = []
    for _ in range(k):
        base = rng.uniform(10, 80)
        slope = rng.uniform(-6, 9)
        curve = rng.choice(["linear", "sat", "peak"])
        t = np.arange(n)
        if curve == "linear":
            y = base + slope * t
        elif curve == "sat":
            y = base + slope * 6 * (1 - np.exp(-t / rng.uniform(1.5, 4)))
        else:
            c = rng.uniform(1, n - 2)
            y = base + rng.uniform(15, 40) * np.exp(-((t - c) ** 2) / rng.uniform(2, 8))
        y = y + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(0.5, 3), n
        )
        out.append(np.round(y, 1))
    return np.array(out)


def _clear_top(values: np.ndarray, frac=0.08, highest=True) -> int | None:
    order = np.argsort(values)
    a, b = (order[-1], order[-2]) if highest else (order[0], order[1])
    span = float(values.max() - values.min()) or 1.0
    return int(a) if abs(values[a] - values[b]) >= frac * span else None


def _options(
    gold: str, pool: list[str], rng: random.Random, n: int = 4
) -> list[str] | None:
    others = [p for p in dict.fromkeys(pool) if p != gold]
    if len(others) < n - 1:
        return None
    return [gold] + rng.sample(others, n - 1)


def _line_item(rng: random.Random, plt):
    xlabel, xkind = rng.choice(X_AXES)
    n, k = rng.randint(6, 12), rng.randint(3, 6)
    x = _x_values(rng, xkind, n)
    ys = _series(rng, n, k)
    names = rng.sample(METHODS, k)
    fig, axes = _figure(rng, plt)
    ax = axes[0][0]
    palette = rng.choice(render.PALETTES)
    markers = rng.choice([True, False])
    for i in range(k):
        ax.plot(
            x,
            ys[i],
            label=names[i],
            color=palette[i % len(palette)],
            marker="osd^v<>*"[i % 8] if markers else None,
            linewidth=rng.uniform(1.2, 2.4),
            linestyle=rng.choice(["-", "-", "--", "-."]) if not markers else "-",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(rng.choice(Y_AXES))
    if rng.random() < 0.7:
        ax.set_title(rng.choice(TITLES))
    ax.legend(
        loc=rng.choice(["best", "upper left", "lower right", "center right"]),
        fontsize="small",
    )
    if rng.random() < 0.6:
        ax.grid(True, alpha=0.4)
    qtype = rng.choice(["max_at", "argmax_x", "largest_gain", "lowest_final"])
    question = options = None
    gold = ""
    if qtype == "max_at":
        j = rng.randrange(n)
        top = _clear_top(ys[:, j])
        if top is not None:
            gold = names[top]
            question = f"Which series has the highest value at {xlabel.split(' (')[0].lower()} = {x[j]}?"
            options = _options(gold, names, rng)
    elif qtype == "argmax_x":
        i = rng.randrange(k)
        j = int(np.argmax(ys[i]))
        span = float(ys[i].max() - ys[i].min()) or 1
        if sorted(ys[i])[-1] - sorted(ys[i])[-2] >= 0.05 * span:
            gold = str(x[j])
            question = f"At which {xlabel.split(' (')[0].lower()} value does {names[i]} reach its maximum?"
            options = _options(gold, [str(v) for v in x], rng)
    elif qtype == "largest_gain":
        gains = ys[:, -1] - ys[:, 0]
        top = _clear_top(gains)
        if top is not None:
            gold = names[top]
            question = (
                f"Which series shows the largest increase between {xlabel.split(' (')[0].lower()} "
                f"{x[0]} and {x[-1]}?"
            )
            options = _options(gold, names, rng)
    else:
        top = _clear_top(ys[:, -1], highest=False)
        if top is not None:
            gold = names[top]
            question = "Which series ends with the lowest value?"
            options = _options(gold, names, rng)
    return fig, question, options, qtype, [*names, xlabel]


def _bar_item(rng: random.Random, plt):
    cats = rng.sample(CATEGORIES, rng.randint(3, 6))
    groups = rng.sample(METHODS, rng.randint(2, 4))
    values = np.round(np.array([[rng.uniform(5, 95) for _ in cats] for _ in groups]), 1)
    fig, axes = _figure(rng, plt)
    ax = axes[0][0]
    palette = rng.choice(render.PALETTES)
    width = 0.8 / len(groups)
    xs = np.arange(len(cats))
    horizontal = rng.random() < 0.3
    labels_on = rng.random() < 0.6
    for g, name in enumerate(groups):
        pos = xs + (g - (len(groups) - 1) / 2) * width
        bars = (ax.barh if horizontal else ax.bar)(
            pos, values[g], width, label=name, color=palette[g % len(palette)]
        )
        if labels_on:
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    (ax.set_yticks if horizontal else ax.set_xticks)(xs)
    (ax.set_yticklabels if horizontal else ax.set_xticklabels)(cats)
    (ax.set_xlabel if horizontal else ax.set_ylabel)(rng.choice(Y_AXES))
    ax.legend(fontsize="small")
    if rng.random() < 0.6:
        ax.set_title(rng.choice(TITLES))
    qtype = rng.choice(
        ["max_total", "max_diff", "value"]
        if labels_on
        else ["max_total", "max_diff", "best_group"]
    )
    question = options = None
    gold = ""
    if qtype == "max_total":
        top = _clear_top(values.sum(axis=0))
        if top is not None:
            gold = cats[top]
            question = (
                f"Which category has the largest total across all {len(groups)} groups?"
            )
            options = _options(gold, cats, rng)
    elif qtype == "max_diff" and len(groups) >= 2:
        diff = np.abs(values[0] - values[1])
        top = _clear_top(diff)
        if top is not None:
            gold = cats[top]
            question = f"In which category is the gap between {groups[0]} and {groups[1]} the largest?"
            options = _options(gold, cats, rng)
    elif qtype == "value":
        g, c = rng.randrange(len(groups)), rng.randrange(len(cats))
        gold = f"{values[g, c]:.1f}"
        pool = [f"{v:.1f}" for v in values.flatten()]
        question = f"What is the value of {groups[g]} for {cats[c]}?"
        options = _options(gold, pool, rng)
    else:
        top = _clear_top(values.mean(axis=1))
        if top is not None:
            gold = groups[top]
            question = "Which group has the highest average across categories?"
            options = _options(
                gold,
                groups + rng.sample([m for m in METHODS if m not in groups], 3),
                rng,
            )
    return fig, question, options, f"bar_{qtype}", [*cats, *groups]


def _subplot_item(rng: random.Random, plt):
    shape = rng.choice([(2, 2), (1, 3), (1, 4), (2, 3)])
    count = shape[0] * shape[1]
    titles = (
        [f"({'abcdef'[i]})" for i in range(count)]
        if rng.random() < 0.3
        else [f"({'abcdef'[i]}) {rng.choice(CATEGORIES)}" for i in range(count)]
    )
    if len(set(titles)) < count:
        titles = [f"({'abcdef'[i]})" for i in range(count)]
    fig, axes = _figure(rng, plt, shape)
    n = rng.randint(8, 15)
    x = np.arange(n)
    trends, peaks = [], []
    for i, ax in enumerate(axes.flatten()):
        slope = rng.uniform(-5, 5)
        while abs(slope) < 1.2:
            slope = rng.uniform(-5, 5)
        y = (
            rng.uniform(20, 60)
            + slope * x
            + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1.5, n)
        )
        ax.plot(
            x,
            y,
            color=rng.choice(rng.choice(render.PALETTES)),
            marker="o" if rng.random() < 0.5 else None,
        )
        ax.set_title(titles[i], fontsize="medium")
        trends.append(slope)
        peaks.append(float(y.max()))
    qtype = rng.choice(["only_decreasing", "highest_peak"])
    question = options = None
    gold = ""
    if qtype == "only_decreasing":
        dec = [i for i, s in enumerate(trends) if s < 0]
        if len(dec) == 1:
            gold = titles[dec[0]]
            question = "Which subplot shows a decreasing trend?"
            options = _options(gold, titles, rng)
        elif len(dec) == count - 1:
            gold = titles[[i for i in range(count) if i not in dec][0]]
            question = "Which subplot shows an increasing trend?"
            options = _options(gold, titles, rng)
    else:
        top = _clear_top(np.array(peaks), 0.05)
        if top is not None:
            gold = titles[top]
            question = "In which subplot does the curve reach the highest value?"
            options = _options(gold, titles, rng)
    if options is not None and len(options) < 4:
        options = None
    return fig, question, options, f"subplot_{qtype}", titles


def _pie_item(rng: random.Random, plt):
    labels = rng.sample(CATEGORIES, rng.randint(4, 7))
    raw = np.array([rng.uniform(3, 40) for _ in labels])
    shares = np.round(raw / raw.sum() * 100, 1)
    fig, axes = _figure(rng, plt)
    ax = axes[0][0]
    wedge = {"width": 0.45} if rng.random() < 0.4 else None
    ax.pie(
        shares,
        labels=labels,
        autopct="%1.1f%%",
        startangle=rng.randint(0, 360),
        colors=rng.choice(render.PALETTES)[: len(labels)],
        wedgeprops=wedge,
        textprops={"fontsize": 8},
    )
    ax.axis("equal")
    if rng.random() < 0.7:
        ax.set_title(rng.choice(TITLES))
    order = np.argsort(shares)[::-1]
    qtype = rng.choice(["second", "sum"])
    if (
        qtype == "second"
        and shares[order[1]] - shares[order[2]] >= 1.0
        and shares[order[0]] - shares[order[1]] >= 1.0
    ):
        gold = labels[order[1]]
        return (
            fig,
            "Which category has the second-largest share?",
            _options(gold, labels, rng),
            "pie_second",
            labels,
        )
    a, b = rng.sample(range(len(labels)), 2)
    total = round(shares[a] + shares[b], 1)
    pool = [
        f"{round(shares[i] + shares[j], 1):.1f}%"
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    gold = f"{total:.1f}%"
    question = f"What is the combined share of {labels[a]} and {labels[b]}?"
    return fig, question, _options(gold, pool, rng), "pie_sum", labels


def _scatter_item(rng: random.Random, plt):
    groups = rng.sample(METHODS, rng.randint(3, 5))
    fig, axes = _figure(rng, plt)
    ax = axes[0][0]
    palette = rng.choice(render.PALETTES)
    threshold = round(rng.uniform(30, 70))
    means, above = [], []
    for g, name in enumerate(groups):
        m = rng.uniform(15, 85)
        count = rng.randint(4, 9)
        gen = np.random.default_rng(rng.randrange(1 << 30))
        xs, ys = gen.uniform(0, 100, count), np.clip(
            gen.normal(m, rng.uniform(3, 9), count), 0, 100
        )
        ys = np.where(np.abs(ys - threshold) < 2.5, ys + 5, ys)
        ax.scatter(
            xs,
            ys,
            label=name,
            color=palette[g % len(palette)],
            marker="osd^v"[g % 5],
            s=rng.uniform(25, 60),
        )
        means.append(float(ys.mean()))
        above.append(int((ys > threshold).sum()))
    ax.axhline(threshold, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel(rng.choice(Y_AXES))
    ax.set_ylabel(rng.choice(Y_AXES))
    ax.legend(fontsize="small")
    if rng.random() < 0.5:
        g = rng.randrange(len(groups))
        gold = str(above[g])
        pool = [str(v) for v in range(0, 10)]
        question = f"How many {groups[g]} points lie above the dashed line?"
        return fig, question, _options(gold, pool, rng), "scatter_count", groups
    top = _clear_top(np.array(means), 0.1)
    if top is None:
        return fig, None, None, "scatter_mean", groups
    gold = groups[top]
    return (
        fig,
        "Which group has the highest average vertical position?",
        _options(gold, groups + ["None of them"], rng),
        "scatter_mean",
        groups,
    )


KINDS = (
    (_line_item, 0.34),
    (_bar_item, 0.26),
    (_subplot_item, 0.14),
    (_pie_item, 0.12),
    (_scatter_item, 0.14),
)


def generate(index: int, ctx=None) -> Item | None:
    from d25.omni.data.rows import rng_for

    rng = rng_for("gen-chart", index)
    plt = _plt()
    builder = rng.choices([k for k, _ in KINDS], weights=[w for _, w in KINDS])[0]
    fig, question, options, qtype, texts = builder(rng, plt)
    if question is None or options is None:
        plt.close(fig)
        return None
    image = _to_image(fig, plt, rng)
    return Item(
        source="gen-chart",
        family="chart",
        skill="chart",
        images=[image],
        image_kinds=["png"],
        question=question,
        options=options,
        gold=0,
        meta={"qtype": qtype, "benchmark_target": "CharXiv"},
        image_text=[" ".join(texts)],
    )


__all__ = ["generate", "fmt_number"]
