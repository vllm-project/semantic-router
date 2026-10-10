"""CharXiv-style proxy: code-rendered scientific figures with reasoning questions.

Each row is one freshly rendered matplotlib figure (line plots, grouped bars, multi-panel plots,
labelled scatter plots, heatmaps, box plots) in a paper-like style and one reasoning question turned
into 4-way multiple choice with every distractor taken from the same figure (series labels, tick
values, panel titles, values of other series). The answer is computed from the plotted data, and
every question keeps a visible margin between the answer and the closest distractor.
"""

from __future__ import annotations

import io
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BENCHMARK = "CharXiv"
NAME = "charxiv-proxy"
VERSION = "1"

METHODS = [
    "Ours",
    "Baseline",
    "ResNet-50",
    "ViT-B/16",
    "ViT-L/14",
    "ConvNeXt-T",
    "Swin-S",
    "MoCo v3",
    "SimCLR",
    "DINO",
    "BYOL",
    "MAE",
    "CLIP",
    "LoRA",
    "Full FT",
    "Adapter",
    "Prefix tuning",
    "BitFit",
    "BERT-base",
    "RoBERTa",
    "T5-small",
    "LSTM",
    "GRU",
    "Transformer",
    "Mamba",
    "RWKV",
    "S4",
    "PPO",
    "SAC",
    "TD3",
    "DQN",
    "A2C",
    "Random",
    "Greedy",
    "Oracle",
    "k-NN",
    "SVM",
    "XGBoost",
    "MLP",
    "GCN",
    "GAT",
    "GraphSAGE",
    "Adam",
    "SGD",
    "AdamW",
    "Lion",
    "Sophia",
    "LAMB",
    "FedAvg",
    "FedProx",
    "SCAFFOLD",
    "DP-SGD",
    "Mixup",
    "CutMix",
    "RandAugment",
    "Dropout",
    "Label smoothing",
    "Distillation",
    "Pruning",
    "Quantization",
    "Ensemble",
    "Bayesian",
    "MC Dropout",
    "Deep Ensemble",
    "Temperature scaling",
    "UNet",
    "DeepLabV3",
    "Mask R-CNN",
    "YOLOv8",
    "DETR",
    "Faster R-CNN",
    "NeRF",
    "3DGS",
    "DDPM",
    "DDIM",
    "Flow matching",
    "VAE",
    "GAN",
]
VARIANTS = [
    "w/o aug",
    "w/o pretrain",
    "+ CL",
    "frozen",
    "large",
    "small",
    "v2",
    "(ours)",
    "+ reg",
]
PARAMS = ["λ", "β", "α", "τ", "k", "r", "T", "γ"]
X_AXES = [
    ("Training set size", "log", [100, 300, 1000, 3000, 10000, 30000, 100000]),
    ("Number of parameters (M)", "log", [10, 30, 100, 300, 1000, 3000]),
    ("Context length", "log2", [512, 1024, 2048, 4096, 8192, 16384, 32768]),
    ("Epoch", "lin", [0, 25, 50, 75, 100, 125, 150, 175, 200]),
    ("Noise level σ", "lin", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    ("Training steps (k)", "lin", [0, 10, 20, 30, 40, 50, 60, 70, 80]),
    ("Labeled fraction (%)", "lin", [1, 5, 10, 20, 50, 100]),
    ("Sparsity (%)", "lin", [0, 20, 40, 60, 80, 90, 95]),
    ("Batch size", "log2", [16, 32, 64, 128, 256, 512, 1024]),
    ("Number of shots", "lin", [0, 1, 2, 4, 8, 16, 32]),
    ("Rank", "log2", [1, 2, 4, 8, 16, 32, 64]),
    ("Temperature", "lin", [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]),
]
Y_AXES = [
    ("Accuracy (%)", "accuracy", 30, 95, True),
    ("Top-1 accuracy (%)", "top-1 accuracy", 40, 90, True),
    ("F1 score", "F1 score", 0.3, 0.95, True),
    ("BLEU", "BLEU", 10, 45, True),
    ("Success rate (%)", "success rate", 5, 95, True),
    ("Average return", "average return", 200, 3500, True),
    ("AUROC", "AUROC", 0.55, 0.98, True),
    ("Recall@10 (%)", "Recall@10", 20, 90, True),
    ("mIoU (%)", "mIoU", 25, 80, True),
    ("Test loss", "test loss", 0.2, 2.5, False),
    ("Perplexity", "perplexity", 8, 60, False),
    ("Error rate (%)", "error rate", 2, 40, False),
    ("MSE", "MSE", 0.01, 0.5, False),
    ("Latency (ms)", "latency", 5, 120, False),
]
DATASETS = [
    "CIFAR-10",
    "CIFAR-100",
    "ImageNet",
    "SVHN",
    "STL-10",
    "SST-2",
    "MNLI",
    "QNLI",
    "SQuAD",
    "COCO",
    "Cityscapes",
    "ADE20K",
    "HalfCheetah",
    "Hopper",
    "Walker2d",
    "Ant",
    "Atari",
    "MiniGrid",
    "WikiText-103",
    "PTB",
    "Librispeech",
    "VoxCeleb",
    "MIMIC-III",
    "OGB-arxiv",
    "Cora",
    "PubMed",
    "QM9",
    "ZINC",
    "KITTI",
    "nuPlan",
    "Places365",
    "iNaturalist",
    "Food-101",
    "Flowers-102",
    "DTD",
    "EuroSAT",
    "GTSRB",
]
CATEGORIES = [
    ["en", "de", "fr", "es", "zh", "ja", "ar", "hi", "ru", "pt"],
    ["News", "Legal", "Medical", "Code", "Reviews", "Finance", "Science", "Dialogue"],
    ["Easy", "Medium", "Hard", "Expert"],
    ["Clean", "Blur", "Noise", "JPEG", "Fog", "Snow", "Contrast", "Pixelate"],
    ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["Small", "Base", "Large", "XL"],
]
STYLES = [
    "default",
    "seaborn-v0_8-whitegrid",
    "seaborn-v0_8-paper",
    "ggplot",
    "bmh",
    "seaborn-v0_8-ticks",
    "classic",
]
PALETTES = ["tab10", "Set1", "Dark2", "tab20", "Paired", "viridis", "plasma", "cividis"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":"]


@dataclass
class Figure:
    """A rendered figure with one question and its 4 options (correct first)."""

    draw: Callable[[Any], None]
    question: str
    correct: str
    distractors: list[str]
    subtask: str
    size: tuple[float, float]
    dpi: int
    style: str
    meta: dict[str, Any] = field(default_factory=dict)


def method_names(r: random.Random, n: int) -> list[str]:
    kind = r.random()
    if kind < 0.6:
        names = r.sample(METHODS, n)
        if "Ours" not in names and r.random() < 0.4:
            names[r.randrange(n)] = "Ours"
    elif kind < 0.8:
        base = r.choice(METHODS)
        names = [base] + [f"{base} ({v})" for v in r.sample(VARIANTS, n - 1)]
    else:
        p = r.choice(PARAMS)
        values = sorted(
            r.sample([0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64], n)
        )
        names = [f"{p} = {v:g}" for v in values]
    return names


def article(word: str) -> str:
    return (
        "an" if word[:1].lower() in "aeiou" or word[:2] in ("F1", "MS", "mI") else "a"
    )


def fmt_tick(value: float, scale: str) -> str:
    if scale == "log" and value >= 1000:
        exponent = math.log10(value)
        if abs(exponent - round(exponent)) < 1e-9:
            return f"10^{int(round(exponent))}"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def fmt_value(value: float, span: float) -> str:
    if span >= 50:
        return f"{value:.0f}"
    if span >= 5:
        return f"{value:.1f}"
    if span >= 0.5:
        return f"{value:.2f}"
    return f"{value:.3f}"


def colors(r: random.Random, n: int) -> list[Any]:
    import matplotlib

    cmap = matplotlib.colormaps[r.choice(PALETTES)]
    if cmap.N < 32:
        idx = r.sample(range(cmap.N), n) if cmap.N >= n else list(range(n))
        return [cmap(i % cmap.N) for i in idx]
    return [cmap(v) for v in np.linspace(0.05, 0.9, n)]


def curve(
    r: random.Random, n: int, lo: float, hi: float, higher_better: bool, shape: str
) -> np.ndarray:
    t = np.linspace(0, 1, n)
    start = r.uniform(0.0, 0.5)
    end = r.uniform(0.45, 1.0)
    rate = r.uniform(2, 8)
    if shape == "saturate":
        f = start + (end - start) * (1 - np.exp(-rate * t)) / (1 - math.exp(-rate))
    elif shape == "peak":
        peak = r.uniform(0.3, 0.8)
        f = start + (end - start) * np.exp(
            -((t - peak) ** 2) / (2 * r.uniform(0.05, 0.2) ** 2)
        )
        f = np.maximum(f, start + 0.3 * (end - start) * t)
    elif shape == "linear":
        f = start + (end - start) * t
    else:
        f = end - (end - start) * (1 - np.exp(-rate * t)) / (1 - math.exp(-rate))
    f = f + np.array([r.gauss(0, 0.02) for _ in range(n)])
    f = np.clip(f, 0.0, 1.0)
    if not higher_better:
        f = 1 - f
    return lo + (hi - lo) * f


def gap_ok(values: list[float], best: int, margin: float) -> bool:
    others = [v for i, v in enumerate(values) if i != best]
    return all(abs(values[best] - v) >= margin for v in others)


def pick_series(
    values: list[float], best: int, r: random.Random, k: int = 3
) -> list[int]:
    return r.sample([i for i in range(len(values)) if i != best], k)


def setup_axis(ax, r: random.Random, xlabel: str, scale: str, ylabel: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if scale == "log":
        ax.set_xscale("log")
    elif scale == "log2":
        ax.set_xscale("log", base=2)
    if r.random() < 0.5:
        ax.grid(True, alpha=r.uniform(0.2, 0.6), linestyle=r.choice(["-", "--", ":"]))


def line_figure(r: random.Random, difficulty: float) -> Figure | None:
    xlabel, scale, ticks = r.choice(X_AXES)
    n_points = r.randint(5, len(ticks)) if len(ticks) > 5 else len(ticks)
    xs = ticks[:n_points]
    ylabel, metric, lo, hi, higher = r.choice(Y_AXES)
    n_series = r.randint(4, 7)
    names = method_names(r, n_series)
    shapes = [
        r.choice(["saturate", "peak", "linear", "decay"]) for _ in range(n_series)
    ]
    if not higher:
        shapes = [r.choice(["saturate", "linear"]) for _ in range(n_series)]
    ys = [curve(r, n_points, lo, hi, higher, s) for s in shapes]
    span = hi - lo
    margin = span * (0.03 + 0.06 * (1 - difficulty))
    best_word = "highest" if higher else "lowest"
    sign = 1 if higher else -1
    kind = r.choice(
        [
            "argmax_at",
            "argmax_at",
            "improve",
            "peak_x",
            "count_above",
            "read",
            "first_beat",
        ]
    )
    question = correct = None
    distractors: list[str] = []
    xi_label = lambda i: fmt_tick(xs[i], scale)  # noqa: E731
    noun = "method" if names[0] in METHODS or "(" in names[-1] else "setting"
    if kind == "argmax_at":
        j = r.choice([n_points - 1, 0, r.randrange(n_points)])
        vals = [float(y[j]) * sign for y in ys]
        best = int(np.argmax(vals))
        if not gap_ok(vals, best, margin):
            return None
        question = f"Which {noun} achieves the {best_word} {metric} when the {xlabel.split(' (')[0].lower()} is {xi_label(j)}?"
        correct = names[best]
        distractors = [names[i] for i in pick_series(vals, best, r)]
    elif kind == "improve":
        a, b = sorted(r.sample(range(n_points), 2))
        if b - a < 2:
            return None
        gains = [float(y[b] - y[a]) * sign for y in ys]
        best = int(np.argmax(gains))
        if not gap_ok(gains, best, margin):
            return None
        verb = "improvement" if higher else "reduction"
        question = (
            f"Which {noun} shows the largest {verb} in {metric} from "
            f"{xi_label(a)} to {xi_label(b)} on the x-axis?"
        )
        correct = names[best]
        distractors = [names[i] for i in pick_series(gains, best, r)]
    elif kind == "peak_x":
        s = r.randrange(n_series)
        vals = [float(v) * sign for v in ys[s]]
        best = int(np.argmax(vals))
        if not gap_ok(vals, best, margin * 0.7) or n_points < 4:
            return None
        word = "maximum" if higher else "minimum"
        question = f"At which {xlabel.split(' (')[0].lower()} does {names[s]} reach its {word} {metric}?"
        correct = xi_label(best)
        distractors = [xi_label(i) for i in pick_series(vals, best, r)]
    elif kind == "count_above":
        j = r.randrange(n_points)
        vals = sorted(float(y[j]) for y in ys)
        gaps = [(vals[i + 1] - vals[i], i) for i in range(len(vals) - 1)]
        gaps = [g for g in gaps if g[0] >= margin * 1.2]
        if not gaps:
            return None
        _, i = r.choice(gaps)
        threshold = (vals[i] + vals[i + 1]) / 2
        threshold = float(fmt_value(threshold, span))
        if not vals[i] < threshold < vals[i + 1]:
            return None
        count = sum(v > threshold for v in vals)
        pool = [c for c in range(0, n_series + 1) if c != count]
        pool.sort(key=lambda c: (abs(c - count), r.random()))
        question = (
            f"How many {noun}s have {article(metric)} {metric} above {fmt_value(threshold, span)} "
            f"when the {xlabel.split(' (')[0].lower()} is {xi_label(j)}?"
        )
        correct = str(count)
        distractors = [str(c) for c in pool[:3]]
    elif kind == "read":
        j = r.randrange(n_points)
        s = r.randrange(n_series)
        vals = [float(y[j]) for y in ys]
        others = [
            i
            for i in range(n_series)
            if i != s and abs(vals[i] - vals[s]) >= margin * 1.5
        ]
        others = [
            i
            for i in others
            if all(abs(vals[i] - vals[k]) >= margin for k in others if k != i)
        ]
        if len(others) < 3:
            return None
        chosen = r.sample(others, 3)
        texts = [fmt_value(vals[i], span) for i in [s, *chosen]]
        if len(set(texts)) < 4:
            return None
        question = (
            f"What is the approximate {metric} of {names[s]} when the "
            f"{xlabel.split(' (')[0].lower()} is {xi_label(j)}?"
        )
        correct, distractors = texts[0], texts[1:]
    else:
        a, b = r.sample(range(n_series), 2)
        diff = [(float(ys[a][i]) - float(ys[b][i])) * sign for i in range(n_points)]
        first = next(
            (
                i
                for i in range(1, n_points)
                if diff[i] >= margin and all(d < 0 for d in diff[:i])
            ),
            None,
        )
        if first is None or diff[first - 1] > -margin * 0.5:
            return None
        question = f"At which {xlabel.split(' (')[0].lower()} does {names[a]} first outperform {names[b]}?"
        correct = xi_label(first)
        pool = [i for i in range(n_points) if i != first]
        distractors = [xi_label(i) for i in r.sample(pool, 3)]
    if correct is None or len(set([correct, *distractors])) < 4:
        return None
    pal = colors(r, n_series)
    markers = r.sample(MARKERS, n_series)
    errbars = r.random() < 0.35
    legend_loc = r.choice(
        ["best", "lower right", "upper left", "outside", "center right"]
    )
    title = r.choice([None, None, r.choice(DATASETS)])

    def draw(fig):
        ax = fig.add_subplot(111)
        for k in range(n_series):
            style = dict(
                color=pal[k],
                marker=markers[k],
                linestyle=r.choice(LINESTYLES) if k > 3 else "-",
                markersize=r.uniform(3.5, 6),
                linewidth=r.uniform(1.0, 2.0),
                label=names[k],
            )
            if errbars:
                err = (
                    np.abs(np.array([r.gauss(0, span * 0.015) for _ in xs]))
                    + span * 0.005
                )
                ax.errorbar(xs, ys[k], yerr=err, capsize=2, **style)
            else:
                ax.plot(xs, ys[k], **style)
        setup_axis(ax, r, xlabel, scale, ylabel)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                (
                    f"$10^{{{int(round(math.log10(x)))}}}$"
                    if scale == "log" and x >= 1000 and math.log10(x).is_integer()
                    else fmt_tick(x, "lin")
                )
                for x in xs
            ]
        )
        if title:
            ax.set_title(title)
        if legend_loc == "outside":
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize="small",
                frameon=False,
            )
        else:
            ax.legend(loc=legend_loc, fontsize="small", ncol=2 if n_series > 5 else 1)

    width = r.uniform(5.0, 7.0) + (1.6 if legend_loc == "outside" else 0)
    return Figure(
        draw,
        question,
        correct,
        distractors,
        f"line:{kind}",
        (width, r.uniform(3.4, 4.6)),
        r.choice([150, 180, 200]),
        r.choice(STYLES),
    )


def bar_figure(r: random.Random, difficulty: float) -> Figure | None:
    cats = r.choice(CATEGORIES)
    n_cat = r.randint(4, min(8, len(cats)))
    categories = (
        r.sample(cats, n_cat)
        if cats[0] not in ("Easy", "Q1", "Small", "Task 1")
        else cats[:n_cat]
    )
    n_methods = r.randint(3, 5)
    names = method_names(r, n_methods)
    ylabel, metric, lo, hi, higher = r.choice(Y_AXES)
    span = hi - lo
    base = [r.uniform(0.25, 0.85) for _ in range(n_cat)]
    skill = [r.uniform(-0.15, 0.15) for _ in range(n_methods)]
    vals = np.array(
        [
            [
                lo + span * np.clip(base[c] + skill[m] + r.gauss(0, 0.07), 0.03, 1.0)
                for c in range(n_cat)
            ]
            for m in range(n_methods)
        ]
    )
    margin = span * (0.03 + 0.05 * (1 - difficulty))
    sign = 1 if higher else -1
    kind = r.choice(["best_in_cat", "largest_gap", "count_wins", "worst_cat"])
    cat_noun = "category"
    if kind == "best_in_cat":
        c = r.randrange(n_cat)
        v = [float(vals[m, c]) * sign for m in range(n_methods)]
        best = int(np.argmax(v))
        if not gap_ok(v, best, margin) or n_methods < 4:
            return None
        word = "highest" if higher else "lowest"
        question = f"Which method has the {word} {metric} on {categories[c]}?"
        correct, distractors = names[best], [names[i] for i in pick_series(v, best, r)]
    elif kind == "largest_gap":
        a, b = r.sample(range(n_methods), 2)
        g = [abs(float(vals[a, c] - vals[b, c])) for c in range(n_cat)]
        best = int(np.argmax(g))
        if not gap_ok(g, best, margin):
            return None
        question = f"On which {cat_noun} is the difference between {names[a]} and {names[b]} the largest?"
        correct, distractors = categories[best], [
            categories[i] for i in pick_series(g, best, r)
        ]
    elif kind == "count_wins":
        a, b = r.sample(range(n_methods), 2)
        d = [(float(vals[a, c] - vals[b, c])) * sign for c in range(n_cat)]
        if any(abs(x) < margin * 0.6 for x in d):
            return None
        count = sum(x > 0 for x in d)
        pool = sorted(
            [c for c in range(0, n_cat + 1) if c != count],
            key=lambda c: (abs(c - count), r.random()),
        )
        word = "outperform"
        question = f"On how many {cat_noun.replace('y', 'ie')}s does {names[a]} {word} {names[b]}?"
        correct, distractors = str(count), [str(c) for c in pool[:3]]
    else:
        m = r.randrange(n_methods)
        v = [-float(vals[m, c]) * sign for c in range(n_cat)]
        best = int(np.argmax(v))
        if not gap_ok(v, best, margin):
            return None
        question = f"On which {cat_noun} does {names[m]} perform worst?"
        correct, distractors = categories[best], [
            categories[i] for i in pick_series(v, best, r)
        ]
    if len(set([correct, *distractors])) < 4:
        return None
    pal = colors(r, n_methods)
    hatches = r.random() < 0.3
    errs = r.random() < 0.4
    horizontal = r.random() < 0.2

    def draw(fig):
        ax = fig.add_subplot(111)
        width = 0.8 / n_methods
        x = np.arange(n_cat)
        for m in range(n_methods):
            kw = dict(color=pal[m], label=names[m], edgecolor="black", linewidth=0.5)
            if hatches:
                kw["hatch"] = ["", "//", "..", "xx", "\\\\"][m % 5]
            if errs:
                kw["yerr" if not horizontal else "xerr"] = [
                    span * r.uniform(0.005, 0.025) for _ in range(n_cat)
                ]
                kw["capsize"] = 2
            pos = x - 0.4 + width * (m + 0.5)
            if horizontal:
                ax.barh(pos, vals[m], height=width, **kw)
            else:
                ax.bar(pos, vals[m], width=width, **kw)
        if horizontal:
            ax.set_yticks(x)
            ax.set_yticklabels(categories)
            ax.set_xlabel(ylabel)
            ax.set_xlim(lo * 0.9 if lo > 0 else 0, hi * 1.05)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=0 if n_cat <= 6 else 30)
            ax.set_ylabel(ylabel)
            ax.set_ylim(lo * 0.9 if lo > 0 else 0, hi * 1.08)
        ax.legend(
            fontsize="small",
            ncol=min(3, n_methods),
            loc=r.choice(["upper left", "upper right", "best"]),
        )
        if r.random() < 0.5:
            ax.grid(True, axis="x" if horizontal else "y", alpha=0.4)

    return Figure(
        draw,
        question,
        correct,
        distractors,
        f"bar:{kind}",
        (r.uniform(5.5, 8.0), r.uniform(3.2, 4.4)),
        r.choice([150, 180, 200]),
        r.choice(STYLES),
    )


def panels_figure(r: random.Random, difficulty: float) -> Figure | None:
    n_panels = r.choice([3, 4, 4, 6])
    titles = r.sample(DATASETS, n_panels)
    xlabel, scale, ticks = r.choice(X_AXES)
    xs = ticks[: max(5, min(len(ticks), r.randint(5, 7)))]
    ylabel, metric, lo, hi, higher = r.choice(Y_AXES)
    n_series = r.randint(3, 5)
    names = method_names(r, n_series)
    data = [
        [
            curve(
                r,
                len(xs),
                lo + r.uniform(0, 0.2) * (hi - lo),
                hi,
                higher,
                r.choice(["saturate", "peak", "linear"]),
            )
            for _ in range(n_series)
        ]
        for _ in range(n_panels)
    ]
    span = hi - lo
    margin = span * (0.03 + 0.05 * (1 - difficulty))
    sign = 1 if higher else -1
    kind = r.choice(["panel_of_peak", "most_wins"])
    if kind == "panel_of_peak":
        s = r.randrange(n_series)
        peaks = [max(float(v) * sign for v in data[p][s]) for p in range(n_panels)]
        best = int(np.argmax(peaks))
        if not gap_ok(peaks, best, margin) or n_panels < 4:
            return None
        word = "highest" if higher else "lowest"
        question = f"In which subplot does {names[s]} reach its {word} {metric}?"
        correct, distractors = titles[best], [
            titles[i] for i in pick_series(peaks, best, r)
        ]
    else:
        wins = [0] * n_series
        for p in range(n_panels):
            final = [float(data[p][k][-1]) * sign for k in range(n_series)]
            top = int(np.argmax(final))
            if not gap_ok(final, top, margin * 0.7):
                return None
            wins[top] += 1
        best = int(np.argmax(wins))
        if sorted(wins)[-1] == sorted(wins)[-2] or n_series < 4:
            return None
        question = f"Which method has the best final {metric} in the largest number of subplots?"
        correct, distractors = names[best], [
            names[i] for i in pick_series(wins, best, r)
        ]
    if len(set([correct, *distractors])) < 4:
        return None
    pal = colors(r, n_series)
    markers = r.sample(MARKERS, n_series)
    cols = 3 if n_panels in (3, 6) else 2
    rows = math.ceil(n_panels / cols)

    def draw(fig):
        axes = fig.subplots(rows, cols, squeeze=False)
        for p in range(rows * cols):
            ax = axes[p // cols][p % cols]
            if p >= n_panels:
                ax.axis("off")
                continue
            for k in range(n_series):
                ax.plot(
                    xs,
                    data[p][k],
                    color=pal[k],
                    marker=markers[k],
                    markersize=3.5,
                    linewidth=1.3,
                    label=names[k],
                )
            if scale == "log":
                ax.set_xscale("log")
            elif scale == "log2":
                ax.set_xscale("log", base=2)
            ax.set_title(titles[p], fontsize="medium")
            ax.tick_params(labelsize="small")
            if p % cols == 0:
                ax.set_ylabel(ylabel, fontsize="small")
            if p // cols == rows - 1:
                ax.set_xlabel(xlabel, fontsize="small")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=n_series,
            fontsize="small",
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))

    return Figure(
        draw,
        question,
        correct,
        distractors,
        f"panels:{kind}",
        (cols * 3.3, rows * 2.8 + 0.5),
        r.choice([150, 170]),
        r.choice(STYLES),
    )


def scatter_figure(r: random.Random, difficulty: float) -> Figure | None:
    n = r.randint(8, 14)
    labels = r.sample(METHODS, n)
    params = np.exp(
        np.array([r.uniform(math.log(5), math.log(5000)) for _ in range(n)])
    )
    ylabel, metric, lo, hi, higher = r.choice([a for a in Y_AXES if a[4]])
    span = hi - lo
    acc = lo + span * np.clip(
        0.25 + 0.12 * np.log10(params) + np.array([r.gauss(0, 0.12) for _ in range(n)]),
        0.05,
        0.98,
    )
    threshold = r.choice([50, 100, 200, 500, 1000])
    small = [i for i in range(n) if params[i] < threshold]
    large = [i for i in range(n) if params[i] >= threshold * 1.15]
    if (
        len(small) < 3
        or len(large) < 1
        or any(threshold / 1.15 <= params[i] < threshold * 1.15 for i in range(n))
    ):
        return None
    margin = span * (0.03 + 0.05 * (1 - difficulty))
    best = max(small, key=lambda i: acc[i])
    v = [float(acc[i]) for i in small]
    if not gap_ok(v, small.index(best), margin):
        return None
    traps = [i for i in large if acc[i] > acc[best]]
    pool = [i for i in small if i != best]
    distractor_ids = (traps[:1] + r.sample(pool, min(len(pool), 3)))[:3]
    if len(distractor_ids) < 3:
        return None
    question = f"Among the models with fewer than {threshold}M parameters, which one achieves the highest {metric}?"
    correct, distractors = labels[best], [labels[i] for i in distractor_ids]
    pal = colors(r, n)
    show_threshold = r.random() < 0.3

    def draw(fig):
        ax = fig.add_subplot(111)
        ax.scatter(
            params,
            acc,
            c=[pal[i] for i in range(n)],
            s=r.uniform(25, 60),
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        for i in range(n):
            ax.annotate(
                labels[i],
                (params[i], acc[i]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize="x-small",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel(ylabel)
        if show_threshold:
            ax.axvline(threshold, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(True, alpha=0.3)

    return Figure(
        draw,
        question,
        correct,
        distractors,
        "scatter:filter_argmax",
        (r.uniform(5.5, 7.0), r.uniform(4.0, 5.0)),
        r.choice([150, 180]),
        r.choice(STYLES),
    )


def heatmap_figure(r: random.Random, difficulty: float) -> Figure | None:
    rows_ = r.sample(METHODS, r.randint(4, 7))
    cols_ = r.sample(DATASETS, r.randint(4, 7))
    data = np.array([[r.uniform(0.2, 0.95) for _ in cols_] for _ in rows_])
    annotate = r.random() < 0.6
    kind = r.choice(
        ["row_max_in_col", "cell_value"] if annotate else ["row_max_in_col"]
    )
    margin = 0.04 + 0.06 * (1 - difficulty) + (0.0 if annotate else 0.06)
    if kind == "row_max_in_col":
        c = r.randrange(len(cols_))
        v = [float(data[i, c]) for i in range(len(rows_))]
        best = int(np.argmax(v))
        if not gap_ok(v, best, margin):
            return None
        question = f"Which method has the highest value in the {cols_[c]} column?"
        correct, distractors = rows_[best], [rows_[i] for i in pick_series(v, best, r)]
    else:
        i, j = r.randrange(len(rows_)), r.randrange(len(cols_))
        target = f"{data[i, j]:.2f}"
        pool = sorted({f"{x:.2f}" for x in data.flatten()} - {target})
        if len(pool) < 3:
            return None
        distractors = r.sample(pool, 3)
        question = f"What is the value for {rows_[i]} on {cols_[j]}?"
        correct = target
    cmap = r.choice(
        ["viridis", "magma", "Blues", "YlGnBu", "RdYlGn", "coolwarm", "Greens"]
    )

    def draw(fig):
        ax = fig.add_subplot(111)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols_)))
        ax.set_xticklabels(cols_, rotation=35, ha="right", fontsize="small")
        ax.set_yticks(range(len(rows_)))
        ax.set_yticklabels(rows_, fontsize="small")
        if annotate:
            for a in range(len(rows_)):
                for b in range(len(cols_)):
                    ax.text(
                        b,
                        a,
                        f"{data[a, b]:.2f}",
                        ha="center",
                        va="center",
                        fontsize="x-small",
                        color=(
                            "white"
                            if data[a, b] < 0.5 and cmap in ("viridis", "magma")
                            else "black"
                        ),
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return Figure(
        draw,
        question,
        correct,
        distractors,
        f"heatmap:{kind}",
        (r.uniform(5.5, 7.0), r.uniform(4.0, 5.2)),
        r.choice([150, 180]),
        "default",
    )


def box_figure(r: random.Random, difficulty: float) -> Figure | None:
    groups = r.sample(METHODS, r.randint(4, 6))
    ylabel, metric, lo, hi, higher = r.choice(Y_AXES)
    span = hi - lo
    centers = [lo + span * r.uniform(0.25, 0.8) for _ in groups]
    spreads = [span * r.uniform(0.03, 0.15) for _ in groups]
    nrng = np.random.default_rng(r.randrange(2**31))
    samples = [nrng.normal(c, s, 40) for c, s in zip(centers, spreads)]
    kind = r.choice(["median", "spread"])
    margin = span * (0.04 + 0.05 * (1 - difficulty))
    if kind == "median":
        v = [float(np.median(s)) for s in samples]
        word = "highest"
        question = f"Which method has the {word} median {metric}?"
    else:
        v = [float(np.subtract(*np.percentile(s, [75, 25]))) for s in samples]
        margin *= 0.6
        question = "Which method shows the largest interquartile range?"
    best = int(np.argmax(v))
    if not gap_ok(v, best, margin):
        return None
    correct, distractors = groups[best], [groups[i] for i in pick_series(v, best, r)]
    violin = r.random() < 0.35
    pal = colors(r, len(groups))

    def draw(fig):
        ax = fig.add_subplot(111)
        if violin:
            parts = ax.violinplot(samples, showmedians=True)
            for body, color in zip(parts["bodies"], pal):
                body.set_facecolor(color)
                body.set_alpha(0.6)
            ax.set_xticks(range(1, len(groups) + 1))
        else:
            bp = ax.boxplot(samples, patch_artist=True)
            for patch, color in zip(bp["boxes"], pal):
                patch.set_facecolor(color)
        ax.set_xticklabels(
            groups, rotation=20 if len(groups) > 4 else 0, fontsize="small"
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    return Figure(
        draw,
        question,
        correct,
        distractors,
        f"box:{kind}",
        (r.uniform(5.0, 6.8), r.uniform(3.4, 4.4)),
        r.choice([150, 180]),
        r.choice(STYLES),
    )


FAMILIES = [
    (line_figure, 0.42),
    (bar_figure, 0.22),
    (panels_figure, 0.14),
    (scatter_figure, 0.08),
    (heatmap_figure, 0.08),
    (box_figure, 0.06),
]


def make(index: int, seed: int) -> Figure:
    from d25.omni.proxy.rows import rng

    for attempt in range(500):
        r = rng(NAME, seed, index, attempt)
        family = r.choices([f for f, _ in FAMILIES], weights=[w for _, w in FAMILIES])[
            0
        ]
        difficulty = r.random()
        figure = family(r, difficulty)
        if figure is not None:
            figure.meta.update({"difficulty": round(difficulty, 3), "attempt": attempt})
            return figure
    raise RuntimeError(f"no valid figure for item {index}")


def render(figure: Figure) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.style.context(figure.style):
        fig = plt.figure(figsize=figure.size, dpi=figure.dpi)
        try:
            figure.draw(fig)
            buffer = io.BytesIO()
            fig.savefig(
                buffer,
                format="png",
                dpi=figure.dpi,
                bbox_inches="tight",
                metadata={"Software": None},
            )
        finally:
            plt.close(fig)
    return buffer.getvalue()


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from d25.omni.proxy.rows import lettered, rng

    figure = make(index, seed)
    png = render(figure)
    criteria, answer = lettered(
        figure.correct, figure.distractors, rng(NAME, seed, index, "options")
    )
    return {
        "item_id": f"{index:05d}",
        "subtask": figure.subtask,
        "payloads": [(png, "png")],
        "instructions": figure.question,
        "criteria": criteria,
        "answer": answer,
        "provenance": [
            {
                "source": "generated",
                "generator": f"d25.omni.proxy.charts v{VERSION}",
                "seed": seed,
                "index": index,
                "licence": "generated (no third-party content)",
            }
        ],
        "extra": {"difficulty": figure.meta},
    }
