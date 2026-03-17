from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PLOT_COLORS = {"standard": "#3498db", "reasoning": "#e74c3c"}


def generate_comparison_plots(comparison: Any, output_dir: Path) -> None:
    """Generate visualization plots comparing standard vs reasoning modes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    _plot_overall_metrics(comparison, output_dir)
    if comparison.category_results:
        _plot_category_accuracy(comparison, output_dir)
        _plot_token_efficiency(comparison, output_dir)
        _plot_time_per_token(comparison, output_dir)
    print(f"  ✅ Plots saved to {output_dir}")


def _plot_overall_metrics(comparison: Any, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = [
        ("accuracy", "Accuracy", "higher is better"),
        ("token_usage_ratio", "Token Usage Ratio\n(completion/prompt)", "efficiency"),
        ("time_per_output_token", "Time per Output Token (ms)", "lower is better"),
        ("avg_response_time", "Avg Response Time (s)", "lower is better"),
    ]

    for ax, (metric_key, metric_label, note) in zip(
        axes.flatten(), metrics_to_plot, strict=True
    ):
        values = [
            getattr(comparison.standard_mode, metric_key, 0),
            getattr(comparison.reasoning_mode, metric_key, 0),
        ]
        bars = ax.bar(
            ["Standard", "Reasoning"],
            values,
            color=[PLOT_COLORS["standard"], PLOT_COLORS["reasoning"]],
            edgecolor="white",
            linewidth=2,
        )
        _annotate_bars(ax, bars, values)
        ax.set_title(f"{metric_label}\n({note})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")

    fig.suptitle(
        f"Standard vs Reasoning Mode: {comparison.dataset_name}\nModel: {comparison.model_name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    _save_plot(fig, output_dir / f"{comparison.dataset_name}_overall_comparison.png")


def _plot_category_accuracy(comparison: Any, output_dir: Path) -> None:
    categories = list(comparison.category_results.keys())
    std_accs = [comparison.category_results[c]["standard"].accuracy for c in categories]
    reas_accs = [
        comparison.category_results[c]["reasoning"].accuracy for c in categories
    ]
    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.5), 8))
    _plot_category_bars(
        ax,
        categories,
        std_accs,
        reas_accs,
        ylabel="Accuracy",
        title=f"Per-Category Accuracy: Standard vs Reasoning\n{comparison.dataset_name}",
    )
    ax.set_ylim(0, 1.1)
    _save_plot(fig, output_dir / f"{comparison.dataset_name}_category_accuracy.png")


def _plot_token_efficiency(comparison: Any, output_dir: Path) -> None:
    categories = list(comparison.category_results.keys())
    std_ratios = [
        comparison.category_results[c]["standard"].token_usage_ratio for c in categories
    ]
    reas_ratios = [
        comparison.category_results[c]["reasoning"].token_usage_ratio
        for c in categories
    ]
    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.5), 8))
    _plot_category_bars(
        ax,
        categories,
        std_ratios,
        reas_ratios,
        ylabel="Token Usage Ratio (completion/prompt)",
        title=f"Token Usage Ratio by Category\n{comparison.dataset_name}",
    )
    _save_plot(fig, output_dir / f"{comparison.dataset_name}_token_usage_ratio.png")


def _plot_time_per_token(comparison: Any, output_dir: Path) -> None:
    categories = list(comparison.category_results.keys())
    std_times = [
        comparison.category_results[c]["standard"].time_per_output_token
        for c in categories
    ]
    reas_times = [
        comparison.category_results[c]["reasoning"].time_per_output_token
        for c in categories
    ]
    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.5), 8))
    _plot_category_bars(
        ax,
        categories,
        std_times,
        reas_times,
        ylabel="Time per Output Token (ms)",
        title=f"Response Time per Output Token by Category\n{comparison.dataset_name}",
    )
    _save_plot(fig, output_dir / f"{comparison.dataset_name}_time_per_token.png")


def _plot_category_bars(
    ax: Any,
    categories: list[str],
    standard_values: list[float],
    reasoning_values: list[float],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(
        x - width / 2,
        standard_values,
        width,
        label="Standard",
        color=PLOT_COLORS["standard"],
    )
    ax.bar(
        x + width / 2,
        reasoning_values,
        width,
        label="Reasoning",
        color=PLOT_COLORS["reasoning"],
    )
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right")


def _annotate_bars(ax: Any, bars: Any, values: list[float]) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def _save_plot(fig: Any, path: Path) -> None:
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
