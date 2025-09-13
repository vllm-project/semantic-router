import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps

# This script plots benchmark results from the 3-case vLLM design:
# - VLLM_NR: Plain prompt, no reasoning toggle (baseline)
# - VLLM_XC: CoT prompt, no reasoning toggle (prompt reasoning)
# - VLLM_NR_REASONING: Plain prompt, reasoning toggle ON (model reasoning)
# - router: Router auto mode for comparison

parser = argparse.ArgumentParser()
parser.add_argument(
    "--summary",
    type=Path,
    required=True,
    help="Path to vLLM summary.json produced by the 3-case benchmark",
)
parser.add_argument(
    "--router-summary",
    type=Path,
    required=False,
    help="Optional path to router summary.json to overlay",
)
parser.add_argument(
    "--metrics",
    type=str,
    nargs="+",
    default=["accuracy", "avg_response_time", "avg_total_tokens"],
    choices=["accuracy", "avg_response_time", "avg_total_tokens"],
    help="One or more metrics to plot (default: all)",
)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("."),
    help="Directory to save plots (default: current directory)",
)
parser.add_argument(
    "--font-scale",
    type=float,
    default=1.6,
    help="Scale factor for fonts and markers (default: 1.6)",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=320,
    help="PNG export DPI (default: 320)",
)
parser.add_argument(
    "--style",
    type=str,
    choices=["points", "lines", "both"],
    default="points",
    help="Plot style for modes: points, lines, or both (default: points)",
)
parser.add_argument(
    "--max-modes",
    type=int,
    default=None,
    help="If set, plot only the top N modes by mean of the current metric (default: all 3 modes)",
)
parser.add_argument(
    "--xtick-rotation",
    type=float,
    default=75.0,
    help="Rotation angle for x tick labels (default: 75)",
)
parser.add_argument(
    "--side-margin",
    type=float,
    default=0.0,
    help="Shrink x-limits inward by this many x units per side (default: 0)",
)
parser.add_argument(
    "--side-expand",
    type=float,
    default=0.25,
    help="Expand x-limits outward by this many x units per side (default: 0.25)",
)
args = parser.parse_args()
summary_path = args.summary

with open(summary_path) as f:
    s = json.load(f)

s_router = None
if args.router_summary:
    with open(args.router_summary) as f:
        s_router = json.load(f)


def derive_metrics(summary_json: dict, summary_path: Path):
    cat_by_mode = summary_json.get("category_by_mode")
    cat_ranges = summary_json.get("category_ranges")
    if cat_by_mode is not None and cat_ranges is not None:
        return cat_by_mode, cat_ranges

    csv_path = summary_path.parent / "detailed_results.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing fields in summary and CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df.get("success", True) == True]
    if "mode_label" not in df.columns:
        raise SystemExit(
            "detailed_results.csv lacks 'mode_label' column; cannot compute per-mode stats"
        )

    grouped = (
        df.groupby(["category", "mode_label"]).agg(
            accuracy=("is_correct", "mean"),
            avg_response_time=("response_time", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_completion_tokens=("completion_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
        )
    ).reset_index()

    cat_by_mode = {}
    cat_ranges = {}
    for cat in grouped["category"].unique():
        sub = grouped[grouped["category"] == cat]
        modes = {}
        for _, row in sub.iterrows():
            modes[str(row["mode_label"])] = {
                "accuracy": (
                    float(row["accuracy"]) if pd.notna(row["accuracy"]) else 0.0
                ),
                "avg_response_time": (
                    float(row["avg_response_time"])
                    if pd.notna(row["avg_response_time"])
                    else 0.0
                ),
                "avg_prompt_tokens": (
                    float(row["avg_prompt_tokens"])
                    if pd.notna(row["avg_prompt_tokens"])
                    else None
                ),
                "avg_completion_tokens": (
                    float(row["avg_completion_tokens"])
                    if pd.notna(row["avg_completion_tokens"])
                    else None
                ),
                "avg_total_tokens": (
                    float(row["avg_total_tokens"])
                    if pd.notna(row["avg_total_tokens"])
                    else None
                ),
            }
        cat_by_mode[cat] = modes

        # ranges
        def _mm(values):
            values = [v for v in values if v is not None]
            if not values:
                return {"min": 0.0, "max": 0.0}
            return {"min": float(min(values)), "max": float(max(values))}

        acc_vals = [v.get("accuracy") for v in modes.values()]
        lat_vals = [v.get("avg_response_time") for v in modes.values()]
        tok_vals = [v.get("avg_total_tokens") for v in modes.values()]
        cat_ranges[cat] = {
            "accuracy": _mm(acc_vals),
            "avg_response_time": _mm(lat_vals),
            "avg_total_tokens": _mm(tok_vals),
        }
    return cat_by_mode, cat_ranges


cat_by_mode, cat_ranges = derive_metrics(s, summary_path)

cats = sorted(cat_ranges.keys())


def plot_metric(metric: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(18, 8))

    x = range(len(cats))

    # Plot router per-category metric FIRST (with both line and diamonds)
    # This ensures router trend is visible even if vLLM dots overlap
    if s_router is not None:
        router_cat = s_router.get("category_metrics", {})
        router_vals = []
        router_x = []
        for idx, c in enumerate(cats):
            v = router_cat.get(c, {}).get(metric)
            if v is not None:
                router_x.append(idx)
                router_vals.append(v)
        if router_vals:
            # Connect router points with a line and draw larger diamond markers
            ax.plot(
                router_x,
                router_vals,
                color="tab:red",
                linestyle="-",
                linewidth=2.0 * args.font_scale,
                alpha=0.85,
                zorder=1,  # Lower zorder so it's plotted first
            )
            ax.scatter(
                router_x,
                router_vals,
                s=90 * args.font_scale,
                color="tab:red",
                marker="D",
                label="router",
                zorder=2,  # Lower zorder so it's plotted first
                edgecolors="white",
                linewidths=0.6 * args.font_scale,
            )

    # Then plot vLLM modes on top
    all_modes = sorted({m for c in cats for m in cat_by_mode.get(c, {}).keys()})
    if len(all_modes) > 0:

        def _mean(values):
            vals = [v for v in values if v is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        if (
            args.max_modes is not None
            and args.max_modes > 0
            and len(all_modes) > args.max_modes
        ):
            mode_means = []
            for mode in all_modes:
                vals = [cat_by_mode.get(c, {}).get(mode, {}).get(metric) for c in cats]
                mode_means.append((mode, _mean(vals)))
            # Accuracy: higher is better; latency/tokens: lower is better
            ascending = metric != "accuracy"
            mode_means = sorted(
                mode_means,
                key=lambda kv: (float("inf") if (kv[1] != kv[1]) else kv[1]),
                reverse=not ascending,
            )
            all_modes = [m for m, _ in mode_means[: args.max_modes]]

        palette = colormaps.get_cmap("tab10").resampled(len(all_modes))
        linestyles = ["-", "--", "-.", ":"]
        for i, mode in enumerate(all_modes):
            ys = [cat_by_mode.get(c, {}).get(mode, {}).get(metric) for c in cats]
            if args.style in ("lines", "both"):
                ax.plot(
                    x,
                    ys,
                    color=palette.colors[i],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.4 * args.font_scale,
                    alpha=0.6,
                    zorder=3,  # Higher zorder so vLLM lines are on top
                )
            if args.style in ("points", "both"):
                ax.scatter(
                    x,
                    ys,
                    s=60 * args.font_scale,
                    color=palette.colors[i],
                    label=mode,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.5 * args.font_scale,
                    zorder=4,  # Higher zorder so vLLM points are on top
                )

    # Set x-axis labels with threshold for readability
    MAX_CATEGORY_LABELS = 20  # Hide labels if more than this many categories

    ax.set_xticks(list(x))
    if len(cats) <= MAX_CATEGORY_LABELS:
        ax.set_xticklabels(
            cats,
            rotation=args.xtick_rotation,
            ha="right",
            fontsize=int(14 * args.font_scale),
        )
    else:
        # Too many categories - hide labels to avoid clutter
        ax.set_xticklabels([])
        ax.set_xlabel(
            f"Categories ({len(cats)} total - labels hidden for readability)",
            fontsize=int(16 * args.font_scale),
        )
    # Control horizontal fit by expanding/shrinking x-limits around the first/last category
    if len(cats) > 0:
        n = len(cats)
        # Base categorical extents
        base_left = -0.5
        base_right = n - 0.5
        # Apply outward expansion first, then inward margin
        expand = max(0.0, float(args.side_expand))
        max_margin = 0.49
        margin = max(0.0, min(float(args.side_margin), max_margin))
        left_xlim = base_left - expand + margin
        right_xlim = base_right + expand - margin
        if right_xlim > left_xlim:
            ax.set_xlim(left_xlim, right_xlim)
    ylabel = metric.replace("_", " ")
    ax.set_ylabel(ylabel, fontsize=int(18 * args.font_scale))
    ax.set_title(
        f"Per-category {ylabel} per-mode values", fontsize=int(22 * args.font_scale)
    )
    ax.tick_params(axis="both", which="major", labelsize=int(14 * args.font_scale))

    # Build a figure-level legend below the axes and reserve space to prevent overlap
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        num_series = len(handles)
        # Force exactly 2 legend rows; compute columns accordingly
        legend_rows = 2
        legend_ncol = max(1, (num_series + legend_rows - 1) // legend_rows)
        num_rows = legend_rows
        scale = args.font_scale / 1.6
        # Reserve generous space for long rotated tick labels and multi-row legend
        bottom_reserved = (0.28 * scale) + (0.12 * num_rows * scale)
        bottom_reserved = max(0.24, min(0.60, bottom_reserved))
        fig.subplots_adjust(left=0.01, right=0.999, top=0.92, bottom=bottom_reserved)
        # Align the legend box width with the axes width
        pos = ax.get_position()
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(pos.x0, 0.02, pos.width, 0.001),
            bbox_transform=fig.transFigure,
            ncol=legend_ncol,
            mode="expand",
            fontsize=int(14 * args.font_scale),
            markerscale=1.6 * args.font_scale,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=0.8 * args.font_scale,
            handlelength=2.2,
        )
    else:
        fig.subplots_adjust(left=0.01, right=0.999, top=0.92, bottom=0.14)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.3)
    # Eliminate additional automatic horizontal padding
    ax.margins(x=0.0)
    # Layout handled via subplots_adjust above to avoid legend overlap
    # Save both PNG and PDF variants
    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")
    plt.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    """Main entry point for the plotting script."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for metric in args.metrics:
        out_path = args.out_dir / f"bench_plot_{metric}.png"
        plot_metric(metric, out_path)


if __name__ == "__main__":
    main()
