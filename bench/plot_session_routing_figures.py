#!/usr/bin/env python3
"""Render session-routing experiment figures from benchmark CSV outputs.

The deterministic experiment generator intentionally stays standard-library
only so it can run on small validation hosts. This companion script handles the
publication figures with matplotlib when a plotting environment is available.
It never synthesizes numbers: every plotted value comes from the maintained CSV
outputs produced by ``bench/agentic_routing_experiment.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

SCENARIO_ORDER = (
    "balanced",
    "tool-heavy",
    "frontier-heavy",
    "idle-heavy",
    "stateful-heavy",
    "drift-heavy",
)
POLICY_ORDER = (
    "single-turn",
    "sticky-session",
    "acr-initial",
    "acr-no-tool-lock",
    "acr-no-context-portability",
    "acr-no-decision-drift-reset",
    "acr-no-idle-boundary",
    "acr-no-remaining-prior",
    "acr-no-frontier-cost",
    "acr-full",
)
LABELS = {
    "acr-initial": "initial ACR",
    "acr-no-tool-lock": "no tool lock",
    "acr-no-context-portability": "no provider-state lock",
    "acr-no-decision-drift-reset": "no drift reset",
    "acr-no-idle-boundary": "no idle boundary",
    "acr-no-remaining-prior": "no remaining prior",
    "acr-no-frontier-cost": "no frontier cost",
    "acr-full": "full ACR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-summary", required=True, type=Path)
    parser.add_argument("--scenario-seed-summary", required=True, type=Path)
    parser.add_argument("--ablation-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: parse_cell(value) for key, value in row.items()} for row in rows]


def parse_cell(value: str) -> Any:
    if value == "":
        return ""
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def ordered(
    rows: list[dict[str, Any]], key: str, preferred: tuple[str, ...]
) -> list[dict[str, Any]]:
    index = {name: offset for offset, name in enumerate(preferred)}
    return sorted(rows, key=lambda row: index.get(str(row[key]), len(index)))


def load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.ticker import FuncFormatter  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised without matplotlib.
        raise SystemExit(
            "matplotlib is required for publication figures. "
            "Install bench plotting dependencies with `pip install -r bench/requirements.txt` "
            "or run through `uv run --with matplotlib`."
        ) from exc
    return plt, FuncFormatter


def pct_formatter(func_formatter: Any) -> Any:
    return func_formatter(lambda value, _pos: f"{value:.0f}%")


def apply_research_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d7e0e7",
            "axes.labelcolor": "#2f3e4d",
            "xtick.color": "#6b7b8c",
            "ytick.color": "#2f3e4d",
            "grid.color": "#e7edf2",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.titlecolor": "#172033",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)


def render_matrix(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    plt, func_formatter = load_pyplot()
    apply_research_style(plt)
    rows = ordered(rows, "scenario", SCENARIO_ORDER)
    scenarios = [str(row["scenario"]) for row in rows]
    y_positions = range(len(rows))

    fig, axes = plt.subplots(
        1, 3, figsize=(13.2, 6.3), gridspec_kw={"width_ratios": [1.2, 1.0, 1.1]}
    )
    fig.suptitle(
        "Session-aware routing matrix",
        x=0.02,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.92,
        "21,600 deterministic turns; bars compare full ACR with the single-turn baseline",
        color="#6b7b8c",
        fontsize=10,
    )

    switch_ax, cost_ax, violation_ax = axes
    switch_values = [float(row["switch_reduction_pct"]) for row in rows]
    cost_values = [float(row["cost_reduction_pct"]) for row in rows]
    tool_values = [float(row["baseline_tool_loop_switch_violations"]) for row in rows]
    context_values = [
        float(row["baseline_context_portability_violations"]) for row in rows
    ]

    switch_ax.barh(y_positions, switch_values, color="#6aa6c8", height=0.58)
    switch_ax.set_title("Switch reduction")
    switch_ax.set_xlim(0, 100)
    switch_ax.set_yticks(list(y_positions), scenarios)
    switch_ax.xaxis.set_major_formatter(pct_formatter(func_formatter))
    switch_ax.grid(axis="x")
    for idx, row in enumerate(rows):
        switch_ax.text(
            switch_values[idx] + 1.4,
            idx,
            f'{row["baseline_switches"]} -> {row["agentic_switches"]}',
            va="center",
            color="#475569",
            fontsize=9,
        )

    cost_ax.barh(y_positions, cost_values, color="#b9ddf0", height=0.58)
    cost_ax.set_title("Estimated cost reduction")
    cost_ax.set_xlim(0, 100)
    cost_ax.set_yticks(list(y_positions), [])
    cost_ax.xaxis.set_major_formatter(pct_formatter(func_formatter))
    cost_ax.grid(axis="x")
    for idx, row in enumerate(rows):
        cost_ax.text(
            cost_values[idx] + 1.4,
            idx,
            f'${float(row["baseline_cost"]):.1f} -> ${float(row["agentic_cost"]):.1f}',
            va="center",
            color="#475569",
            fontsize=9,
        )

    width = 0.36
    violation_ax.barh(
        [pos - width / 2 for pos in y_positions],
        tool_values,
        color="#c7b7e6",
        height=width,
        label="tool-loop baseline",
    )
    violation_ax.barh(
        [pos + width / 2 for pos in y_positions],
        context_values,
        color="#455e70",
        height=width,
        label="provider-state baseline",
    )
    violation_ax.set_title("Unsafe baseline switches")
    violation_ax.set_yticks(list(y_positions), [])
    violation_ax.grid(axis="x")
    violation_ax.legend(frameon=False, fontsize=8, loc="lower right")
    for idx, row in enumerate(rows):
        total = int(row["baseline_tool_loop_switch_violations"]) + int(
            row["baseline_context_portability_violations"]
        )
        violation_ax.text(
            total + 8, idx, "ACR: 0", va="center", color="#16a34a", fontsize=9
        )

    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, output, dpi)
    plt.close(fig)


def render_ablation(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    plt, func_formatter = load_pyplot()
    apply_research_style(plt)
    rows = ordered(rows, "policy", POLICY_ORDER)
    policies = [LABELS.get(str(row["policy"]), str(row["policy"])) for row in rows]
    y_positions = range(len(rows))
    colors = ["#455e70" if row["policy"] == "acr-full" else "#9ed0ea" for row in rows]
    colors[0] = "#98a6b5"
    if len(colors) > 1:
        colors[1] = "#d7e0e7"

    fig, axes = plt.subplots(
        1, 3, figsize=(13.6, 7.2), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]}
    )
    fig.suptitle("Policy ablation", x=0.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.92,
        "The initial ACR baseline models the merged #1974 capability boundary",
        color="#6b7b8c",
        fontsize=10,
    )

    switch_ax, cost_ax, violation_ax = axes
    switches = [float(row["switches"]) for row in rows]
    cost_reduction = [float(row["cost_reduction_pct"]) for row in rows]
    tool_violations = [float(row["tool_loop_switch_violations"]) for row in rows]
    context_violations = [float(row["context_portability_violations"]) for row in rows]

    switch_ax.barh(y_positions, switches, color=colors, height=0.56)
    switch_ax.set_title("Model switches")
    switch_ax.set_yticks(list(y_positions), policies)
    switch_ax.grid(axis="x")
    for idx, value in enumerate(switches):
        switch_ax.text(
            value + 70, idx, f"{int(value):,}", va="center", color="#475569", fontsize=9
        )

    cost_ax.barh(y_positions, cost_reduction, color=colors, height=0.56)
    cost_ax.set_title("Cost reduction")
    cost_ax.set_xlim(0, 100)
    cost_ax.set_yticks(list(y_positions), [])
    cost_ax.xaxis.set_major_formatter(pct_formatter(func_formatter))
    cost_ax.grid(axis="x")
    for idx, value in enumerate(cost_reduction):
        cost_ax.text(
            value + 1.2, idx, f"{value:.1f}%", va="center", color="#475569", fontsize=9
        )

    width = 0.36
    violation_ax.barh(
        [pos - width / 2 for pos in y_positions],
        tool_violations,
        color="#c7b7e6",
        height=width,
        label="tool-loop",
    )
    violation_ax.barh(
        [pos + width / 2 for pos in y_positions],
        context_violations,
        color="#455e70",
        height=width,
        label="provider-state",
    )
    violation_ax.set_title("Unsafe switches")
    violation_ax.set_yticks(list(y_positions), [])
    violation_ax.grid(axis="x")
    violation_ax.legend(frameon=False, fontsize=8, loc="lower right")
    for idx, row in enumerate(rows):
        if row["policy"] in {"acr-initial", "acr-full"}:
            violation_ax.text(
                225,
                idx,
                f'quality {float(row["quality_delta"]):+.4f}',
                va="center",
                color="#475569",
                fontsize=9,
            )

    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, output, dpi)
    plt.close(fig)


def render_seed_stability(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    plt, func_formatter = load_pyplot()
    apply_research_style(plt)
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_ORDER}
    for row in rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)

    scenarios = [name for name in SCENARIO_ORDER if grouped.get(name)]
    y_positions = range(len(scenarios))
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    fig.suptitle("Seed stability", x=0.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.02,
        0.92,
        "Five seeds per workload; dots are switch-reduction observations and bars show min-to-max spread",
        color="#6b7b8c",
        fontsize=10,
    )

    for idx, scenario in enumerate(scenarios):
        values = [float(row["switch_reduction_pct"]) for row in grouped[scenario]]
        qualities = [float(row["quality_delta"]) for row in grouped[scenario]]
        ax.hlines(
            idx, min(values), max(values), color="#98a6b5", linewidth=7, alpha=0.65
        )
        ax.scatter(
            values,
            [idx] * len(values),
            s=52,
            color="#6aa6c8",
            edgecolor="white",
            zorder=3,
        )
        ax.scatter(
            [sum(values) / len(values)],
            [idx],
            s=42,
            color="#455e70",
            edgecolor="white",
            zorder=4,
        )
        ax.text(
            101.5,
            idx,
            f"{min(qualities):+.4f} to {max(qualities):+.4f}",
            va="center",
            color="#475569",
            fontsize=9,
        )

    ax.set_yticks(list(y_positions), scenarios)
    ax.set_xlim(0, 118)
    ax.xaxis.set_major_formatter(pct_formatter(func_formatter))
    ax.set_xlabel("switch reduction")
    ax.text(101.5, -0.65, "quality delta range", color="#6b7b8c", fontsize=10)
    ax.grid(axis="x")
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, output, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scenario_summary = read_csv(args.scenario_summary)
    scenario_seed_summary = read_csv(args.scenario_seed_summary)
    ablation_summary = read_csv(args.ablation_summary)

    render_matrix(scenario_summary, args.output_dir / "experiment-matrix.png", args.dpi)
    render_ablation(ablation_summary, args.output_dir / "policy-ablation.png", args.dpi)
    render_seed_stability(
        scenario_seed_summary, args.output_dir / "seed-stability.png", args.dpi
    )

    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
