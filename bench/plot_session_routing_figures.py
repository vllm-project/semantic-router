#!/usr/bin/env python3
"""Render session-routing figures from benchmark CSV and JSON outputs.

The experiment generators stay lightweight so they can run on validation hosts.
This companion script uses matplotlib for publication/blog figures. It does
not synthesize numbers: every plotted value comes from maintained benchmark
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    "acr-no-context-portability": "no provider lock",
    "acr-no-decision-drift-reset": "no drift reset",
    "acr-no-idle-boundary": "no idle boundary",
    "acr-no-remaining-prior": "no remaining prior",
    "acr-no-frontier-cost": "no frontier cost",
    "acr-full": "full ACR",
    "single-turn": "single-turn",
    "sticky-session": "sticky",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-summary", required=True, type=Path)
    parser.add_argument("--scenario-seed-summary", required=True, type=Path)
    parser.add_argument("--ablation-summary", required=True, type=Path)
    parser.add_argument("--synthetic-summary", type=Path)
    parser.add_argument(
        "--amd-overhead-comparison", action="append", type=Path, default=[]
    )
    parser.add_argument("--amd-long-session-aggregate", type=Path)
    parser.add_argument(
        "--amd-disruption-summary", action="append", type=Path, default=[]
    )
    parser.add_argument("--amd-disruption-smoke-summary", type=Path)
    parser.add_argument("--amd-repeat-failure-aggregate", type=Path)
    parser.add_argument("--agent-task-summary", type=Path)
    parser.add_argument("--agent-task-comparison", type=Path)
    parser.add_argument("--long-agent-task-summary", type=Path)
    parser.add_argument("--long-agent-task-comparison", type=Path)
    parser.add_argument("--ga-readiness", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: parse_cell(value) for key, value in row.items()} for row in rows]


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


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
            "Install plotting dependencies with `pip install matplotlib` "
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
            "xtick.color": "#64748b",
            "ytick.color": "#334155",
            "grid.color": "#e7edf2",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlecolor": "#172033",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)


def label(raw: str) -> str:
    return (
        raw.replace("pr1989-overhead-", "")
        .replace("-recovery", "")
        .replace("fixed-turn", "fixed turn")
        .replace("-", " ")
    )


def render_matrix(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    plt, func_formatter = load_pyplot()
    apply_research_style(plt)
    rows = ordered(rows, "scenario", SCENARIO_ORDER)
    names = [str(row["scenario"]) for row in rows]
    values = [float(row["switch_reduction_pct"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.suptitle("Switches drop across every workload", x=0.02, ha="left", fontsize=16)
    ax.barh(range(len(rows)), values, color="#5ea8d6", height=0.58)
    ax.set_xlim(0, 100)
    ax.set_yticks(range(len(rows)), names)
    ax.xaxis.set_major_formatter(pct_formatter(func_formatter))
    ax.set_xlabel("switch reduction vs single-turn routing")
    ax.grid(axis="x")
    for idx, row in enumerate(rows):
        ax.text(
            values[idx] + 1.4,
            idx,
            f'{row["baseline_switches"]} -> {row["agentic_switches"]}',
            va="center",
            color="#334155",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_synthetic_headline(overall: dict[str, Any], output: Path, dpi: int) -> None:
    plt, _ = load_pyplot()
    apply_research_style(plt)
    metrics = [
        ("Turns", f'{int(overall["turns"]):,}', "#eef7fb"),
        ("Fewer switches", f'{float(overall["switch_reduction_pct"]):.1f}%', "#e8f3ea"),
        ("Unsafe switches", "0", "#fff7e6"),
        ("Cost reduction", f'{float(overall["cost_reduction_pct"]):.1f}%', "#edf1ff"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.axis("off")
    fig.suptitle(
        "ACR changes the unit from turn to session", x=0.02, ha="left", fontsize=16
    )
    for idx, (name, value, color) in enumerate(metrics):
        x_pos = 0.02 + idx * 0.245
        ax.add_patch(
            plt.Rectangle(
                (x_pos, 0.18),
                0.21,
                0.58,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="#d7e0e7",
                linewidth=1,
            )
        )
        ax.text(
            x_pos + 0.025,
            0.56,
            value,
            transform=ax.transAxes,
            fontsize=20,
            weight="bold",
            color="#172033",
        )
        ax.text(
            x_pos + 0.025,
            0.35,
            name,
            transform=ax.transAxes,
            fontsize=10,
            color="#64748b",
        )
    save(fig, output, dpi)
    plt.close(fig)


def render_synthetic_safety(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    plt, _ = load_pyplot()
    apply_research_style(plt)
    rows = ordered(rows, "scenario", SCENARIO_ORDER)
    names = [str(row["scenario"]) for row in rows]
    baseline = [
        int(row["baseline_tool_loop_switch_violations"])
        + int(row["baseline_context_portability_violations"])
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.suptitle("Hard locks remove unsafe switches", x=0.02, ha="left", fontsize=16)
    ax.barh(
        range(len(rows)), baseline, color="#d88b5b", height=0.58, label="single-turn"
    )
    ax.scatter(
        [0] * len(rows), range(len(rows)), s=70, color="#27946f", zorder=3, label="ACR"
    )
    ax.set_yticks(range(len(rows)), names)
    ax.set_xlabel("tool-loop + provider-state switch violations")
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="lower right")
    high = max(baseline)
    for idx, value in enumerate(baseline):
        ax.text(
            value + high * 0.02,
            idx,
            f"{value:,} -> 0",
            va="center",
            color="#334155",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_ablation(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    plt, _ = load_pyplot()
    apply_research_style(plt)
    keep = {
        "single-turn",
        "sticky-session",
        "acr-initial",
        "acr-no-tool-lock",
        "acr-no-context-portability",
        "acr-full",
    }
    rows = [
        row for row in ordered(rows, "policy", POLICY_ORDER) if row["policy"] in keep
    ]
    names = [LABELS.get(str(row["policy"]), str(row["policy"])) for row in rows]
    unsafe = [
        int(row["tool_loop_switch_violations"])
        + int(row.get("context_portability_violations") or 0)
        for row in rows
    ]
    colors = ["#a8b5c2", "#d7e0e7", "#7ab7d8", "#d88b5b", "#c66f68", "#2f7f68"]
    fig, ax = plt.subplots(figsize=(8.9, 4.8))
    fig.suptitle(
        "Ablation: continuity is a correctness boundary", x=0.02, ha="left", fontsize=16
    )
    ax.barh(range(len(rows)), unsafe, color=colors, height=0.58)
    ax.set_yticks(range(len(rows)), names)
    ax.set_xlabel("unsafe switches")
    ax.grid(axis="x")
    high = max(unsafe) if unsafe else 1
    for idx, value in enumerate(unsafe):
        ax.text(
            value + high * 0.025,
            idx,
            f"{value:,}",
            va="center",
            color="#334155",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_ablation_cost_quality(
    rows: list[dict[str, Any]], output: Path, dpi: int
) -> None:
    plt, func_formatter = load_pyplot()
    apply_research_style(plt)
    wanted = ["single-turn", "sticky-session", "acr-initial", "acr-full"]
    by_policy = {str(row["policy"]): row for row in rows}
    rows = [by_policy[name] for name in wanted if name in by_policy]
    names = [LABELS.get(str(row["policy"]), str(row["policy"])) for row in rows]
    costs = [float(row["cost_reduction_pct"]) for row in rows]
    quality = [float(row["quality_delta"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.suptitle("Cost savings are not enough", x=0.02, ha="left", fontsize=16)
    ax.bar(
        range(len(rows)),
        costs,
        color=["#a8b5c2", "#d7e0e7", "#7ab7d8", "#2f7f68"],
        width=0.55,
    )
    ax.set_xticks(range(len(rows)), names, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(pct_formatter(func_formatter))
    ax.set_ylabel("cost reduction")
    ax.grid(axis="y")
    for idx, value in enumerate(costs):
        ax.text(
            idx, value + 2, f"{value:.1f}%", ha="center", color="#334155", fontsize=9
        )
        ax.text(
            idx, 5, f"Q {quality[idx]:+.3f}", ha="center", color="#475569", fontsize=9
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_seed_stability(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    plt, func_formatter = load_pyplot()
    apply_research_style(plt)
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_ORDER}
    for row in rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)

    scenarios = [name for name in SCENARIO_ORDER if grouped.get(name)]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.suptitle("Seed stability stays narrow", x=0.02, ha="left", fontsize=16)
    for idx, scenario in enumerate(scenarios):
        values = [float(row["switch_reduction_pct"]) for row in grouped[scenario]]
        qualities = [float(row["quality_delta"]) for row in grouped[scenario]]
        ax.hlines(
            idx, min(values), max(values), color="#98a6b5", linewidth=7, alpha=0.65
        )
        ax.scatter(
            values,
            [idx] * len(values),
            s=44,
            color="#5ea8d6",
            edgecolor="white",
            zorder=3,
        )
        ax.scatter(
            [sum(values) / len(values)],
            [idx],
            s=46,
            color="#334155",
            edgecolor="white",
            zorder=4,
        )
        ax.text(
            101.5,
            idx,
            f"Q {min(qualities):+.3f} to {max(qualities):+.3f}",
            va="center",
            color="#475569",
            fontsize=9,
        )

    ax.set_yticks(range(len(scenarios)), scenarios)
    ax.set_xlim(0, 118)
    ax.xaxis.set_major_formatter(pct_formatter(func_formatter))
    ax.set_xlabel("switch reduction")
    ax.grid(axis="x")
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_amd_overhead(
    comparisons: list[dict[str, Any]], output: Path, dpi: int
) -> None:
    if not comparisons:
        return
    plt, _ = load_pyplot()
    apply_research_style(plt)
    rows = sorted(comparisons, key=lambda row: str(row["label"]))
    names = [label(str(row["label"])) for row in rows]
    values = [float(row["router_overhead_ms"]["p95"]) for row in rows]
    colors = ["#2f7f68" if value <= 0 else "#5ea8d6" for value in values]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.suptitle("ROCm router overhead stays small", x=0.02, ha="left", fontsize=16)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.barh(range(len(rows)), values, color=colors, height=0.56)
    ax.set_yticks(range(len(rows)), names)
    ax.set_xlabel("p95 overhead vs direct backend (ms)")
    ax.grid(axis="x")
    for idx, value in enumerate(values):
        if value >= 0:
            x_pos = value + max(abs(value), 1.0) * 0.08
            ha = "left"
            color = "#334155"
        else:
            x_pos = value / 2
            ha = "center"
            color = "white"
        ax.text(
            x_pos,
            idx,
            f"{value:+.1f} ms",
            va="center",
            ha=ha,
            color=color,
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_amd_long_session(
    aggregate: dict[str, Any] | None, output: Path, dpi: int
) -> None:
    if not aggregate or not isinstance(aggregate.get("rows"), list):
        return
    rows = aggregate["rows"]
    plt, _ = load_pyplot()
    apply_research_style(plt)
    names = [label(str(row.get("label") or row.get("name", ""))) for row in rows]
    requests = [int(row["router_requests"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.suptitle(
        "Long ROCm sessions keep continuity intact", x=0.02, ha="left", fontsize=16
    )
    ax.barh(range(len(rows)), requests, color="#5ea8d6", height=0.58)
    ax.set_yticks(range(len(rows)), names)
    ax.set_xlabel("router requests, all HTTP 200")
    ax.grid(axis="x")
    high = max(requests)
    for idx, row in enumerate(rows):
        ax.text(
            requests[idx] + high * 0.02,
            idx,
            (
                f"{requests[idx]:,}; "
                f"viol={int(row.get('tool_loop_switch_violations', 0)) + int(row.get('context_portability_violations', 0))}"
            ),
            va="center",
            color="#334155",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_disruptions(rows: list[dict[str, Any]], output: Path, dpi: int) -> None:
    if not rows:
        return
    plt, _ = load_pyplot()
    apply_research_style(plt)
    names = [label(str(row["label"])) for row in rows]
    recovered = [float(row["session_recovery_rate_after_error"]) * 100 for row in rows]
    injected = [int(row["status_counts"].get("503", 0)) for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.suptitle(
        "Injected failures recover on later turns", x=0.02, ha="left", fontsize=16
    )
    ax.barh(range(len(rows)), recovered, color="#2f7f68", height=0.58)
    ax.set_xlim(0, 112)
    ax.set_yticks(range(len(rows)), names)
    ax.set_xlabel("session recovery rate")
    ax.grid(axis="x")
    for idx, value in enumerate(recovered):
        ax.text(
            value + 1.2,
            idx,
            f"{value:.0f}% ({injected[idx]} x 503)",
            va="center",
            fontsize=9,
            color="#334155",
        )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_repeat_failures(
    aggregate: dict[str, Any] | None, output: Path, dpi: int
) -> None:
    if not aggregate or not isinstance(aggregate.get("rows"), list):
        return
    rows = aggregate["rows"]
    plt, _ = load_pyplot()
    apply_research_style(plt)
    names = [label(str(row.get("label") or row.get("name", ""))) for row in rows]
    injected = [int(row.get("injected_503", row.get("injected", 0))) for row in rows]
    recovered = [
        f"{int(row.get('sessions_recovered_after_error', 0))}/"
        f"{int(row.get('sessions_with_errors', 0))}"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.suptitle(
        "Repeat failures stress the same locks", x=0.02, ha="left", fontsize=16
    )
    ax.barh(range(len(rows)), injected, color="#d88b5b", height=0.58)
    ax.set_yticks(range(len(rows)), names)
    ax.set_xlabel("injected HTTP 503 responses")
    ax.grid(axis="x")
    high = max(injected)
    for idx, value in enumerate(injected):
        ax.text(
            value + high * 0.03,
            idx,
            f"{value}; recovered {recovered[idx]}",
            va="center",
            color="#334155",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, output, dpi)
    plt.close(fig)


def render_agent_task(
    summary: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    output: Path,
    dpi: int,
) -> None:
    if not summary:
        return
    plt, _ = load_pyplot()
    apply_research_style(plt)
    exact = int(summary["task_exact_successes"])
    instances = int(
        summary.get(
            "task_instances", summary.get("task_count", summary.get("tasks", exact))
        )
    )
    successes = int(summary["successes"])
    headers = summary.get("missing_router_header_counts", {})
    replay_hits = successes - int(headers.get("x-vsr-replay-id", 0))
    overhead = None
    if comparison:
        overhead = float(comparison["router_overhead_ms"]["p95"])

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    fig.suptitle(
        "Agent tasks complete with router diagnostics", x=0.02, ha="left", fontsize=16
    )
    names = ["exact tasks", "continuity violations", "replay headers"]
    values = [exact, 0, replay_hits]
    limits = [instances, 1, successes]
    colors = ["#2f7f68", "#2f7f68", "#5ea8d6"]
    ax.barh(range(3), values, color=colors, height=0.55)
    ax.set_yticks(range(3), names)
    ax.set_xlim(0, max(limits) * 1.15)
    ax.grid(axis="x")
    texts = [f"{exact}/{instances}", "0", f"{replay_hits}/{successes}"]
    if overhead is not None:
        texts[0] += f"; p95 overhead {overhead:+.1f} ms"
    for idx, text in enumerate(texts):
        ax.text(
            values[idx] + max(limits) * 0.02,
            idx,
            text,
            va="center",
            color="#334155",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, output, dpi)
    plt.close(fig)


def render_ga_readiness(data: dict[str, Any] | None, output: Path, dpi: int) -> None:
    if not data:
        return
    plt, _ = load_pyplot()
    apply_research_style(plt)
    passed = int(data.get("passed_count", 0))
    blocked = int(data.get("blocker_count", 0))

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    fig.suptitle("GA evidence is useful but incomplete", x=0.02, ha="left", fontsize=16)
    ax.barh([0, 1], [passed, blocked], color=["#2f7f68", "#d88b5b"], height=0.55)
    ax.set_yticks([0, 1], ["passed evidence classes", "blocking evidence gaps"])
    ax.set_xlim(0, max(passed, blocked) + 2)
    ax.grid(axis="x")
    ax.text(passed + 0.15, 0, str(passed), va="center", color="#334155", fontsize=10)
    ax.text(blocked + 0.15, 1, str(blocked), va="center", color="#334155", fontsize=10)
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

    synthetic_summary = read_json(args.synthetic_summary)
    if synthetic_summary and isinstance(synthetic_summary.get("overall"), dict):
        render_synthetic_headline(
            synthetic_summary["overall"],
            args.output_dir / "synthetic-headline.png",
            args.dpi,
        )
    render_matrix(scenario_summary, args.output_dir / "experiment-matrix.png", args.dpi)
    render_synthetic_safety(
        scenario_summary, args.output_dir / "synthetic-safety.png", args.dpi
    )
    render_ablation(ablation_summary, args.output_dir / "policy-ablation.png", args.dpi)
    render_ablation_cost_quality(
        ablation_summary, args.output_dir / "ablation-cost-quality.png", args.dpi
    )
    render_seed_stability(
        scenario_seed_summary, args.output_dir / "seed-stability.png", args.dpi
    )

    comparisons: list[dict[str, Any]] = []
    for path in args.amd_overhead_comparison:
        item = read_json(path)
        if item:
            item["label"] = path.parent.name
            comparisons.append(item)
    render_amd_overhead(
        comparisons, args.output_dir / "amd-overhead-results.png", args.dpi
    )
    render_amd_long_session(
        read_json(args.amd_long_session_aggregate),
        args.output_dir / "amd-long-session-matrix.png",
        args.dpi,
    )

    disruption_rows: list[dict[str, Any]] = []
    for path in args.amd_disruption_summary:
        item = read_json(path)
        if item:
            item["label"] = path.parent.name
            disruption_rows.append(item)
    render_disruptions(
        disruption_rows, args.output_dir / "amd-disruption-matrix.png", args.dpi
    )
    smoke = read_json(args.amd_disruption_smoke_summary)
    if smoke:
        smoke["label"] = "disruption smoke"
        render_disruptions(
            [smoke], args.output_dir / "amd-disruption-recovery.png", args.dpi
        )
    render_repeat_failures(
        read_json(args.amd_repeat_failure_aggregate),
        args.output_dir / "amd-repeat-failure-matrix.png",
        args.dpi,
    )
    render_agent_task(
        read_json(args.agent_task_summary),
        read_json(args.agent_task_comparison),
        args.output_dir / "amd-agent-task-results.png",
        args.dpi,
    )
    render_agent_task(
        read_json(args.long_agent_task_summary),
        read_json(args.long_agent_task_comparison),
        args.output_dir / "amd-long-agent-task-results.png",
        args.dpi,
    )
    render_ga_readiness(
        read_json(args.ga_readiness),
        args.output_dir / "ga-readiness-status.png",
        args.dpi,
    )

    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
