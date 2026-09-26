#!/usr/bin/env python3
"""Turn runs/ into results.md, and optionally into draft charts.

    python report.py                 # runs/results.md
    python report.py --figures       # also runs/figures/*.png (needs matplotlib)

Every value is read from summary.json, calls.jsonl and grades.json. Cost prefers
the router's /metrics delta and falls back to the x-vsr-cost headers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ARM_ORDER = ("all-frontier", "per-agent", "per-call", "all-local")
AGENTS = ("planner", "summarizer", "reviewer", "writer")


def load_runs(root: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*/*/summary.json")):
        summary = json.loads(path.read_text())
        grades_path = path.parent / "grades.json"
        summary["grades"] = (
            json.loads(grades_path.read_text()) if grades_path.exists() else None
        )
        summary["call_rows"] = [
            json.loads(line)
            for line in (path.parent / "calls.jsonl").read_text().splitlines()
            if line
        ]
        summary["dir"] = path.parent
        runs[summary["arm"]].append(summary)
    return runs


def run_cost(run: dict[str, Any]) -> float | None:
    # A $0 model never moves llm_model_cost_total, so an empty delta is a real $0.
    metrics = run.get("router_metrics") or {}
    if "cost_total" in metrics:
        return sum(metrics["cost_total"].values())
    headers = run["cost_from_headers"]
    if headers["total"] and not headers["unpriced_calls"]:
        return sum(headers["total"].values())
    return None


def cost_source(runs: list[dict[str, Any]]) -> str:
    sources = {
        "/metrics" if "cost_total" in (r.get("router_metrics") or {}) else "headers"
        for r in runs
        if run_cost(r) is not None
    }
    return ", ".join(sorted(sources)) or "none"


def spread(values: list[float | None], digits: int = 0) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "not measured"

    def fmt(v: float) -> str:
        return f"{v:.{digits}f}" if digits else f"{v:,.0f}"

    median = statistics.median(present)
    if len(present) == 1:
        return fmt(median)
    return f"{fmt(median)} ({fmt(min(present))} to {fmt(max(present))})"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def arm_table(runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    arms = [a for a in ARM_ORDER if runs.get(a)]
    rows = {
        "Runs": lambda rs: str(len(rs)),
        "Model calls per task": lambda rs: spread([r["calls"] for r in rs]),
        "Calls served by frontier": lambda rs: spread(
            [r["frontier_calls"] for r in rs]
        ),
        "Failed calls": lambda rs: spread([r["failed_calls"] for r in rs]),
        "Prompt tokens": lambda rs: spread([r["prompt_tokens"] for r in rs]),
        "Completion tokens": lambda rs: spread([r["completion_tokens"] for r in rs]),
        "Wall time (s)": lambda rs: spread([r["wall_time_s"] for r in rs], 1),
        "Cost per task (USD)": lambda rs: spread([run_cost(r) for r in rs], 6),
        "Cost source": cost_source,
        "Defects found (of 6)": lambda rs: spread(
            [r["grades"]["defects_found"] if r["grades"] else None for r in rs]
        ),
        "Truncated calls": lambda rs: spread([r["truncated_calls"] for r in rs]),
    }
    lines = ["| | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for label, fn in rows.items():
        lines.append(f"| {label} | " + " | ".join(fn(runs[a]) for a in arms) + " |")
    lines.append("")
    lines.append("Median, with the range across runs in brackets.")
    return lines


def agent_table(runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = ["| Arm | " + " | ".join(AGENTS) + " |", "|---|" + "---|" * len(AGENTS)]
    for arm in (a for a in ARM_ORDER if runs.get(a)):
        cells = []
        for agent in AGENTS:
            per_run = [
                Counter(run["by_agent"][agent]["selected_models"]) for run in runs[arm]
            ]
            models = sorted(set().union(*per_run))
            medians = {
                m: statistics.median(c.get(m, 0) for c in per_run) for m in models
            }
            cells.append(", ".join(f"{m} {n:g}" for m, n in medians.items()) or "-")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Calls per agent in one task, by the model that served them (median across runs)."
    )
    return lines


def defect_table(runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    defects = json.loads((HERE / "defects.json").read_text())
    arms = [a for a in ARM_ORDER if runs.get(a)]
    lines = ["| Defect | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for defect in defects:
        cells = []
        for arm in arms:
            graded = [
                r["grades"]["found"][defect["id"]] for r in runs[arm] if r["grades"]
            ]
            cells.append(f"{sum(graded)} of {len(graded)}" if graded else "-")
        lines.append(
            f"| {defect['id']} {defect['file']}: {defect['defect']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    lines.append(
        "How many graded runs of each arm found the defect in the final review."
    )
    return lines


def routing_table(runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        "| Arm | samples | p50 ms | p95 ms | max ms | /metrics mean ms | median model call s | routing share of wall time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in (a for a in ARM_ORDER if runs.get(a)):
        ok = [c for r in runs[arm] for c in r["call_rows"] if not c.get("error")]
        values = [
            c["routing_latency_ms"]
            for c in ok
            if c.get("routing_latency_ms") is not None
        ]
        means = [
            (r.get("router_metrics") or {}).get("routing_latency_ms_mean")
            for r in runs[arm]
        ]
        header_cells = (
            [
                str(len(values)),
                f"{percentile(values, 0.5):.3f}",
                f"{percentile(values, 0.95):.3f}",
                f"{max(values):.3f}",
            ]
            if values
            else ["0", "-", "-", "-"]
        )
        median_call = (
            f"{statistics.median(c['latency_s'] for c in ok):.2f}" if ok else "-"
        )
        wall = sum(r["wall_time_s"] for r in runs[arm])
        share = f"{sum(values) / 1000 / wall * 100:.4f}%" if values and wall else "-"
        lines.append(
            f"| {arm} | "
            + " | ".join(header_cells)
            + f" | {spread(means, 3)} | {median_call} | {share} |"
        )
    lines.append("")
    lines.append(
        "From x-vsr-routing-latency-ms on each response. /metrics mean is per run. "
        "Routing share is total routing time over total wall time for the arm."
    )
    return lines


def figures(runs: dict[str, list[dict[str, Any]]], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    colors = {"local": "#2e7d32", "frontier": "#6a1b9a"}
    written = []

    if runs.get("per-call"):
        # Draw the run whose call count is the median, so the tape matches the post's number.
        ok_rows = [
            [c for c in r["call_rows"] if not c.get("error")] for r in runs["per-call"]
        ]
        target = statistics.median(len(rows) for rows in ok_rows)
        pick = min(range(len(ok_rows)), key=lambda i: abs(len(ok_rows[i]) - target))
        run, rows = runs["per-call"][pick], ok_rows[pick]
        fig, ax = plt.subplots(figsize=(12, 2.6))
        for i, call in enumerate(rows):
            ax.bar(i, 1, width=0.8, color=colors.get(call["selected_model"], "#9e9e9e"))
        ax.set_yticks([])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([c["agent"][0].upper() for c in rows], fontsize=8)
        ax.set_title(
            f"One task, {len(rows)} model calls (per-call arm, run {run['run_id']})",
            loc="left",
        )
        written.append(out / "call-tape.png")
        fig.savefig(written[-1], dpi=200, bbox_inches="tight")
        plt.close(fig)

    arms = [a for a in ARM_ORDER if runs.get(a)]
    costs = [
        statistics.median([c for c in map(run_cost, runs[a]) if c is not None] or [0])
        for a in arms
    ]
    found = [
        statistics.median(
            [r["grades"]["defects_found"] for r in runs[a] if r["grades"]] or [0]
        )
        for a in arms
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(arms, costs, color="#5ea8d6")
    ax.set_ylabel("cost per task (USD, median)")
    twin = ax.twinx()
    twin.plot(arms, found, "o-", color="#e65100")
    twin.set_ylim(0, 6.5)
    twin.set_ylabel("defects found of 6 (median)")
    ax.set_title("Cost and defects found per arm", loc="left")
    written.append(out / "arms.png")
    fig.savefig(written[-1], dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = [0.0] * len(arms)
    shades = ("#c6dbef", "#6baed6", "#2171b5", "#08306b")
    for agent, shade in zip(AGENTS, shades, strict=True):
        widths = [
            statistics.median(
                [
                    r["by_agent"][agent]["prompt_tokens"]
                    + r["by_agent"][agent]["completion_tokens"]
                    for r in runs[a]
                ]
            )
            for a in arms
        ]
        ax.barh(arms, widths, left=left, color=shade, label=agent)
        left = [base + width for base, width in zip(left, widths, strict=True)]
    ax.set_xlabel("tokens per task (median)")
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    written.append(out / "tokens-by-agent.png")
    fig.savefig(written[-1], dpi=200, bbox_inches="tight")
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    runs = load_runs(args.runs)
    if not runs:
        parser.error(f"no runs under {args.runs}")
    sections = [
        "# Results",
        "",
        "## The four arms",
        "",
        *arm_table(runs),
        "",
        "## Which model served each agent",
        "",
        *agent_table(runs),
        "",
        "## Defects found",
        "",
        *defect_table(runs),
        "",
        "## Routing time",
        "",
        *routing_table(runs),
        "",
    ]
    target = args.runs / "results.md"
    target.write_text("\n".join(sections))
    print(f"wrote {target}")
    if args.figures:
        for path in figures(runs, args.runs / "figures"):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
