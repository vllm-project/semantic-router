#!/usr/bin/env python3
"""Generate agentic-routing experiment data from replay or a deterministic workload.

The script intentionally uses the Python standard library so maintainers can
run it on dev and AMD validation hosts without installing plotting stacks.
It writes CSV/JSON plus a compact SVG figure under .agent-harness by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    name: str
    quality: float
    prompt_per_1m: float
    completion_per_1m: float
    cached_per_1m: float


MODELS = (
    ModelProfile("qwen3-8b", 0.64, 0.05, 0.10, 0.01),
    ModelProfile("qwen3-32b", 0.78, 0.30, 0.60, 0.06),
    ModelProfile("frontier-reasoner", 0.94, 8.00, 24.00, 1.20),
)
FRONTIER_DEMAND_THRESHOLD = 0.78
MID_DEMAND_THRESHOLD = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-replay", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--turns", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(".agent-harness/experiments/agentic-routing") / stamp


def simulate_workload(
    session_count: int, turn_count: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for sidx in range(session_count):
        session_id = f"sess-{sidx:03d}"
        baseline_model = MODELS[0]
        agentic_model = MODELS[0]
        switch_count = 0
        history_tokens = rng.randint(512, 2048)
        idle_break_turn = rng.choice((0, turn_count // 2, turn_count + 1))

        for turn in range(turn_count):
            phase = phase_for_turn(turn)
            idle_expired = turn == idle_break_turn
            demand = demand_score(turn, turn_count, phase, rng)
            base_choice = choose_base_model(demand)

            baseline_next = base_choice
            agentic_next, reason = choose_agentic_model(
                base_choice=base_choice,
                current=agentic_model,
                phase=phase,
                turn=turn,
                switch_count=switch_count,
                history_tokens=history_tokens,
                idle_expired=idle_expired,
            )

            baseline_switch = baseline_next.name != baseline_model.name
            agentic_switch = agentic_next.name != agentic_model.name
            if agentic_switch:
                switch_count += 1

            prompt_tokens = 800 + history_tokens
            completion_tokens = 220 + int(180 * demand)
            cached_ratio = cached_ratio_for(
                agentic_next, phase, history_tokens, agentic_switch
            )
            baseline_cost = turn_cost(
                baseline_next, prompt_tokens, 0.0, completion_tokens
            )
            agentic_cost = turn_cost(
                agentic_next, prompt_tokens, cached_ratio, completion_tokens
            )

            rows.append(
                {
                    "session_id": session_id,
                    "turn": turn,
                    "phase": phase,
                    "idle_expired": idle_expired,
                    "demand": round(demand, 4),
                    "history_tokens": history_tokens,
                    "baseline_model": baseline_next.name,
                    "agentic_model": agentic_next.name,
                    "baseline_switch": baseline_switch,
                    "agentic_switch": agentic_switch,
                    "tool_loop_switch_violation_baseline": phase == "tool_loop"
                    and baseline_switch,
                    "tool_loop_switch_violation_agentic": phase == "tool_loop"
                    and agentic_switch,
                    "cached_prompt_ratio": round(cached_ratio, 4),
                    "baseline_quality": baseline_next.quality,
                    "agentic_quality": agentic_next.quality,
                    "baseline_cost": round(baseline_cost, 8),
                    "agentic_cost": round(agentic_cost, 8),
                    "policy_reason": reason,
                }
            )

            baseline_model = baseline_next
            agentic_model = agentic_next
            history_tokens = min(
                32000, history_tokens + prompt_tokens // 8 + completion_tokens
            )
    return rows


def phase_for_turn(turn: int) -> str:
    cycle = turn % 5
    if cycle in (2, 3):
        return "tool_loop"
    return "user_turn"


def demand_score(turn: int, turn_count: int, phase: str, rng: random.Random) -> float:
    progress = turn / max(turn_count - 1, 1)
    phase_bonus = 0.16 if phase == "tool_loop" else 0.0
    wave = 0.18 * math.sin(turn * math.pi / 4)
    return max(
        0.0,
        min(
            1.0, 0.42 + 0.28 * progress + phase_bonus + wave + rng.uniform(-0.09, 0.09)
        ),
    )


def choose_base_model(demand: float) -> ModelProfile:
    if demand >= FRONTIER_DEMAND_THRESHOLD:
        return MODELS[2]
    if demand >= MID_DEMAND_THRESHOLD:
        return MODELS[1]
    return MODELS[0]


def choose_agentic_model(
    base_choice: ModelProfile,
    current: ModelProfile,
    phase: str,
    turn: int,
    switch_count: int,
    history_tokens: int,
    idle_expired: bool,
) -> tuple[ModelProfile, str]:
    if turn == 0 or idle_expired:
        return base_choice, "cold_or_idle_select"
    if phase == "tool_loop":
        return current, "tool_loop_hard_lock"
    if base_choice.name == current.name:
        return current, "base_selects_current"

    quality_gain = base_choice.quality - current.quality
    continuation_mass = min(1.0, history_tokens / 16000)
    frontier_multiplier = 1.0 + 1.6 * max(
        cost_pressure(base_choice), cost_pressure(current)
    )
    switch_cost = (
        0.05
        + 0.20 * continuation_mass * frontier_multiplier
        + 0.04 * min(switch_count / 8, 1)
    )
    if quality_gain > switch_cost:
        return base_choice, "quality_gain_clears_switch_cost"
    return current, "continuity_cost_blocks_switch"


def cost_pressure(model: ModelProfile) -> float:
    max_cost = max(m.prompt_per_1m + m.completion_per_1m for m in MODELS)
    return (model.prompt_per_1m + model.completion_per_1m) / max_cost


def cached_ratio_for(
    model: ModelProfile, phase: str, history_tokens: int, switched: bool
) -> float:
    base = min(0.72, history_tokens / 48000)
    if phase == "tool_loop":
        base += 0.12
    if model.name == "frontier-reasoner":
        base += 0.06
    if switched:
        base *= 0.20
    return max(0.0, min(0.85, base))


def turn_cost(
    model: ModelProfile, prompt_tokens: int, cached_ratio: float, completion_tokens: int
) -> float:
    cached = int(prompt_tokens * cached_ratio)
    uncached = prompt_tokens - cached
    return (
        uncached * model.prompt_per_1m
        + cached * model.cached_per_1m
        + completion_tokens * model.completion_per_1m
    ) / 1_000_000


def rows_from_replay(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    records = data.get("data", data) if isinstance(data, dict) else data
    rows = []
    for rec in records:
        policy = rec.get("session_policy") or {}
        rows.append(
            {
                "session_id": rec.get("session_id", ""),
                "turn": rec.get("turn_index", 0),
                "phase": policy.get("phase", ""),
                "idle_expired": policy.get("idle_expired", False),
                "demand": "",
                "history_tokens": "",
                "baseline_model": policy.get("fallback_selected_model", ""),
                "agentic_model": rec.get("selected_model", ""),
                "baseline_switch": "",
                "agentic_switch": policy.get("current_model")
                != rec.get("selected_model"),
                "tool_loop_switch_violation_baseline": "",
                "tool_loop_switch_violation_agentic": policy.get("phase") == "tool_loop"
                and policy.get("current_model") != rec.get("selected_model"),
                "cached_prompt_ratio": "",
                "baseline_quality": "",
                "agentic_quality": "",
                "baseline_cost": rec.get("baseline_cost", ""),
                "agentic_cost": rec.get("actual_cost", ""),
                "policy_reason": policy.get("decision_reason", ""),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    sessions = {row["session_id"] for row in rows}
    baseline_switches = sum(
        bool(row["baseline_switch"]) for row in rows if row["baseline_switch"] != ""
    )
    agentic_switches = sum(
        bool(row["agentic_switch"]) for row in rows if row["agentic_switch"] != ""
    )
    baseline_tool_violations = sum(
        bool(row["tool_loop_switch_violation_baseline"])
        for row in rows
        if row["tool_loop_switch_violation_baseline"] != ""
    )
    agentic_tool_violations = sum(
        bool(row["tool_loop_switch_violation_agentic"])
        for row in rows
        if row["tool_loop_switch_violation_agentic"] != ""
    )
    baseline_cost = sum(float(row["baseline_cost"] or 0) for row in rows)
    agentic_cost = sum(float(row["agentic_cost"] or 0) for row in rows)
    cached_values = [
        float(row["cached_prompt_ratio"])
        for row in rows
        if row["cached_prompt_ratio"] != ""
    ]
    agentic_quality = [
        float(row["agentic_quality"]) for row in rows if row["agentic_quality"] != ""
    ]
    baseline_quality = [
        float(row["baseline_quality"]) for row in rows if row["baseline_quality"] != ""
    ]
    return {
        "sessions": len(sessions),
        "turns": len(rows),
        "baseline_switches": baseline_switches,
        "agentic_switches": agentic_switches,
        "switch_reduction": baseline_switches - agentic_switches,
        "baseline_tool_loop_switch_violations": baseline_tool_violations,
        "agentic_tool_loop_switch_violations": agentic_tool_violations,
        "baseline_cost": round(baseline_cost, 6),
        "agentic_cost": round(agentic_cost, 6),
        "cost_delta": round(agentic_cost - baseline_cost, 6),
        "mean_cached_prompt_ratio": (
            round(mean(cached_values), 4) if cached_values else None
        ),
        "baseline_mean_quality": (
            round(mean(baseline_quality), 4) if baseline_quality else None
        ),
        "agentic_mean_quality": (
            round(mean(agentic_quality), 4) if agentic_quality else None
        ),
    }


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    fieldnames = list(rows[0].keys()) if rows else []
    with (out_dir / "turns.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "summary.svg").write_text(render_svg(summary))
    return summary


def render_svg(summary: dict[str, Any]) -> str:
    metrics = [
        ("Baseline switches", float(summary.get("baseline_switches") or 0)),
        ("Agentic switches", float(summary.get("agentic_switches") or 0)),
        (
            "Baseline tool violations",
            float(summary.get("baseline_tool_loop_switch_violations") or 0),
        ),
        (
            "Agentic tool violations",
            float(summary.get("agentic_tool_loop_switch_violations") or 0),
        ),
    ]
    max_value = max([value for _, value in metrics] + [1])
    bars = []
    for idx, (label, value) in enumerate(metrics):
        y = 48 + idx * 42
        width = int(420 * value / max_value)
        color = "#2f6fed" if "Agentic" in label else "#8892a0"
        bars.append(
            f'<text x="24" y="{y + 15}" font-size="13" fill="#1f2937">{label}</text>'
        )
        bars.append(
            f'<rect x="190" y="{y}" width="{width}" height="22" rx="4" fill="{color}"/>'
        )
        bars.append(
            f'<text x="{198 + width}" y="{y + 15}" font-size="12" fill="#111827">{int(value)}</text>'
        )
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="680" height="240" viewBox="0 0 680 240">',
            '<rect width="680" height="240" fill="#f8fafc"/>',
            '<text x="24" y="28" font-size="18" font-weight="700" fill="#111827">Agentic Continuity Routing</text>',
            *bars,
            f'<text x="24" y="224" font-size="12" fill="#4b5563">Cost delta: {summary.get("cost_delta")} | Mean cached ratio: {summary.get("mean_cached_prompt_ratio")}</text>',
            "</svg>",
        ]
    )


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or default_output_dir()
    rows = (
        rows_from_replay(args.from_replay)
        if args.from_replay
        else simulate_workload(args.sessions, args.turns, args.seed)
    )
    summary = write_outputs(rows, out_dir)
    print(
        json.dumps(
            {"output_dir": str(out_dir), "summary": summary}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
