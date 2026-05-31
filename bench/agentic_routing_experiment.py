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


@dataclass(frozen=True)
class WorkloadScenario:
    name: str
    base_demand: float
    progress_weight: float
    phase_bonus: float
    wave_amplitude: float
    noise: float
    tool_cycle: int
    tool_slots: tuple[int, ...]
    history_min: int
    history_max: int
    idle_mode: str
    remaining_turn_prior: float
    token_growth: float = 1.0
    state_cycle: int = 0
    state_slots: tuple[int, ...] = ()
    drift_cycle: int = 0
    drift_slots: tuple[int, ...] = ()


@dataclass(frozen=True)
class AgenticDecision:
    model: ModelProfile
    reason: str
    continuation_mass: float
    prior_mass: float
    remaining_turns_estimate: float


MODELS = (
    ModelProfile("qwen3-8b", 0.64, 0.05, 0.10, 0.01),
    ModelProfile("qwen3-32b", 0.78, 0.30, 0.60, 0.06),
    ModelProfile("frontier-reasoner", 0.94, 8.00, 24.00, 1.20),
)
FRONTIER_DEMAND_THRESHOLD = 0.78
MID_DEMAND_THRESHOLD = 0.55
DEFAULT_SCENARIOS = (
    "balanced",
    "tool-heavy",
    "frontier-heavy",
    "idle-heavy",
    "stateful-heavy",
    "drift-heavy",
)
DEFAULT_ABLATION_POLICIES = (
    "sticky-session",
    "acr-no-tool-lock",
    "acr-no-context-portability",
    "acr-no-decision-drift-reset",
    "acr-no-idle-boundary",
    "acr-no-remaining-prior",
    "acr-no-frontier-cost",
    "acr-full",
)
SCENARIOS = {
    "balanced": WorkloadScenario(
        name="balanced",
        base_demand=0.42,
        progress_weight=0.28,
        phase_bonus=0.16,
        wave_amplitude=0.18,
        noise=0.09,
        tool_cycle=5,
        tool_slots=(2, 3),
        history_min=512,
        history_max=2048,
        idle_mode="single",
        remaining_turn_prior=6.0,
    ),
    "tool-heavy": WorkloadScenario(
        name="tool-heavy",
        base_demand=0.55,
        progress_weight=0.22,
        phase_bonus=0.18,
        wave_amplitude=0.14,
        noise=0.08,
        tool_cycle=6,
        tool_slots=(2, 3, 4),
        history_min=1024,
        history_max=3072,
        idle_mode="single",
        remaining_turn_prior=9.0,
        token_growth=1.12,
    ),
    "frontier-heavy": WorkloadScenario(
        name="frontier-heavy",
        base_demand=0.55,
        progress_weight=0.34,
        phase_bonus=0.20,
        wave_amplitude=0.17,
        noise=0.07,
        tool_cycle=5,
        tool_slots=(2, 3),
        history_min=1536,
        history_max=4096,
        idle_mode="rare",
        remaining_turn_prior=10.0,
        token_growth=1.20,
    ),
    "idle-heavy": WorkloadScenario(
        name="idle-heavy",
        base_demand=0.42,
        progress_weight=0.30,
        phase_bonus=0.14,
        wave_amplitude=0.16,
        noise=0.10,
        tool_cycle=5,
        tool_slots=(2, 3),
        history_min=512,
        history_max=2048,
        idle_mode="frequent",
        remaining_turn_prior=4.0,
    ),
    "stateful-heavy": WorkloadScenario(
        name="stateful-heavy",
        base_demand=0.58,
        progress_weight=0.30,
        phase_bonus=0.24,
        wave_amplitude=0.15,
        noise=0.08,
        tool_cycle=6,
        tool_slots=(3, 4),
        history_min=128,
        history_max=1024,
        idle_mode="rare",
        remaining_turn_prior=3.0,
        token_growth=0.95,
        state_cycle=5,
        state_slots=(1, 2),
    ),
    "drift-heavy": WorkloadScenario(
        name="drift-heavy",
        base_demand=0.46,
        progress_weight=0.28,
        phase_bonus=0.26,
        wave_amplitude=0.12,
        noise=0.08,
        tool_cycle=6,
        tool_slots=(3,),
        history_min=2048,
        history_max=6144,
        idle_mode="rare",
        remaining_turn_prior=7.0,
        token_growth=1.08,
        drift_cycle=6,
        drift_slots=(1, 2),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-replay", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--turns", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="balanced",
        help="Synthetic workload shape to simulate.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the maintained scenario matrix instead of one workload.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run policy ablations across the maintained scenario matrix.",
    )
    parser.add_argument(
        "--agentic-policy",
        choices=DEFAULT_ABLATION_POLICIES,
        default="acr-full",
        help="Session policy variant used for a single synthetic workload.",
    )
    parser.add_argument(
        "--seeds",
        default="20260530,20260531,20260532,20260533,20260534",
        help="Comma-separated seeds for --matrix.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(".agent-harness/experiments/agentic-routing") / stamp


def simulate_workload(
    session_count: int,
    turn_count: int,
    seed: int,
    scenario: WorkloadScenario = SCENARIOS["balanced"],
    agentic_policy: str = "acr-full",
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for sidx in range(session_count):
        session_id = f"sess-{sidx:03d}"
        baseline_model = MODELS[0]
        agentic_model = MODELS[0]
        switch_count = 0
        history_tokens = rng.randint(scenario.history_min, scenario.history_max)
        idle_turns = idle_turns_for(turn_count, scenario, rng)

        for turn in range(turn_count):
            phase = phase_for_turn(turn, scenario)
            idle_expired = turn in idle_turns
            demand = demand_score(turn, turn_count, phase, rng, scenario)
            base_choice = choose_base_model(demand)

            baseline_next = base_choice
            agentic_decision = choose_agentic_model(
                base_choice=base_choice,
                current=agentic_model,
                phase=phase,
                turn=turn,
                switch_count=switch_count,
                history_tokens=history_tokens,
                idle_expired=idle_expired,
                remaining_turn_prior=scenario.remaining_turn_prior,
                policy=agentic_policy,
            )
            agentic_next = agentic_decision.model

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
                    "scenario": scenario.name,
                    "seed": seed,
                    "session_id": session_id,
                    "turn": turn,
                    "phase": phase,
                    "idle_expired": idle_expired,
                    "demand": round(demand, 4),
                    "history_tokens": history_tokens,
                    "remaining_turn_prior": round(scenario.remaining_turn_prior, 4),
                    "remaining_turns_estimate": round(
                        agentic_decision.remaining_turns_estimate, 4
                    ),
                    "continuation_mass": round(agentic_decision.continuation_mass, 4),
                    "prior_mass": round(agentic_decision.prior_mass, 4),
                    "baseline_model": baseline_next.name,
                    "agentic_model": agentic_next.name,
                    "baseline_switch": baseline_switch,
                    "agentic_switch": agentic_switch,
                    "tool_loop_switch_violation_baseline": phase == "tool_loop"
                    and baseline_switch,
                    "tool_loop_switch_violation_agentic": phase == "tool_loop"
                    and agentic_switch,
                    "context_portability_violation_baseline": phase == "provider_state"
                    and baseline_switch,
                    "context_portability_violation_agentic": phase == "provider_state"
                    and agentic_switch,
                    "decision_drift": phase == "topic_drift",
                    "agentic_drift_switch": phase == "topic_drift" and agentic_switch,
                    "cached_prompt_ratio": round(cached_ratio, 4),
                    "baseline_quality": baseline_next.quality,
                    "agentic_quality": agentic_next.quality,
                    "baseline_cost": round(baseline_cost, 8),
                    "agentic_cost": round(agentic_cost, 8),
                    "policy_reason": agentic_decision.reason,
                }
            )

            baseline_model = baseline_next
            agentic_model = agentic_next
            history_tokens = min(
                64000,
                history_tokens
                + int((prompt_tokens // 8 + completion_tokens) * scenario.token_growth),
            )
    return rows


def idle_turns_for(
    turn_count: int, scenario: WorkloadScenario, rng: random.Random
) -> set[int]:
    if scenario.idle_mode == "frequent":
        interval = max(5, turn_count // 4)
        return set(range(interval, turn_count, interval))
    if scenario.idle_mode == "rare":
        return {turn_count + 1}
    return {rng.choice((0, turn_count // 2, turn_count + 1))}


def phase_for_turn(turn: int, scenario: WorkloadScenario) -> str:
    if scenario.drift_cycle > 0 and turn % scenario.drift_cycle in scenario.drift_slots:
        return "topic_drift"
    if scenario.state_cycle > 0 and turn % scenario.state_cycle in scenario.state_slots:
        return "provider_state"
    cycle = turn % scenario.tool_cycle
    if cycle in scenario.tool_slots:
        return "tool_loop"
    return "user_turn"


def demand_score(
    turn: int,
    turn_count: int,
    phase: str,
    rng: random.Random,
    scenario: WorkloadScenario,
) -> float:
    progress = turn / max(turn_count - 1, 1)
    phase_bonus = scenario.phase_bonus if phase == "tool_loop" else 0.0
    if phase == "provider_state":
        phase_bonus = scenario.phase_bonus
    if phase == "topic_drift":
        phase_bonus = scenario.phase_bonus
    wave = scenario.wave_amplitude * math.sin(turn * math.pi / 4)
    return max(
        0.0,
        min(
            1.0,
            scenario.base_demand
            + scenario.progress_weight * progress
            + phase_bonus
            + wave
            + rng.uniform(-scenario.noise, scenario.noise),
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
    remaining_turn_prior: float,
    policy: str = "acr-full",
) -> AgenticDecision:
    continuation_mass, prior_mass, remaining_turns_estimate = continuation_evidence(
        history_tokens=history_tokens,
        turn=turn,
        remaining_turn_prior=remaining_turn_prior,
        use_remaining_prior=policy != "acr-no-remaining-prior",
    )
    boundary = agentic_boundary_decision(
        base_choice=base_choice,
        current=current,
        phase=phase,
        turn=turn,
        idle_expired=idle_expired,
        policy=policy,
    )
    if boundary is not None:
        model, reason = boundary
        return AgenticDecision(
            model,
            reason,
            continuation_mass,
            prior_mass,
            remaining_turns_estimate,
        )
    if base_choice.name == current.name:
        return AgenticDecision(
            current,
            "base_selects_current",
            continuation_mass,
            prior_mass,
            remaining_turns_estimate,
        )

    quality_gain = base_choice.quality - current.quality
    frontier_multiplier = (
        1.0
        if policy == "acr-no-frontier-cost"
        else 1.0 + 1.6 * max(cost_pressure(base_choice), cost_pressure(current))
    )
    cache_warmth = cache_warmth_estimate(current, phase, history_tokens)
    switch_cost = (
        0.05
        + 0.20 * continuation_mass * cache_warmth * frontier_multiplier
        + 0.04 * min(switch_count / 8, 1)
    )
    if quality_gain > switch_cost:
        return AgenticDecision(
            base_choice,
            "quality_gain_clears_switch_cost",
            continuation_mass,
            prior_mass,
            remaining_turns_estimate,
        )
    return AgenticDecision(
        current,
        "continuity_cost_blocks_switch",
        continuation_mass,
        prior_mass,
        remaining_turns_estimate,
    )


def agentic_boundary_decision(
    *,
    base_choice: ModelProfile,
    current: ModelProfile,
    phase: str,
    turn: int,
    idle_expired: bool,
    policy: str,
) -> tuple[ModelProfile, str] | None:
    if policy == "sticky-session" and turn > 0:
        return current, "sticky_session"
    if turn == 0:
        return base_choice, "cold_select"
    if phase == "tool_loop" and policy != "acr-no-tool-lock":
        return current, "tool_loop_hard_lock"
    if phase == "provider_state" and policy != "acr-no-context-portability":
        return current, "context_portability_hard_lock"
    if phase == "topic_drift" and policy != "acr-no-decision-drift-reset":
        return base_choice, "decision_drift_select"
    if idle_expired and policy != "acr-no-idle-boundary":
        return base_choice, "idle_select"
    return None


def continuation_evidence(
    *,
    history_tokens: int,
    turn: int,
    remaining_turn_prior: float,
    use_remaining_prior: bool,
) -> tuple[float, float, float]:
    history_mass = min(1.0, max(0.0, history_tokens / 16000))
    if not use_remaining_prior:
        return history_mass, 0.0, 0.0
    remaining_turns_estimate = max(0.0, remaining_turn_prior - turn)
    prior_mass = min(1.0, remaining_turns_estimate / 8.0)
    return max(history_mass, prior_mass), prior_mass, remaining_turns_estimate


def cost_pressure(model: ModelProfile) -> float:
    max_cost = max(cache_checkout_cost(m) for m in MODELS)
    if max_cost <= 0:
        return 0.5
    return cache_checkout_cost(model) / max_cost


def cache_warmth_estimate(
    model: ModelProfile, phase: str, history_tokens: int, switched: bool = False
) -> float:
    # Mirror the live selector's conservative default when exact cache warmth is
    # unknown: early sessions still have future-continuation risk even before a
    # long prefix has accumulated.
    base = max(0.5, min(0.72, history_tokens / 48000))
    if phase == "tool_loop":
        base += 0.12
    if phase == "provider_state":
        base += 0.08
    if model.name == "frontier-reasoner":
        base += 0.06
    if switched:
        base *= 0.20
    return max(0.0, min(0.85, base))


def cache_checkout_cost(model: ModelProfile) -> float:
    delta = model.prompt_per_1m - model.cached_per_1m
    if delta > 0:
        return delta
    return max(model.prompt_per_1m, 0.0)


def cached_ratio_for(
    model: ModelProfile, phase: str, history_tokens: int, switched: bool
) -> float:
    return cache_warmth_estimate(model, phase, history_tokens, switched)


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
                "scenario": rec.get("scenario", "replay"),
                "seed": rec.get("seed", ""),
                "session_id": rec.get("session_id", ""),
                "turn": rec.get("turn_index", 0),
                "phase": policy.get("phase", ""),
                "idle_expired": policy.get("idle_expired", False),
                "demand": "",
                "history_tokens": "",
                "remaining_turn_prior": policy.get("remaining_turn_prior", ""),
                "remaining_turns_estimate": policy.get("remaining_turns_estimate", ""),
                "continuation_mass": policy.get("continuation_mass", ""),
                "prior_mass": "",
                "baseline_model": policy.get("fallback_selected_model", ""),
                "agentic_model": rec.get("selected_model", ""),
                "baseline_switch": "",
                "agentic_switch": policy.get("current_model")
                != rec.get("selected_model"),
                "tool_loop_switch_violation_baseline": "",
                "tool_loop_switch_violation_agentic": policy.get("phase") == "tool_loop"
                and policy.get("current_model") != rec.get("selected_model"),
                "context_portability_violation_baseline": "",
                "context_portability_violation_agentic": bool(
                    policy.get("has_non_portable_context")
                )
                and policy.get("current_model") != rec.get("selected_model"),
                "decision_drift": bool(policy.get("decision_drift")),
                "agentic_drift_switch": bool(policy.get("decision_drift"))
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
    baseline_context_violations = sum(
        bool(row["context_portability_violation_baseline"])
        for row in rows
        if row.get("context_portability_violation_baseline", "") != ""
    )
    agentic_context_violations = sum(
        bool(row["context_portability_violation_agentic"])
        for row in rows
        if row.get("context_portability_violation_agentic", "") != ""
    )
    drift_turns = [row for row in rows if bool(row.get("decision_drift"))]
    agentic_drift_switches = sum(
        bool(row.get("agentic_drift_switch")) for row in drift_turns
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
    baseline_frontier_turns = sum(
        row.get("baseline_model") == "frontier-reasoner" for row in rows
    )
    agentic_frontier_turns = sum(
        row.get("agentic_model") == "frontier-reasoner" for row in rows
    )
    idle_turns = [row for row in rows if bool(row.get("idle_expired"))]
    agentic_idle_switches = sum(bool(row.get("agentic_switch")) for row in idle_turns)
    continuation_values = [
        float(row["continuation_mass"])
        for row in rows
        if row.get("continuation_mass", "") != ""
    ]
    remaining_estimates = [
        float(row["remaining_turns_estimate"])
        for row in rows
        if row.get("remaining_turns_estimate", "") != ""
    ]
    return {
        "sessions": len(sessions),
        "turns": len(rows),
        "baseline_switches": baseline_switches,
        "agentic_switches": agentic_switches,
        "switch_reduction": baseline_switches - agentic_switches,
        "baseline_tool_loop_switch_violations": baseline_tool_violations,
        "agentic_tool_loop_switch_violations": agentic_tool_violations,
        "baseline_context_portability_violations": baseline_context_violations,
        "agentic_context_portability_violations": agentic_context_violations,
        "decision_drift_turns": len(drift_turns),
        "agentic_drift_switches": agentic_drift_switches,
        "baseline_cost": round(baseline_cost, 6),
        "agentic_cost": round(agentic_cost, 6),
        "cost_delta": round(agentic_cost - baseline_cost, 6),
        "cost_reduction_pct": (
            round((baseline_cost - agentic_cost) / baseline_cost * 100, 2)
            if baseline_cost
            else None
        ),
        "switch_reduction_pct": (
            round((baseline_switches - agentic_switches) / baseline_switches * 100, 2)
            if baseline_switches
            else None
        ),
        "baseline_frontier_turns": baseline_frontier_turns,
        "agentic_frontier_turns": agentic_frontier_turns,
        "frontier_turn_delta": agentic_frontier_turns - baseline_frontier_turns,
        "idle_turns": len(idle_turns),
        "agentic_idle_switches": agentic_idle_switches,
        "mean_cached_prompt_ratio": (
            round(mean(cached_values), 4) if cached_values else None
        ),
        "mean_continuation_mass": (
            round(mean(continuation_values), 4) if continuation_values else None
        ),
        "mean_remaining_turns_estimate": (
            round(mean(remaining_estimates), 4) if remaining_estimates else None
        ),
        "baseline_mean_quality": (
            round(mean(baseline_quality), 4) if baseline_quality else None
        ),
        "agentic_mean_quality": (
            round(mean(agentic_quality), 4) if agentic_quality else None
        ),
        "quality_delta": (
            round(mean(agentic_quality) - mean(baseline_quality), 4)
            if agentic_quality and baseline_quality
            else None
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


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def write_matrix_outputs(
    scenarios: tuple[str, ...],
    seeds: list[int],
    sessions: int,
    turns: int,
    out_dir: Path,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    for scenario_name in scenarios:
        scenario = SCENARIOS[scenario_name]
        scenario_rows: list[dict[str, Any]] = []
        for seed in seeds:
            rows = simulate_workload(sessions, turns, seed, scenario)
            all_rows.extend(rows)
            scenario_rows.extend(rows)
            seed_summaries.append(
                {
                    "scenario": scenario_name,
                    "seed": seed,
                    **summarize(rows),
                }
            )
        scenario_summaries.append(
            {"scenario": scenario_name, **summarize(scenario_rows)}
        )

    summary = {
        "matrix": {
            "scenarios": list(scenarios),
            "seeds": seeds,
            "sessions_per_seed": sessions,
            "turns_per_session": turns,
        },
        "overall": summarize(all_rows),
        "by_scenario": scenario_summaries,
        "by_scenario_seed": seed_summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, out_dir / "turns.csv")
    write_csv(scenario_summaries, out_dir / "scenario_summary.csv")
    write_csv(seed_summaries, out_dir / "scenario_seed_summary.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "summary.svg").write_text(render_matrix_svg(summary))
    return summary


def summarize_baseline_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    return {
        "policy": "single-turn",
        "turns": summary["turns"],
        "switches": summary["baseline_switches"],
        "switch_reduction_pct": 0.0,
        "tool_loop_switch_violations": summary["baseline_tool_loop_switch_violations"],
        "context_portability_violations": summary[
            "baseline_context_portability_violations"
        ],
        "decision_drift_turns": summary["decision_drift_turns"],
        "drift_switches": "",
        "estimated_cost": summary["baseline_cost"],
        "cost_reduction_pct": 0.0,
        "frontier_turns": summary["baseline_frontier_turns"],
        "idle_switches": "",
        "mean_cached_prompt_ratio": "",
        "mean_continuation_mass": "",
        "mean_remaining_turns_estimate": "",
        "mean_quality": summary["baseline_mean_quality"],
        "quality_delta": 0.0,
    }


def summarize_agentic_policy(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    return {
        "policy": policy,
        "turns": summary["turns"],
        "switches": summary["agentic_switches"],
        "switch_reduction_pct": summary["switch_reduction_pct"],
        "tool_loop_switch_violations": summary["agentic_tool_loop_switch_violations"],
        "context_portability_violations": summary[
            "agentic_context_portability_violations"
        ],
        "decision_drift_turns": summary["decision_drift_turns"],
        "drift_switches": summary["agentic_drift_switches"],
        "estimated_cost": summary["agentic_cost"],
        "cost_reduction_pct": summary["cost_reduction_pct"],
        "frontier_turns": summary["agentic_frontier_turns"],
        "idle_switches": summary["agentic_idle_switches"],
        "mean_cached_prompt_ratio": summary["mean_cached_prompt_ratio"],
        "mean_continuation_mass": summary["mean_continuation_mass"],
        "mean_remaining_turns_estimate": summary["mean_remaining_turns_estimate"],
        "mean_quality": summary["agentic_mean_quality"],
        "quality_delta": summary["quality_delta"],
    }


def write_ablation_outputs(
    scenarios: tuple[str, ...],
    seeds: list[int],
    sessions: int,
    turns: int,
    out_dir: Path,
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    policy_rows: dict[str, list[dict[str, Any]]] = {}
    baseline_rows: list[dict[str, Any]] = []

    for policy in DEFAULT_ABLATION_POLICIES:
        rows_for_policy: list[dict[str, Any]] = []
        for scenario_name in scenarios:
            scenario = SCENARIOS[scenario_name]
            for seed in seeds:
                rows = simulate_workload(
                    sessions,
                    turns,
                    seed,
                    scenario,
                    agentic_policy=policy,
                )
                rows_for_policy.extend(rows)
                if policy == "acr-full":
                    baseline_rows.extend(rows)
        policy_rows[policy] = rows_for_policy

    summary_rows.append(summarize_baseline_policy(baseline_rows))
    for policy in DEFAULT_ABLATION_POLICIES:
        summary_rows.append(summarize_agentic_policy(policy, policy_rows[policy]))

    summary = {
        "matrix": {
            "scenarios": list(scenarios),
            "seeds": seeds,
            "sessions_per_seed": sessions,
            "turns_per_session": turns,
        },
        "by_policy": summary_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, out_dir / "ablation_summary.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "summary.svg").write_text(render_ablation_svg(summary))
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        (
            "Agentic context violations",
            float(summary.get("agentic_context_portability_violations") or 0),
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
    height = 270
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{height}" viewBox="0 0 680 {height}">',
            f'<rect width="680" height="{height}" fill="#ffffff"/>',
            '<text x="24" y="28" font-size="18" font-weight="700" fill="#111827">Agentic Continuity Routing</text>',
            *bars,
            f'<text x="24" y="{height - 16}" font-size="12" fill="#4b5563">Cost delta: {summary.get("cost_delta")} | Mean cached ratio: {summary.get("mean_cached_prompt_ratio")}</text>',
            "</svg>",
        ]
    )


def render_matrix_svg(summary: dict[str, Any]) -> str:
    scenarios = summary.get("by_scenario", [])
    max_switch = max([row["baseline_switches"] for row in scenarios] + [1])
    max_cost = max([row["baseline_cost"] for row in scenarios] + [1])
    rows = []
    for idx, row in enumerate(scenarios):
        y = 68 + idx * 74
        rows.append(
            f'<text x="28" y="{y + 15}" font-size="14" font-weight="700" fill="#111827">{row["scenario"]}</text>'
        )
        baseline_switch_w = int(260 * row["baseline_switches"] / max_switch)
        agentic_switch_w = int(260 * row["agentic_switches"] / max_switch)
        baseline_cost_w = int(260 * row["baseline_cost"] / max_cost)
        agentic_cost_w = int(260 * row["agentic_cost"] / max_cost)
        rows.extend(
            [
                f'<rect x="160" y="{y}" width="{baseline_switch_w}" height="14" rx="3" fill="#94a3b8"/>',
                f'<rect x="160" y="{y + 18}" width="{agentic_switch_w}" height="14" rx="3" fill="#16a34a"/>',
                f'<text x="430" y="{y + 12}" font-size="11" fill="#475569">switches {row["baseline_switches"]} -> {row["agentic_switches"]}</text>',
                f'<rect x="560" y="{y}" width="{baseline_cost_w}" height="14" rx="3" fill="#cbd5e1"/>',
                f'<rect x="560" y="{y + 18}" width="{agentic_cost_w}" height="14" rx="3" fill="#22c55e"/>',
                f'<text x="830" y="{y + 12}" font-size="11" fill="#475569">cost -{row["cost_reduction_pct"]}%</text>',
                f'<text x="160" y="{y + 52}" font-size="11" fill="#64748b">tool-loop violations {row["baseline_tool_loop_switch_violations"]} -> {row["agentic_tool_loop_switch_violations"]}; context violations {row["baseline_context_portability_violations"]} -> {row["agentic_context_portability_violations"]}; drift switches {row["agentic_drift_switches"]}/{row["decision_drift_turns"]}; quality delta {row["quality_delta"]}</text>',
            ]
        )
    height = 110 + len(scenarios) * 74
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}" viewBox="0 0 980 {height}">',
            '<rect width="980" height="100%" fill="#ffffff"/>',
            '<text x="28" y="32" font-size="20" font-weight="700" fill="#0f172a">Session-aware Agentic Routing Matrix</text>',
            '<text x="160" y="54" font-size="12" fill="#64748b">model switches</text>',
            '<text x="560" y="54" font-size="12" fill="#64748b">estimated cost</text>',
            *rows,
            f'<rect x="28" y="{height - 28}" width="12" height="12" fill="#94a3b8"/>',
            f'<text x="46" y="{height - 18}" font-size="11" fill="#64748b">baseline selector</text>',
            f'<rect x="160" y="{height - 28}" width="12" height="12" fill="#16a34a"/>',
            f'<text x="178" y="{height - 18}" font-size="11" fill="#64748b">session-aware policy</text>',
            "</svg>",
        ]
    )


def render_ablation_svg(summary: dict[str, Any]) -> str:
    policies = summary.get("by_policy", [])
    max_switch = max([row["switches"] for row in policies] + [1])
    costs = [float(row["estimated_cost"] or 0) for row in policies]
    max_cost = max([*costs, 1.0])
    rows = []
    for idx, row in enumerate(policies):
        y = 72 + idx * 52
        switch_w = int(250 * row["switches"] / max_switch)
        cost_w = int(250 * float(row["estimated_cost"] or 0) / max_cost)
        color = "#2563eb" if row["policy"] == "acr-full" else "#94a3b8"
        rows.extend(
            [
                f'<text x="28" y="{y + 14}" font-size="13" font-weight="700" fill="#0f172a">{row["policy"]}</text>',
                f'<rect x="230" y="{y}" width="{switch_w}" height="14" fill="{color}"/>',
                f'<text x="490" y="{y + 12}" font-size="11" fill="#475569">{row["switches"]} switches</text>',
                f'<rect x="610" y="{y}" width="{cost_w}" height="14" fill="{color}"/>',
                f'<text x="870" y="{y + 12}" font-size="11" fill="#475569">${float(row["estimated_cost"] or 0):.4f}</text>',
                f'<text x="230" y="{y + 34}" font-size="11" fill="#64748b">tool-loop violations {row["tool_loop_switch_violations"]}; context violations {row["context_portability_violations"]}; drift switches {row["drift_switches"]}/{row["decision_drift_turns"]}; cached {row["mean_cached_prompt_ratio"]}; continuation {row["mean_continuation_mass"]}</text>',
            ]
        )
    height = 112 + len(policies) * 52
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}" viewBox="0 0 980 {height}">',
            '<rect width="980" height="100%" fill="#ffffff"/>',
            '<text x="28" y="32" font-size="20" font-weight="700" fill="#0f172a">Agentic Routing Ablation</text>',
            '<text x="230" y="56" font-size="12" fill="#64748b">model switches</text>',
            '<text x="610" y="56" font-size="12" fill="#64748b">estimated cost</text>',
            *rows,
            "</svg>",
        ]
    )


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or default_output_dir()
    if args.ablation:
        summary = write_ablation_outputs(
            DEFAULT_SCENARIOS,
            parse_seeds(args.seeds),
            args.sessions,
            args.turns,
            out_dir,
        )
    elif args.matrix:
        summary = write_matrix_outputs(
            DEFAULT_SCENARIOS,
            parse_seeds(args.seeds),
            args.sessions,
            args.turns,
            out_dir,
        )
    else:
        rows = (
            rows_from_replay(args.from_replay)
            if args.from_replay
            else simulate_workload(
                args.sessions,
                args.turns,
                args.seed,
                SCENARIOS[args.scenario],
                agentic_policy=args.agentic_policy,
            )
        )
        summary = write_outputs(rows, out_dir)
    print(
        json.dumps(
            {"output_dir": str(out_dir), "summary": summary}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
