"""Agentic adjudication (§3.5.5, AG7).

An agentic workload is judged on task success rate through the M2 paired
machinery like every other workload; three lenses are added on top:

- Stability hard gates: unprotected tool-loop switches, rejected (503)
  selections, tool-loop interruptions and the per-task switch average are
  ceilings - an exceeded gate vetoes the verdict with FAIL, an unevaluable
  gate keeps the verdict INCONCLUSIVE (missing evidence is never counted
  as zero). Z5: upstream #5567 releases the tool-loop lock when
  classification drifts to a different decision, so a raw tool-loop
  switch count cannot be a zero ceiling - the drift transfer is in-design.
  The switch gate splits by cause: `max_unprotected_tool_loop_switches`
  caps the continuity tool-loop total minus the drift-released switches
  (the residual the identity-missing and bypassed switches live in, since
  only sr-bench's continuity sees those), and
  `max_decision_drift_switches_per_task` caps the drift transfers per
  replay session - exceeding it points at threshold calibration, not a
  routing regression. Continuity evidence comes from sr-bench's
  continuity section (`model_switches_by_phase`, `mean_switches_per_task`);
  the drift split and the rejected / interrupted evidence come from
  session attribution over the arm's own replay records
  (`session_attribution.analyze_sessions` for events, the protection
  traces' `decision_drift` for the cause).
- The R arm (random mixture) is computed per task (per session): the
  candidate arm's per-model task shares weight the reference arms'
  per-task results; the point delta is analytic and the interval is a
  joint bootstrap that redraws R's per-task assignment on every resample
  (the C8 closed form, generalized to fractional per-task scores).
- Cost is reported side by side: actual spend (cache discounts included)
  and the cache-neutral counterfactual, read verbatim from the compare
  payloads' `comparisons[]` cost fields, never recomputed.

Pure functions, no I/O, deterministic: same inputs -> same outputs.
"""

from __future__ import annotations

import bisect
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .session_attribution import (
    SESSION_CLASS_SELECTION_FAILED,
    SESSION_CLASS_TOOL_LOOP_INTERRUPTED,
)

# sr-bench transport.request_phase names (the Router's own tool-loop rule).
TOOL_LOOP_PHASE = "tool_loop"

_BOOTSTRAP_SEED = 20260918  # same fixed seed as sr-bench compare
_BOOTSTRAP_SAMPLES = 2000


@dataclass(frozen=True)
class AgenticGateResult:
    """One agentic stability ceiling (AG7).

    `value` is the measured evidence count on the candidate arm, `bound`
    the resolved ceiling (a `baseline` pin resolves against the baseline
    arm's measured value). None on either side means the gate did not
    complete: the verdict becomes INCONCLUSIVE, never a pass.
    """

    name: str
    value: float | None
    bound: float | None

    @property
    def evaluated(self) -> bool:
        return self.value is not None and self.bound is not None

    @property
    def passed(self) -> bool:
        return self.evaluated and self.value <= self.bound

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "bound": self.bound}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _phase_tool_loop_switches(continuity: Mapping[str, Any] | None) -> float | None:
    """Switches whose *arriving* request sat in the tool-loop phase.

    The continuity section counts observed model changes per arriving
    phase; a section without the `tool_loop` key observed no such change.
    A missing section is missing evidence (unevaluated), not zero.
    """
    if not isinstance(continuity, Mapping):
        return None
    by_phase = continuity.get("model_switches_by_phase")
    if not isinstance(by_phase, Mapping):
        return None
    return float(by_phase.get(TOOL_LOOP_PHASE, 0))


def _mean_switches(continuity: Mapping[str, Any] | None) -> float | None:
    """The continuity section's `mean_switches_per_task`, verbatim.

    sr-bench averages over the multi-request tasks only; with no
    multi-request task there is nothing observable and the field is null -
    that stays unevaluated (missing evidence is never guessed as zero).
    """
    if not isinstance(continuity, Mapping):
        return None
    return _number(continuity.get("mean_switches_per_task"))


def _event_count(events: Mapping[str, int] | None, event_class: str) -> float | None:
    if events is None:
        return None
    return float(events.get(event_class, 0))


@dataclass(frozen=True)
class SwitchEvidence:
    """Replay-side switch classification by cause (Z5).

    `tool_loop_switches` counts the session-ordered routed-model changes
    whose *arriving* request sat in the tool-loop phase (the same arriving
    rule the continuity section counts by); `decision_drift` counts those
    the protection trace released as decision drift (an in-design
    transfer, #5567); `sessions` is the distinct session count the rows
    cover (one sr-bench task = one session). The unprotected residual is
    the total minus the drift-released switches - the switches whose
    protection was missing entirely (no session identity, bypassed
    provider state, observe mode) are invisible to the replay rows and
    survive only in the continuity total, so the residual is what the
    zero ceiling judges.
    """

    tool_loop_switches: float
    decision_drift: float
    sessions: int

    @property
    def unprotected(self) -> float:
        return self.tool_loop_switches - self.decision_drift


def _switch_session_key(row: Mapping[str, Any]) -> str | None:
    """The conversation key one replay row belongs to (Z5).

    Same convention as session attribution: the row's session evidence
    names the conversation first, the session id second. One conversation
    is one sr-bench task.
    """
    session = row.get("session")
    if not isinstance(session, Mapping):
        return None
    for key in ("conversation_id", "session_id"):
        value = session.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _switch_order(row: Mapping[str, Any]) -> tuple[int, str, str]:
    """The request order inside one session (session attribution's order)."""
    session = row.get("session")
    turn_index = session.get("turn_index") if isinstance(session, Mapping) else 0
    return (
        turn_index,
        str(row.get("record_ts") or ""),
        str(row.get("request_id") or ""),
    )


def _arriving_policy(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The arriving request's protection trace, when the record carried one."""
    policy = row.get("session_policy")
    return policy if isinstance(policy, Mapping) else None


def replay_switch_evidence(
    rows: Sequence[Mapping[str, Any]] | None,
) -> SwitchEvidence | None:
    """Classify one arm's replay rows into switches by cause (Z5).

    Rows are the `session_rows_for_run` output of one arm run. Per session
    (conversation), requests sort by (turn_index, record_ts, request_id) -
    session attribution's order - and adjacent routed-model changes count
    as switches when the arriving request's protection trace names the
    tool-loop phase. A trace marked `decision_drift` classifies the switch
    as a drift transfer; without a trace the arriving phase is unknown and
    the change stays out (the continuity total still sees it, so the
    residual gate absorbs it on the unprotected side). Rows without
    session identity or a turn index cannot be ordered into a task and
    stay out of the session count. None rows mean no evidence was
    fetched; empty rows are evidence of zero recorded sessions, not
    missing evidence.
    """
    if rows is None:
        return None
    sessions: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = _switch_session_key(row)
        if key is None:
            continue
        session = row.get("session")
        turn_index = session.get("turn_index") if isinstance(session, Mapping) else None
        # not an orderable session request (session attribution's rule)
        if isinstance(turn_index, bool) or not isinstance(turn_index, int):
            continue
        sessions.setdefault(key, []).append(row)
    tool_loop_switches = 0.0
    decision_drift = 0.0
    for conversation in sorted(sessions):
        routed = [
            row
            for row in sorted(sessions[conversation], key=_switch_order)
            if str(row.get("routed") or "")
        ]
        for previous, arriving in pairwise(routed):
            if str(previous.get("routed") or "") == str(arriving.get("routed") or ""):
                continue
            policy = _arriving_policy(arriving)
            if policy is None:
                continue  # arriving phase unknown; continuity still counts it
            if policy.get("phase") != TOOL_LOOP_PHASE:
                continue
            tool_loop_switches += 1
            if policy.get("decision_drift") is True:
                decision_drift += 1
    return SwitchEvidence(
        tool_loop_switches=tool_loop_switches,
        decision_drift=decision_drift,
        sessions=len(sessions),
    )


def agentic_gate_results(
    gates: Mapping[str, float | str],
    *,
    candidate_continuity: Mapping[str, Any] | None,
    baseline_continuity: Mapping[str, Any] | None = None,
    candidate_events: Mapping[str, int] | None = None,
    baseline_events: Mapping[str, int] | None = None,
    candidate_switches: SwitchEvidence | None = None,
) -> list[AgenticGateResult]:
    """Resolve the agentic stability ceilings against the measured evidence.

    `candidate_events` / `baseline_events` are the session-attribution
    `event_counts` of the candidate and baseline arms (None = no replay
    records were provided, so the replay-sourced gates stay unevaluated).
    `candidate_switches` is the arm run's replay switch classification
    (Z5): the unprotected-switch gate subtracts its drift count from the
    continuity tool-loop total, and the drift gate divides it by the
    replay session count (one task = one session; zero sessions keep the
    per-task rate unevaluated). A `baseline` bound resolves against the
    baseline arm's measured value; a missing baseline value keeps the
    gate unevaluated, never a pass.
    """
    results: list[AgenticGateResult] = []
    for name in sorted(gates):
        bound_raw = gates[name]
        if name == "max_unprotected_tool_loop_switches":
            total = _phase_tool_loop_switches(candidate_continuity)
            if (
                candidate_switches is None
                or total is None
                or candidate_switches.decision_drift > total
            ):
                value = None
            else:
                value = total - candidate_switches.decision_drift
        elif name == "max_decision_drift_switches_per_task":
            if candidate_switches is None or candidate_switches.sessions <= 0:
                value = None
            else:
                value = candidate_switches.decision_drift / (
                    candidate_switches.sessions
                )
        elif name == "max_mean_switches_per_task":
            value = _mean_switches(candidate_continuity)
        elif name == "max_selection_rejected":
            value = _event_count(candidate_events, SESSION_CLASS_SELECTION_FAILED)
        elif name == "max_tool_loop_interrupted":
            value = _event_count(candidate_events, SESSION_CLASS_TOOL_LOOP_INTERRUPTED)
        else:  # contract validation keeps unknown names out; stay unevaluated
            value = None
        if bound_raw == "baseline":
            baseline = _event_count(
                baseline_events, SESSION_CLASS_TOOL_LOOP_INTERRUPTED
            )
            bound = baseline
        else:
            bound = _number(bound_raw)
        results.append(AgenticGateResult(name=name, value=value, bound=bound))
    return results


_COMPARISON_COST_KEYS = (
    "baseline_total_cost_usd",
    "candidate_total_cost_usd",
    "cache_neutral_baseline_cost_usd",
    "cache_neutral_candidate_cost_usd",
)


def agentic_cost_summary(pair_comparisons: Sequence[Any]) -> dict[str, Any]:
    """Side-by-side actual vs cache-neutral cost across the AB/BA pairs.

    Each pair's compare payload carries one `comparisons[]` entry with the
    two arms' total spend and its cache-neutral counterfactual; the pairs
    replay the same case set, so the sides sum across pairs. An unknown
    value on any pair (sr-bench serializes unknown usage as null) keeps
    that side unknown with its reason - unknown usage is never zero, and
    a saving percentage is only computed over complete sides.
    """
    summary: dict[str, Any] = {
        "actual_cost_usd": {"baseline": None, "candidate": None},
        "cache_neutral_cost_usd": {"baseline": None, "candidate": None},
        "saving_percent": {"total": None, "cache_neutral": None},
        "complete": True,
        "reasons": [],
        "notes": {},
    }
    sums = dict.fromkeys(_COMPARISON_COST_KEYS, 0.0)
    for index, comparison in enumerate(pair_comparisons, 1):
        entries = (
            comparison.get("comparisons") if isinstance(comparison, dict) else None
        )
        if (
            not isinstance(entries, list)
            or len(entries) != 1
            or not isinstance(entries[0], dict)
        ):
            summary["complete"] = False
            summary["reasons"].append(f"pair {index}: no single comparisons[] entry")
            continue
        entry = entries[0]
        for key, field in (
            ("total_cost_comparison_reason", "total_cost_reason"),
            ("cache_neutral_cost_basis", "cache_neutral_basis"),
        ):
            if isinstance(entry.get(key), str) and field not in summary["notes"]:
                summary["notes"][field] = entry[key]
        for key in _COMPARISON_COST_KEYS:
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                summary["complete"] = False
                summary["reasons"].append(f"pair {index}: {key} is unknown")
                continue
            sums[key] += float(value)
    if summary["complete"]:
        summary["actual_cost_usd"] = {
            "baseline": sums["baseline_total_cost_usd"],
            "candidate": sums["candidate_total_cost_usd"],
        }
        summary["cache_neutral_cost_usd"] = {
            "baseline": sums["cache_neutral_baseline_cost_usd"],
            "candidate": sums["cache_neutral_candidate_cost_usd"],
        }
        for side, keys in (
            ("total", ("baseline_total_cost_usd", "candidate_total_cost_usd")),
            (
                "cache_neutral",
                (
                    "cache_neutral_baseline_cost_usd",
                    "cache_neutral_candidate_cost_usd",
                ),
            ),
        ):
            base = sums[keys[0]]
            if base > 0:
                summary["saving_percent"][side] = (base - sums[keys[1]]) / base * 100.0
    return summary


def session_random_mix_stats(
    candidate_task_scores: Mapping[str, float],
    reference_task_scores: Mapping[str, Mapping[str, float]],
    shares: Mapping[str, float],
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = _BOOTSTRAP_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """The session-level R arm: random mixture expectation against the
    reference arms' per-task results (§3.5.5, AG7).

    R answers task i with model m drawn with probability p_m (the
    candidate arm's per-model task shares), so the point delta is analytic:
    the candidate's mean task score minus the share-weighted reference
    means over the tasks every weighted reference judged. The interval is
    a joint bootstrap - every resample redraws the task indices with
    replacement *and* R's per-task model assignment by the shares - with
    the same fixed seed and quantile indexing as `compute_paired_stats`,
    so an agentic R row is directly comparable to a C8 random-mix row.
    Tasks may carry fractional scores (an agentic task is 0/1, a session
    expectation need not be). No pairing yields None - missing evidence is
    never counted as zero.
    """
    aliases = sorted(
        a for a, p in shares.items() if isinstance(p, (int, float)) and p > 0
    )
    if not aliases or not candidate_task_scores:
        return None
    needed_refs: dict[str, dict[str, float]] = {}
    for alias in aliases:
        rows = reference_task_scores.get(alias)
        if not isinstance(rows, Mapping):
            return None
        needed_refs[alias] = {
            task: float(score)
            for task, score in rows.items()
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        }
    paired = [
        task
        for task, score in candidate_task_scores.items()
        if isinstance(score, (int, float))
        and not isinstance(score, bool)
        and all(task in needed_refs[alias] for alias in aliases)
    ]
    if not paired:
        return None
    weight_sum = sum(float(shares[a]) for a in aliases)
    probs = {a: float(shares[a]) / weight_sum for a in aliases}
    cand_scores = [float(candidate_task_scores[task]) for task in paired]
    ref_scores = {
        alias: [needed_refs[alias][task] for task in paired] for alias in aliases
    }
    candidate_mean = statistics.mean(cand_scores)
    expected_r = sum(probs[a] * statistics.mean(ref_scores[a]) for a in aliases)
    cumulative: list[float] = []
    total = 0.0
    for alias in aliases:
        total += probs[alias]
        cumulative.append(total)
    rng = random.Random(seed)
    n = len(paired)
    samples: list[float] = []
    for _ in range(bootstrap_samples):
        indices = rng.choices(range(n), k=n)
        cand_sum = 0.0
        ref_sum = 0.0
        for i in indices:
            cand_sum += cand_scores[i]
            draw = rng.random() * total
            alias = aliases[bisect.bisect_left(cumulative, draw)]
            ref_sum += ref_scores[alias][i]
        samples.append(cand_sum / n - ref_sum / n)
    samples.sort()
    low_i = max(0, int(0.025 * bootstrap_samples) - 1)
    high_i = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples) - 1)
    return {
        "paired_tasks": n,
        "shares": probs,
        "candidate_accuracy": candidate_mean,
        "expected_r_accuracy": expected_r,
        "delta": candidate_mean - expected_r,
        "bootstrap_mean": statistics.mean(samples),
        "bootstrap_ci95": (samples[low_i], samples[high_i]),
    }
