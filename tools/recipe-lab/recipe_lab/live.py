"""Live-tier orchestration (Sub-issue 3).

The live tier's evaluation engine is sr-bench; this module only wires the
evidence together:

1. the caller (or an upstream script) runs sr-bench for baseline and
   candidate and holds two run IDs;
2. `live_verdict_from_runs` compares them, reads the objective metric,
   applies the three-state verdict with the guard floors and the objective
   constraints, and returns the live-tier evidence block (bench_refs
   included - the ledger record is self-sufficient for Sub-issue 4's
   report replay).

Guards (A3): the contract is the only source of floors. `spec.gates`
declares `{name, metric, floor}`; the `--guard-measurements` file carries
measurements only (`name` / `candidate` / `baseline` values produced
offline by model_eval guard tooling). A `floor` inside the measurement
file is a contract bypass and is rejected. `floor: "baseline"` in the
contract resolves against the measured baseline value. A declared gate
without a usable measurement is not a failure: the gate is unevaluated
and the verdict becomes INCONCLUSIVE.

Verified cost fields (E): sr-bench compare exposes per-arm spend on the
`baseline_targets` / `candidate_targets` entries as `total_spend_usd`,
`cost_usd`, `cache_neutral_cost_usd` and `evaluation_cost_usd`; unknown
usage is serialized as `null` (with a `total_cost_comparison_reason`),
so a constraint on an unknown cost must be unevaluated (INCONCLUSIVE),
never zero.

Paired statistics (M2): when a compare payload carries exactly one
`comparisons[]` entry, its `paired_cases` / `quality_delta` /
`quality_delta_ci95` / `quality_delta_bootstrap_ci95` / `wins` / `losses` /
`ties` are read verbatim (never recomputed) and the verdict decides from
the confidence interval against `objective.min_improvement_pp`, with
`objective.min_cases` as the minimum paired sample. The stats are recorded
in the ledger record's `paired` field.

Evidence quality (M3): each compare payload's `baseline_targets` /
`candidate_targets` entries carry `output_diagnostics` (planned/result
cases, the per-case output-limit count, subject-call finish reasons).
Truncation is the per-case count (`output_limit_cases / planned_cases`);
the call-level finish-reason length share is reference only. Truncation,
missing-result, and no-answer rates beyond the spec's `evidence_quality`
ceilings mark the arm's score unreliable evidence and turn the verdict
INCONCLUSIVE; unknown rates (no diagnostics, or no per-case no-answer
marker) are reported as null, never guessed as zero. Across repeated
pairs each rate merges conservatively (worst pair).

Reference arms and cost basis (M4): a spec may pin `references.strong`
/ `references.weak` to single-model sr-bench target ids; constraints
bound with `ref:strong` / `ref:weak` resolve against those arms'
measured metric values (an unknown reference value is not evaluable,
never zero). `factor` scales a reference bound multiplicatively;
`tolerance_pp` relaxes any bound by absolute percentage points. The
verify driver runs one single-model live run per declared alias on the
frozen protocol, cached by (dataset hash, manifest hash, alias) in a
JSON run-id cache, and reports each arm's cost: measured spend when
sr-bench exposes it, otherwise an estimate flagged "assumed" from
`cost_basis.assumed_pricing` per-million token rates. Pareto rows
(accuracy, cost, latency per arm) are recorded for the promotion
review.

Live `verify` orchestration (I): `verify_live` drives sr-bench end to end
for one candidate - a preview run (routing distribution into the ledger
notes) plus `repetitions` AB/BA pairs, where each pair re-runs the
baseline's frozen manifest AND submits a fresh candidate-plan, so drift
spreads across both arms. Per-pair compare feeds descriptive medians and
ranges (abba.paired_summary); the verdict reuses `evaluate_verdict`, and
any incomplete pair makes the whole verdict INCONCLUSIVE.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import abba
from .contract import (
    Constraint,
    ContractError,
    CostBasis,
    EvidenceQuality,
    ExperimentSpec,
    Gate,
    References,
)
from .srbench import SrbenchCli, extract_metric_values
from .verdict import (
    GuardResult,
    PairedStats,
    evaluate_verdict,
    evidence_quality_reasons,
)

# An AB/BA pair always carries exactly two arms' metrics.
PAIRED_ARM_COUNT = 2

_MEASUREMENT_KEYS = ("name", "candidate", "baseline")


def measurements_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a guard-measurements JSONL (measurements only, no floors).

    Each row is `{name, candidate, baseline}`; `candidate` and `baseline`
    are optional numbers (missing = not measured). A `floor` key means the
    caller is trying to bypass the contract's gates - ContractError. Keys
    outside the measurement schema are rejected the same way.
    """
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ContractError(f"guard measurement row {index} must be a JSON object")
        unknown = sorted(set(row) - set(_MEASUREMENT_KEYS))
        if unknown:
            raise ContractError(
                f"guard measurement row {index} has non-measurement keys {unknown}; "
                "floors come from the spec gates, the file carries measurements only"
            )
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ContractError(
                f"guard measurement row {index}: name must be a non-empty string"
            )
        for key in ("candidate", "baseline"):
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(
                    f"guard measurement row {index}: {key} must be a number or null"
                )
        validated.append({key: row.get(key) for key in _MEASUREMENT_KEYS})
    return validated


def resolve_gates(
    gates: Sequence[Gate],
    measurements: list[dict[str, Any]],
) -> tuple[list[GuardResult], list[str]]:
    """Resolve the contract's gates against measured values (A3).

    Returns the GuardResult list (one per declared gate, in contract
    order) and the sorted names of measurements the contract never
    declared. `floor: "baseline"` resolves against the measured baseline
    value; a gate with no measurement, no baseline value for a baseline
    floor, or no candidate value stays unevaluated (value=None, floor=None)
    so the verdict becomes INCONCLUSIVE instead of guessing.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for row in measurements:
        name = row["name"]
        if name in by_name:
            raise ContractError(f"duplicate guard measurement for {name!r}")
        by_name[name] = row
    undeclared = sorted(set(by_name) - {gate.name for gate in gates})

    guards: list[GuardResult] = []
    for gate in gates:
        row = by_name.get(gate.name)
        if row is None:
            guards.append(
                GuardResult(name=gate.name, metric=gate.metric, value=None, floor=None)
            )
            continue
        floor = gate.floor
        if floor == "baseline":
            base = row.get("baseline")
            if base is None:
                guards.append(
                    GuardResult(
                        name=gate.name, metric=gate.metric, value=None, floor=None
                    )
                )
                continue
            floor = float(base)
        candidate = row.get("candidate")
        guards.append(
            GuardResult(
                name=gate.name,
                metric=gate.metric,
                value=None if candidate is None else float(candidate),
                floor=floor,
            )
        )
    return guards, undeclared


def _arm_accuracy(targets: Any, metric_key: str, *, routing_arm: bool) -> float | None:
    """Accuracy of one run's comparison arm from a target list.

    A candidate run typically carries a reference single-target lane (the
    "no routing" arm) before the routing arm, so traversal order alone would
    read the reference instead of the router. The routing arm is the `mom`
    target when present, else the last target with the metric.
    """
    if not isinstance(targets, list):
        return None
    entries = [t for t in targets if isinstance(t, dict)]
    if routing_arm:
        for entry in entries:
            if entry.get("kind") == "mom":
                values = extract_metric_values(entry, metric_key)
                if values:
                    return values[0]
    for entry in reversed(entries):
        values = extract_metric_values(entry, metric_key)
        if values:
            return values[0]
    return None


def _pair_metrics(
    comparison: Any,
    metric_key: str,
    incomplete: list[str],
) -> tuple[float | None, float | None]:
    """Pair baseline/candidate metric values from a compare payload.

    Preferred shape: {"baseline_targets": [...], "candidate_targets": [...]}
    sections (sr-bench compare), where the candidate arm is the routing
    target, not its reference single lane. Flat payloads with exactly two
    numeric occurrences are read in traversal order; anything else is
    ambiguous evidence and becomes INCONCLUSIVE rather than a guess.
    """
    if isinstance(comparison, dict):
        base_section = comparison.get("baseline", comparison.get("baseline_targets"))
        cand_section = comparison.get("candidate", comparison.get("candidate_targets"))
        if isinstance(base_section, (dict, list)) and isinstance(
            cand_section, (dict, list)
        ):
            if isinstance(base_section, list):
                baseline_metric = _arm_accuracy(
                    base_section, metric_key, routing_arm=False
                )
            else:
                base_values = extract_metric_values(base_section, metric_key)
                baseline_metric = base_values[0] if base_values else None
            if isinstance(cand_section, list):
                candidate_metric = _arm_accuracy(
                    cand_section, metric_key, routing_arm=True
                )
            else:
                cand_values = extract_metric_values(cand_section, metric_key)
                candidate_metric = cand_values[0] if cand_values else None
            return baseline_metric, candidate_metric
    values = extract_metric_values(comparison, metric_key)
    if len(values) == PAIRED_ARM_COUNT:
        return values[0], values[1]
    incomplete.append(
        f"ambiguous compare payload for {metric_key!r}: {len(values)} numeric values found"
    )
    return None, None


def _constraint_evidence(
    comparison: Any,
    constraints: list[Constraint],
    incomplete: list[str],
) -> list[dict[str, Any]]:
    """Measure every declared constraint metric on both arms (E).

    The same pairing rules as the objective metric apply (see
    `_pair_metrics`). Unknown values stay None: sr-bench serializes
    unknown spend as null and an unevaluable constraint is INCONCLUSIVE,
    never zero.
    """
    evidence: list[dict[str, Any]] = []
    for constraint in constraints:
        baseline_value, candidate_value = _pair_metrics(
            comparison, constraint.metric, incomplete
        )
        evidence.append(
            {
                "name": constraint.name,
                "metric": constraint.metric,
                "op": constraint.op,
                "bound": constraint.bound,
                "tolerance_pct": constraint.tolerance_pct,
                "tolerance_pp": constraint.tolerance_pp,
                "factor": constraint.factor,
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
        )
    return evidence


def paired_stats_from_comparison(comparison: Any) -> PairedStats | None:
    """M2: paired statistics from a compare payload's `comparisons[]`.

    sr-bench compare emits one `comparisons[]` entry per candidate target.
    A payload with exactly one entry is unambiguous; zero entries, several,
    or a malformed entry (missing or non-numeric fields) yield None - the
    verdict then explains why instead of guessing which comparison applies.
    """
    if not isinstance(comparison, dict):
        return None
    entries = comparison.get("comparisons")
    if not isinstance(entries, list) or len(entries) != 1:
        return None
    entry = entries[0]
    if not isinstance(entry, dict):
        return None
    paired_cases = entry.get("paired_cases")
    delta = entry.get("quality_delta")
    wins = entry.get("wins")
    losses = entry.get("losses")
    ties = entry.get("ties")
    for value in (paired_cases, delta, wins, losses, ties):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    if paired_cases < 0:
        return None

    def _interval(raw: Any) -> tuple[float, float] | None:
        if (
            isinstance(raw, (list, tuple))
            and len(raw) == PAIRED_ARM_COUNT
            and all(
                not isinstance(v, bool) and isinstance(v, (int, float)) for v in raw
            )
        ):
            return (float(raw[0]), float(raw[1]))
        return None

    return PairedStats(
        paired_cases=int(paired_cases),
        quality_delta=float(delta),
        quality_delta_ci95=_interval(entry.get("quality_delta_ci95")),
        quality_delta_bootstrap_ci95=_interval(
            entry.get("quality_delta_bootstrap_ci95")
        ),
        wins=int(wins),
        losses=int(losses),
        ties=int(ties),
    )


def _merge_paired_stats(stats: list[PairedStats]) -> PairedStats | None:
    """M2: conservative merge of the per-pair statistics across AB/BA pairs
    and repeated runs.

    `paired_cases` takes the minimum, not the sum: every pair replays the
    same case set, so summing would inflate the distinct-case count and let
    repetitions dilute the min_cases bar (70 cases x 3 runs would read 210).
    The merge width and the total observation count are kept in
    `repetitions` and `paired_observations`. wins/losses/ties still sum
    (each observation counts) and the delta averages; each interval takes
    the widest span (min low, max high). A recomputed merged interval would
    need the per-case diffs, which a compare payload does not expose, so
    the merged bounds are conservative, never optimistic. Any pair without
    an interval drops the merged interval to None (the verdict then says
    why).
    """
    if not stats:
        return None
    intervals = [s.quality_delta_ci95 for s in stats]
    boots = [s.quality_delta_bootstrap_ci95 for s in stats]

    def _span(values: list[tuple[float, float] | None]) -> tuple[float, float] | None:
        if any(v is None for v in values):
            return None
        return (min(lo for lo, _ in values), max(hi for _, hi in values))

    return PairedStats(
        paired_cases=min(s.paired_cases for s in stats),
        quality_delta=statistics.mean([s.quality_delta for s in stats]),
        quality_delta_ci95=_span(intervals),
        quality_delta_bootstrap_ci95=_span(boots),
        wins=sum(s.wins for s in stats),
        losses=sum(s.losses for s in stats),
        ties=sum(s.ties for s in stats),
        repetitions=len(stats),
        paired_observations=sum(s.paired_cases for s in stats),
    )


_EVIDENCE_RATE_KEYS = (
    "truncation_rate",
    "missing_result_rate",
    "no_answer_rate",
    "finish_reason_length_rate",
)


def arm_evidence_quality(
    targets: Any,
    *,
    routing_arm: bool,
    selected_id: str | None = None,
) -> dict[str, float | None] | None:
    """M3: one arm's evidence-quality rates from its target summaries.

    sr-bench compares whole targets and each target summary carries
    `output_diagnostics`. Truncation is the per-case count:
    `output_limit_cases / planned_cases` - sr-bench flags a case
    `quality_failure: "output_limit"` when the subject output hits the
    token limit and the engine stops the task early (W3), and counts
    those cases per target. The call-level share
    (`subject_calls.finish_reasons.length / subject_calls.total`) rides
    along as reference only: calls can outnumber cases under multi-call
    tasks, so it is not the truncation rate. The missing-result rate is
    `(planned_cases - result_cases) / planned_cases`. The no-answer rate
    needs per-case judging details, which a compare payload does not
    carry: until sr-bench publishes a per-case no-answer count
    (`no_answer_cases`), it stays null - missing evidence is reported,
    never guessed as zero (unanswered cases already count as incorrect
    in sr-bench's planned-denominator accuracy).

    The baseline arm pins the target compare named
    (`baseline_selected_target_id`); the candidate arm is the routing
    target (`mom` kind, else the last target), matching `_arm_accuracy`.
    No diagnostics or a non-positive planned count yield None.
    """
    if not isinstance(targets, list):
        return None
    entries = [t for t in targets if isinstance(t, dict)]

    def _rates(entry: dict) -> dict[str, float | None] | None:
        diagnostics = entry.get("output_diagnostics")
        if not isinstance(diagnostics, dict):
            return None
        planned = diagnostics.get("planned_cases")
        if isinstance(planned, bool) or not isinstance(planned, int) or planned <= 0:
            return None
        result_cases = diagnostics.get("result_cases")
        if isinstance(result_cases, bool) or not isinstance(result_cases, int):
            result_cases = 0
        output_limit = diagnostics.get("output_limit_cases")
        if isinstance(output_limit, bool) or not isinstance(output_limit, int):
            output_limit = 0
        length_rate: float | None = None
        calls = diagnostics.get("subject_calls")
        if isinstance(calls, dict):
            total = calls.get("total")
            finish = calls.get("finish_reasons")
            if (
                isinstance(total, int)
                and not isinstance(total, bool)
                and total > 0
                and isinstance(finish, dict)
            ):
                length = finish.get("length", 0)
                if isinstance(length, int) and not isinstance(length, bool):
                    length_rate = length / total
        no_answer = diagnostics.get("no_answer_cases")
        if isinstance(no_answer, bool) or not isinstance(no_answer, int):
            no_answer = None
        return {
            "truncation_rate": output_limit / planned,
            "missing_result_rate": (planned - result_cases) / planned,
            "no_answer_rate": (no_answer / planned if no_answer is not None else None),
            "finish_reason_length_rate": length_rate,
        }

    if isinstance(selected_id, str):
        for entry in entries:
            if entry.get("id") == selected_id:
                rates = _rates(entry)
                if rates is not None:
                    return rates
    if routing_arm:
        for entry in entries:
            if entry.get("kind") == "mom":
                rates = _rates(entry)
                if rates is not None:
                    return rates
    for entry in reversed(entries):
        rates = _rates(entry)
        if rates is not None:
            return rates
    return None


def _evidence_entry(
    rates: dict[str, float | None] | None,
    arm: str,
    ceilings: EvidenceQuality,
    incomplete: list[str],
) -> dict[str, float | None]:
    """M3: one arm's evidence-quality result block plus its reasons.

    Rates are fractions (0-1); reasons compare them against the spec's
    percentage ceilings and land in `incomplete_reasons` (INCONCLUSIVE).
    `finish_reason_length_rate` is reference only and never gated.
    """
    entry: dict[str, float | None] = dict.fromkeys(_EVIDENCE_RATE_KEYS)
    if rates is not None:
        entry.update({key: rates.get(key) for key in _EVIDENCE_RATE_KEYS})
    incomplete.extend(
        evidence_quality_reasons(
            arm,
            truncation_rate=entry["truncation_rate"],
            missing_result_rate=entry["missing_result_rate"],
            no_answer_rate=entry["no_answer_rate"],
            max_truncation_pct=ceilings.max_truncation_pct,
            max_no_answer_pct=ceilings.max_no_answer_pct,
        )
    )
    return entry


def _worst_rates(
    rates: list[dict[str, float | None] | None],
) -> dict[str, float | None] | None:
    """M3: worst-case per-arm rates across repeated pairs (max each).

    A pair without diagnostics contributes nothing and never masks a
    breach seen in another pair; if no pair has diagnostics the rate
    stays unknown (None).
    """
    if not rates:
        return None
    merged: dict[str, float | None] = {}
    for key in _EVIDENCE_RATE_KEYS:
        values = [
            entry[key]
            for entry in rates
            if isinstance(entry, dict) and entry.get(key) is not None
        ]
        merged[key] = max(values) if values else None
    return merged


def reference_metric_values(
    targets: Any,
    references: References | None,
    metric_key: str,
) -> dict[str, float | None]:
    """M4: the declared reference arms' metric values from a target list.

    `references.strong` / `references.weak` name single-model target ids;
    the first matching target in the baseline section carries the value.
    Unknown arms stay None - a ref-bound constraint is then unevaluated
    (INCONCLUSIVE), never counted as zero.
    """
    values: dict[str, float | None] = {"strong": None, "weak": None}
    if references is None or not isinstance(targets, list):
        return values
    for role in ("strong", "weak"):
        alias = getattr(references, role)
        for entry in targets:
            if isinstance(entry, dict) and entry.get("id") == alias:
                found = extract_metric_values(entry, metric_key)
                if found:
                    values[role] = found[0]
                break
    return values


def arm_cost(
    target: Any, cost_basis: CostBasis | None
) -> tuple[float | None, str | None]:
    """M4: one target's (cost_usd, basis_note) for the Pareto table.

    Measured spend (`total_spend_usd`, then `cost_usd`) wins and is noted
    "measured". Only when the target carries exactly one model's token
    usage and the spec declares assumed pricing for that model is the
    spend estimated from rates - noted "assumed" so no estimate can pass
    as billing. Anything else stays unknown (None), never zero.
    """
    if not isinstance(target, dict):
        return None, None
    for key in ("total_spend_usd", "cost_usd"):
        value = target.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return float(value), "measured"
    tokens = target.get("tokens")
    if not isinstance(tokens, dict) or cost_basis is None:
        return None, None
    assumed = cost_basis.assumed_pricing
    if not assumed:
        return None, None
    models = target.get("selected_models")
    if not isinstance(models, dict) or len(models) != 1:
        return None, None
    rates = assumed.get(str(next(iter(models))))
    if not rates:
        return None, None

    def _tokens(key: str) -> float:
        value = tokens.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return 0.0
        return float(value)

    prompt = (
        _tokens("input_tokens")
        + _tokens("cached_input_tokens")
        + _tokens("cache_write_tokens")
    )
    completion = _tokens("output_tokens")
    cost = (
        prompt / 1e6 * rates["prompt_per_1m"]
        + completion / 1e6 * rates["completion_per_1m"]
    )
    return cost, "assumed"


def pareto_rows(
    baseline_targets: Any,
    candidate_targets: Any,
    references: References | None,
    cost_basis: CostBasis | None,
) -> list[dict[str, Any]]:
    """M4: per-arm accuracy/cost/latency rows for the Pareto table.

    Every compared target contributes one row: accuracy with its Wilson
    interval, spend (measured or assumed, see `arm_cost`), and subject
    latency percentiles. Reference arms are tagged with their role so the
    report can anchor S/W comparisons.
    """
    rows: list[dict[str, Any]] = []
    for arm, targets in (
        ("baseline", baseline_targets),
        ("candidate", candidate_targets),
    ):
        if not isinstance(targets, list):
            continue
        for entry in targets:
            if not isinstance(entry, dict):
                continue
            role = None
            if references is not None:
                if entry.get("id") == references.strong:
                    role = "strong"
                elif entry.get("id") == references.weak:
                    role = "weak"
            cost, note = arm_cost(entry, cost_basis)
            rows.append(
                {
                    "arm": arm,
                    "id": entry.get("id"),
                    "role": role,
                    "accuracy": entry.get("accuracy"),
                    "accuracy_ci95": entry.get("accuracy_ci95"),
                    "cost_usd": cost,
                    "cost_note": note,
                    "latency_p50_s": entry.get("latency_p50_s"),
                    "latency_p95_s": entry.get("latency_p95_s"),
                }
            )
    return rows


def _load_run_cache(path: str | Path | None) -> dict[str, str]:
    """M4: run-id cache (key -> sr-bench run id); unreadable means empty."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _save_run_cache(path: str | Path | None, cache: dict[str, str]) -> None:
    if path is None:
        return
    Path(path).write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _dataset_identity(raw: Any) -> str | None:
    """Normalize an sr-bench dataset sha256 (bare hexdigest) to the A4 form."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = raw.strip()
    return value if value.startswith("sha256:") else "sha256:" + value


def _doc_dataset_identity(doc: Any, role: str, run_id: str) -> str | None:
    """Extract the dataset identity from one run document (M1).

    sr-bench verifies the dataset bytes at submission time and projects
    `manifest.dataset.sha256` into every run document, so this is what the
    run actually scored - not what the caller believes it scored.
    """
    if not isinstance(doc, dict):
        raise ContractError(
            f"{role} run {run_id} did not return a run document; "
            "cannot verify its dataset identity"
        )
    manifest = doc.get("manifest")
    dataset = manifest.get("dataset") if isinstance(manifest, dict) else None
    raw = dataset.get("sha256") if isinstance(dataset, dict) else None
    return _dataset_identity(raw)


def run_dataset_hash(cli: SrbenchCli, run_id: str) -> str | None:
    """The frozen dataset identity of one sr-bench run document (M1)."""
    return _doc_dataset_identity(cli.show(run_id), "run", run_id)


def require_run_datasets(
    spec: ExperimentSpec,
    cli: SrbenchCli,
    runs: list[tuple[str, str]],
    workload_split: str,
) -> None:
    """M1: every cited run must have scored the contract's own frozen split.

    `runs` carries (role, run_id) pairs. Raises ContractError - before any
    ledger write, so callers map it to exit 2 - when a run's recorded dataset
    identity differs from the workload split's live dataset identity (X3:
    the declared `srbench_dataset` sha256, else the legacy cases hash), or
    when the run document exposes no dataset identity at all: a verdict
    citing evidence it cannot bind to the contract is never written down.
    """
    expected = spec.workload.live_dataset_hash(workload_split)
    for role, run_id in runs:
        actual = run_dataset_hash(cli, run_id)
        if actual is None:
            raise ContractError(
                f"{role} run {run_id} exposes no dataset sha256 in its run "
                "document; cannot bind the verdict to the contract workload"
            )
        if actual != expected:
            raise ContractError(
                f"{role} run {run_id} scored dataset {actual}, but the contract "
                f"workload.{workload_split} is {expected}"
            )


def live_verdict_from_runs(
    *,
    spec: ExperimentSpec,
    cli: SrbenchCli,
    baseline_run_id: str,
    candidate_run_id: str,
    measurements: list[dict[str, Any]],
    incomplete_reasons: list[str] | None = None,
    workload_split: str = "dev",
) -> dict[str, Any]:
    """Compare two sr-bench runs and decide the live-tier verdict (A3).

    Guard floors come only from `spec.gates`; `measurements` carries the
    offline measured values. Undeclared measurement names are reported
    (in `undeclared_measurements` and the reasons) but never veto. An
    empty contract adds an explicit `gates_note`.

    M2: when the compare payload carries exactly one `comparisons[]` entry
    the verdict is decided from the paired confidence interval (see
    `paired_stats_from_comparison`); the entry's stats are returned in
    `paired` for the ledger.

    M3: both arms' output-quality rates come from the compare payload's
    target summaries; a rate beyond the spec's `evidence_quality` ceiling
    marks the evidence unreliable (INCONCLUSIVE) and the rates are
    returned in `evidence_quality` for the ledger.
    """
    # M1: both runs must have scored the contract's own frozen split, else
    # the verdict would cite evidence from a different question set.
    require_run_datasets(
        spec,
        cli,
        [("baseline", baseline_run_id), ("candidate", candidate_run_id)],
        workload_split,
    )
    comparison = cli.compare(baseline_run_id, candidate_run_id)
    incomplete = list(incomplete_reasons or [])
    baseline_metric, candidate_metric = _pair_metrics(
        comparison, spec.objective.metric_key, incomplete
    )
    # M2: paired statistics ride along when the compare payload carries them.
    stats = paired_stats_from_comparison(comparison)
    # M3: output-quality ceilings from the compare payload's target summaries.
    base_targets = (
        comparison.get("baseline_targets") if isinstance(comparison, dict) else None
    )
    cand_targets = (
        comparison.get("candidate_targets") if isinstance(comparison, dict) else None
    )
    selected = (
        comparison.get("baseline_selected_target_id")
        if isinstance(comparison, dict)
        else None
    )
    evidence = {
        "baseline": _evidence_entry(
            arm_evidence_quality(
                base_targets,
                routing_arm=False,
                selected_id=selected if isinstance(selected, str) else None,
            ),
            "baseline",
            spec.evidence_quality,
            incomplete,
        ),
        "candidate": _evidence_entry(
            arm_evidence_quality(cand_targets, routing_arm=True),
            "candidate",
            spec.evidence_quality,
            incomplete,
        ),
    }
    # M4: reference arms ride in the baseline run's target sections; their
    # values anchor ref-bound constraints and the Pareto table.
    reference_values = reference_metric_values(
        base_targets, spec.references, spec.objective.metric_key
    )
    pareto = pareto_rows(base_targets, cand_targets, spec.references, spec.cost_basis)

    guards, undeclared = resolve_gates(spec.gates, measurements)
    constraint_evidence = _constraint_evidence(
        comparison, list(spec.objective.constraints), incomplete
    )
    verdict = evaluate_verdict(
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        direction=spec.objective.direction,
        min_improvement_pct=spec.objective.min_improvement_pct,
        guards=guards,
        incomplete_reasons=incomplete,
        constraints=constraint_evidence,
        stats=stats,
        min_cases=spec.objective.min_cases,
        min_improvement_pp=spec.objective.min_improvement_pp,
        references=reference_values,
    )
    reasons = list(verdict.reasons)
    if undeclared:
        reasons.append(
            "undeclared guard measurements ignored (not declared in spec.gates): "
            + ", ".join(undeclared)
        )
    result: dict[str, Any] = {
        "verdict": verdict.verdict,
        "reasons": reasons,
        "improvement_pct": verdict.improvement_pct,
        "metrics": {
            "baseline": baseline_metric,
            "candidate": candidate_metric,
        },
        "gates": {
            g.name: (
                "pass" if g.passed else ("incomplete" if not g.evaluated else "fail")
            )
            for g in guards
        },
        "constraints": verdict.constraints,
        "undeclared_measurements": undeclared,
        "bench_refs": [baseline_run_id, candidate_run_id],
        "paired": stats.to_dict() if stats else None,
        "evidence_quality": evidence,
        "references": reference_values,
        "pareto": pareto,
    }
    if not spec.gates:
        result["gates_note"] = "no guard declared"
    return result


def extract_routing_distribution(report: Any) -> dict[str, int]:
    """Merge every `selected_models` counter in a preview report (I).

    The preview report carries per-target routing distributions as
    `selected_models` (model -> case count, verified in sr-bench's
    report builder). Depth-first merge keeps us decoupled from the exact
    nesting; a report without any counter (or a raw-text payload) yields
    an empty dict - missing evidence is reported, never guessed.
    """
    distribution: dict[str, int] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            selected = node.get("selected_models")
            if isinstance(selected, dict):
                for model, count in selected.items():
                    if isinstance(count, (int, float)) and not isinstance(count, bool):
                        distribution[str(model)] = distribution.get(
                            str(model), 0
                        ) + int(count)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(report)
    return distribution


def _submit_baseline_manifest(
    spec: ExperimentSpec,
    cli: SrbenchCli,
    manifest_source: str | Path,
    work_dir: Path,
    workload_split: str,
) -> str:
    """R2: turn a caller manifest into the completed AB/BA baseline run.

    The frozen AB/BA protocol needs a completed live baseline run whose
    manifest carries at least one `kind: single` target (sr-bench refuses
    to compare against a MoM-only baseline: "Baseline must contain a
    single-model target"). A caller manifest without a single target - the
    plain MoM router manifest - gets the spec's strong reference target
    injected as the single arm, embedded verbatim from the sr-bench
    registry (a submitted target must match the registry field-for-field
    anyway). When the manifest binds its dataset by file reference, the
    recorded sha256 is checked against the contract split before any run
    is spent on it.
    """
    path = Path(manifest_source)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read baseline manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError(f"baseline manifest {path} is not a JSON object")
    targets = raw.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ContractError(f"baseline manifest {path} carries no targets")
    if not any(
        isinstance(entry, dict) and entry.get("kind") == "single" for entry in targets
    ):
        if spec.references is None:
            raise ContractError(
                f"baseline manifest {path} has no single-model target and the "
                "spec declares no references; cannot construct the S arm"
            )
        alias = spec.references.strong
        match = next(
            (entry for entry in cli.targets() if entry.get("id") == alias), None
        )
        if match is None:
            raise ContractError(
                f"reference target {alias!r} is not registered in sr-bench; "
                "cannot construct the single-model baseline arm"
            )
        raw["targets"] = [match, *targets]
    dataset = raw.get("dataset")
    if isinstance(dataset, dict) and dataset.get("sha256"):
        # X3: bound via the split's srbench_dataset hash when declared,
        # else the legacy cases hash.
        expected = spec.workload.live_dataset_hash(workload_split)
        actual = _dataset_identity(dataset.get("sha256"))
        if actual != expected:
            raise ContractError(
                f"baseline manifest {path} binds dataset {actual}, but the "
                f"contract workload.{workload_split} is {expected}"
            )
    # R2: the submission snapshot (what was sent to sr-bench, injection
    # included) is kept apart from verify_live's own `baseline-manifest.json`
    # dump of the run document's echoed manifest below.
    frozen = work_dir / "baseline-manifest.submitted.json"
    frozen.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    doc = cli.run_manifest(frozen)
    run_id = doc.get("id") if isinstance(doc, dict) else None
    status = doc.get("status") if isinstance(doc, dict) else None
    if not run_id or status != "completed":
        raise ContractError(
            f"baseline manifest run ended {status!r}; AB/BA pairs need a "
            "completed baseline"
        )
    return run_id


def verify_live(
    *,
    spec: ExperimentSpec,
    cli: SrbenchCli,
    baseline_run_id: str | None,
    targets: list[str],
    repetitions: int,
    measurements: list[dict[str, Any]],
    incomplete_reasons: list[str] | None = None,
    work_dir: str | Path,
    workload_split: str = "dev",
    run_cache: str | Path | None = None,
    baseline_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Drive sr-bench for one candidate and decide the live verdict (I).

    Per repetition, BOTH arms are measured live in AB/BA order: the
    baseline arm re-runs the baseline run's frozen manifest, the candidate
    arm is a fresh `candidate-plan` (inheriting the same frozen protocol
    and questions). Each completed pair is compared immediately; medians
    and ranges stay descriptive (abba.paired_summary). The final verdict
    reuses `evaluate_verdict` with the per-label medians - any incomplete
    pair, missing metric, or unmeasurable constraint lands in
    `incomplete_reasons` and turns the verdict INCONCLUSIVE.

    M2: the pairs' `comparisons[]` statistics merge conservatively (counts
    sum, intervals take the widest span) and drive the paired decision, so
    every repeated case counts toward `min_cases`.

    M3: the pairs' output-quality rates merge conservatively (worst pair)
    against the spec's `evidence_quality` ceilings and are returned in
    `evidence_quality` for the ledger.

    M4: when the spec declares `references`, each reference arm runs once
    as a single-model run on the frozen protocol (the baseline manifest
    with its targets replaced by the registry document); run ids are
    cached by (dataset hash, protocol hash, alias) in `run_cache` so
    several candidates reuse the same reference runs. The arms' values
    anchor ref-bound constraints, and a Pareto table (accuracy with its
    interval, measured-or-assumed cost, latency percentiles) lands in
    `pareto` for the report.

    `targets` are the registered sr-bench MoM target ids the router is
    serving the candidate config on. The preview run is diagnostic only:
    a failed preview is recorded in the reasons but does not fail the
    verdict (it produces no quality numbers anyway).

    R2: exactly one of `baseline_run_id` (a terminal completed baseline)
    or `baseline_manifest` (a manifest file to submit as the baseline) is
    required. A manifest without a single-model target gets the spec's
    strong reference injected as the S arm before submission, so a plain
    MoM router manifest can seed the AB/BA protocol directly.
    """
    if repetitions < 1:
        raise ContractError("verify needs at least one repetition")
    if (baseline_run_id is None) == (baseline_manifest is None):
        raise ContractError(
            "verify needs exactly one of baseline_run_id or baseline_manifest"
        )
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if baseline_manifest is not None:
        baseline_run_id = _submit_baseline_manifest(
            spec, cli, baseline_manifest, work_dir, workload_split
        )

    baseline_doc = cli.show(baseline_run_id)
    if not isinstance(baseline_doc, dict):
        raise ContractError(
            f"baseline run {baseline_run_id} did not return a run document"
        )
    if baseline_doc.get("status") != "completed":
        raise ContractError(
            f"baseline run {baseline_run_id} is {baseline_doc.get('status')!r}, not completed; "
            "AB/BA pairs must inherit a terminal completed baseline protocol"
        )
    manifest = baseline_doc.get("manifest")
    if not isinstance(manifest, dict):
        raise ContractError(
            f"baseline run {baseline_run_id} does not expose its frozen manifest"
        )
    manifest_path = work_dir / "baseline-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # M1: the frozen baseline protocol must be the contract's own split (X3:
    # bound via the split's srbench_dataset hash when declared, else the
    # legacy cases hash).
    actual = _doc_dataset_identity(baseline_doc, "baseline", baseline_run_id)
    expected = spec.workload.live_dataset_hash(workload_split)
    if actual is None:
        raise ContractError(
            f"baseline run {baseline_run_id} exposes no dataset sha256; "
            "cannot bind the AB/BA pairs to the contract workload"
        )
    if actual != expected:
        raise ContractError(
            f"baseline run {baseline_run_id} scored dataset {actual}, but the "
            f"contract workload.{workload_split} is {expected}"
        )

    incomplete = list(incomplete_reasons or [])
    diagnostics: list[str] = []
    bench_refs = [baseline_run_id]

    # preview first: routing distribution for the ledger notes (no quality).
    # Diagnostic only - a failed preview is recorded but never fails the
    # verdict, since the pairs below carry all quality evidence.
    preview_run = cli.candidate_plan(baseline_run_id, targets=targets, mode="preview")
    preview_id = preview_run.get("id")
    preview_info: dict[str, Any] = {
        "run_id": preview_id,
        "status": preview_run.get("status"),
    }
    bench_refs.append(preview_id)
    if preview_run.get("status") == "completed":
        preview_info["distribution"] = extract_routing_distribution(
            cli.report(preview_id, work_dir / "preview-report.json")
        )
    else:
        diagnostics.append(
            f"preview run {preview_id} ended {preview_run.get('status')!r}; no routing distribution"
        )

    # AB/BA pairs: both arms live, alternating execution order per repetition
    trials: dict[int, dict[str, Any]] = {}
    for label, rep in abba.schedule(repetitions):
        arm = trials.setdefault(rep, {})
        if label == abba.BASELINE:
            run = cli.run_manifest(manifest_path)
        else:
            run = cli.candidate_plan(baseline_run_id, targets=targets, mode="live")
        arm[label] = {"run_id": run.get("id"), "status": run.get("status")}
        bench_refs.append(run.get("id"))

    # per-pair compare + metric/constraint pairing
    metric_key = spec.objective.metric_key
    rows: list[dict[str, Any]] = []
    pair_stats: list[PairedStats] = []
    pair_evidence: list[
        tuple[tuple[float | None, float | None], tuple[float | None, float | None]]
    ] = []
    # M4: the last completed pair's target sections feed the Pareto table.
    last_sections: tuple[Any, Any] | None = None
    constraint_values: dict[str, list[tuple[float | None, float | None]]] = {
        constraint.name: [] for constraint in spec.objective.constraints
    }
    for rep in sorted(trials):
        base_run = trials[rep].get(abba.BASELINE, {})
        cand_run = trials[rep].get(abba.CANDIDATE, {})
        if (
            base_run.get("status") != "completed"
            or cand_run.get("status") != "completed"
        ):
            incomplete.append(
                f"pair {rep}: baseline ended {base_run.get('status')!r}, candidate ended {cand_run.get('status')!r}"
            )
            continue
        comparison = cli.compare(base_run["run_id"], cand_run["run_id"])
        # X8: keep each pair's raw compare payload on disk so the verdict's
        # merged paired statistics stay auditable against their sources.
        (work_dir / f"compare-rep{rep}.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stats = paired_stats_from_comparison(comparison)
        if stats is not None:
            pair_stats.append(stats)
        elif spec.objective.min_improvement_pp is not None:
            # M2: the pp bar is decided from paired CIs only. A completed
            # pair without paired statistics must not silently shrink the
            # evidence to the remaining pairs - one pair's interval then
            # decided a PASS that two pairs were contracted to support.
            incomplete.append(
                f"pair {rep}: compare payload carries no paired statistics "
                "(objective.min_improvement_pp requires paired evidence)"
            )
        # M3: per-pair output quality; merged worst-case below.
        base_sec = (
            comparison.get("baseline_targets") if isinstance(comparison, dict) else None
        )
        cand_sec = (
            comparison.get("candidate_targets")
            if isinstance(comparison, dict)
            else None
        )
        selected = (
            comparison.get("baseline_selected_target_id")
            if isinstance(comparison, dict)
            else None
        )
        pair_evidence.append(
            (
                arm_evidence_quality(
                    base_sec,
                    routing_arm=False,
                    selected_id=selected if isinstance(selected, str) else None,
                ),
                arm_evidence_quality(cand_sec, routing_arm=True),
            )
        )
        last_sections = (base_sec, cand_sec)
        baseline_value, candidate_value = _pair_metrics(
            comparison, metric_key, incomplete
        )
        if baseline_value is not None:
            rows.append(
                {"label": abba.BASELINE, "metrics": {metric_key: baseline_value}}
            )
        if candidate_value is not None:
            rows.append(
                {"label": abba.CANDIDATE, "metrics": {metric_key: candidate_value}}
            )
        for constraint in spec.objective.constraints:
            constraint_values[constraint.name].append(
                _pair_metrics(comparison, constraint.metric, incomplete)
            )

    summary = abba.paired_summary(rows, metric_key)
    baseline_metric = (
        summary[abba.BASELINE]["median"] if summary[abba.BASELINE] else None
    )
    candidate_metric = (
        summary[abba.CANDIDATE]["median"] if summary[abba.CANDIDATE] else None
    )

    # constraints: median over pairs; any unmeasured pair stays unevaluated
    constraint_evidence: list[dict[str, Any]] = []
    for constraint in spec.objective.constraints:
        pairs = constraint_values[constraint.name]
        measured = [(b, c) for b, c in pairs if b is not None and c is not None]
        if repetitions and len(measured) == repetitions:
            baseline_c = statistics.median(b for b, _ in measured)
            candidate_c = statistics.median(c for _, c in measured)
        else:
            incomplete.append(
                f"constraint {constraint.name}: measured in {len(measured)} of {repetitions} pairs"
            )
            baseline_c = candidate_c = None
        constraint_evidence.append(
            {
                "name": constraint.name,
                "metric": constraint.metric,
                "op": constraint.op,
                "bound": constraint.bound,
                "tolerance_pct": constraint.tolerance_pct,
                "tolerance_pp": constraint.tolerance_pp,
                "factor": constraint.factor,
                "baseline": baseline_c,
                "candidate": candidate_c,
            }
        )

    # M4: reference arms - one single-model run per declared alias on the
    # frozen protocol, cached by (dataset hash, protocol hash, alias) so
    # several candidates reuse the same reference runs. A failed reference
    # run is incomplete evidence, never a zero.
    references_evidence: dict[str, Any] = {}
    reference_values: dict[str, float | None] = {}
    ref_rows: list[dict[str, Any]] = []
    if spec.references is not None:
        cache = _load_run_cache(run_cache)
        manifest_hash = (
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )
        cache_dirty = False
        for role in ("strong", "weak"):
            alias = getattr(spec.references, role)
            key = f"{actual}:{manifest_hash}:{alias}"
            run_id = cache.get(key)
            cached = run_id is not None
            completed = cached
            if not cached:
                registry_doc = next(
                    (entry for entry in cli.targets() if entry.get("id") == alias),
                    None,
                )
                if registry_doc is None:
                    incomplete.append(
                        f"reference arm {role!r}: target {alias!r} is not "
                        "registered in sr-bench"
                    )
                else:
                    # Single-model targets have no candidate-plan protocol
                    # (sr-bench reserves candidate plans for MoM targets);
                    # submit the frozen baseline manifest with its targets
                    # replaced by the registry document - the same
                    # field-for-field validation path as the injected S arm.
                    ref_manifest = dict(manifest)
                    # The frozen echo carries the baseline's review receipt;
                    # after the targets swap it no longer matches this
                    # derived plan, so drop it and let the service freeze
                    # the reference plan fresh.
                    ref_manifest.pop("plan_sha256", None)
                    ref_manifest["targets"] = [registry_doc]
                    ref_path = work_dir / f"reference-{role}-manifest.json"
                    ref_path.write_text(
                        json.dumps(ref_manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    ref_run = cli.run_manifest(ref_path)
                    run_id = ref_run.get("id")
                    completed = ref_run.get("status") == "completed"
                    if completed:
                        cache[key] = run_id
                        cache_dirty = True
                    else:
                        incomplete.append(
                            f"reference arm {role!r} run {run_id} ended "
                            f"{ref_run.get('status')!r}"
                        )
            references_evidence[role] = {
                "alias": alias,
                "run_id": run_id,
                "cached": cached,
            }
            if run_id:
                bench_refs.append(run_id)
            if run_id and completed:
                ref_report = cli.report(
                    run_id, work_dir / f"reference-{role}-report.json"
                )
                ref_targets = (
                    ref_report.get("summary", {}).get("targets")
                    if isinstance(ref_report, dict)
                    else None
                )
                entry = next(
                    (
                        t
                        for t in ref_targets or []
                        if isinstance(t, dict) and t.get("id") == alias
                    ),
                    None,
                )
                values = extract_metric_values(entry, metric_key) if entry else []
                reference_values[role] = values[0] if values else None
                cost, note = arm_cost(entry, spec.cost_basis) if entry else (None, None)
                references_evidence[role].update(
                    {
                        "metric": reference_values[role],
                        "cost_usd": cost,
                        "cost_note": note,
                    }
                )
                if entry:
                    ref_rows.append(
                        {
                            "arm": "reference",
                            "id": alias,
                            "role": role,
                            "accuracy": entry.get("accuracy"),
                            "accuracy_ci95": entry.get("accuracy_ci95"),
                            "cost_usd": cost,
                            "cost_note": note,
                            "latency_p50_s": entry.get("latency_p50_s"),
                            "latency_p95_s": entry.get("latency_p95_s"),
                        }
                    )
        if cache_dirty:
            _save_run_cache(run_cache, cache)

    # M3: conservative merge - each rate takes the worst observed pair; a
    # pair without diagnostics never masks a breach seen in another pair.
    evidence = {
        "baseline": _evidence_entry(
            _worst_rates([base for base, _ in pair_evidence]),
            "baseline",
            spec.evidence_quality,
            incomplete,
        ),
        "candidate": _evidence_entry(
            _worst_rates([cand for _, cand in pair_evidence]),
            "candidate",
            spec.evidence_quality,
            incomplete,
        ),
    }
    # M4: Pareto rows - the last completed pair's arms plus the references.
    pareto: list[dict[str, Any]] = []
    if last_sections is not None:
        pareto.extend(
            pareto_rows(
                last_sections[0], last_sections[1], spec.references, spec.cost_basis
            )
        )
    pareto.extend(ref_rows)

    guards, undeclared = resolve_gates(spec.gates, measurements)
    # M2: the pairs' statistics merge conservatively and drive the paired
    # decision (min_cases reads the distinct-case count, not the sum over
    # repeated runs - see _merge_paired_stats).
    stats = _merge_paired_stats(pair_stats)
    verdict = evaluate_verdict(
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        direction=spec.objective.direction,
        min_improvement_pct=spec.objective.min_improvement_pct,
        guards=guards,
        incomplete_reasons=incomplete,
        constraints=constraint_evidence,
        stats=stats,
        min_cases=spec.objective.min_cases,
        min_improvement_pp=spec.objective.min_improvement_pp,
        references=reference_values,
    )
    reasons = list(verdict.reasons)
    reasons.extend(diagnostics)
    if undeclared:
        reasons.append(
            "undeclared guard measurements ignored (not declared in spec.gates): "
            + ", ".join(undeclared)
        )
    result: dict[str, Any] = {
        "verdict": verdict.verdict,
        "reasons": reasons,
        "improvement_pct": verdict.improvement_pct,
        "metrics": {
            "baseline": baseline_metric,
            "candidate": candidate_metric,
        },
        "gates": {
            g.name: (
                "pass" if g.passed else ("incomplete" if not g.evaluated else "fail")
            )
            for g in guards
        },
        "constraints": verdict.constraints,
        "undeclared_measurements": undeclared,
        "bench_refs": bench_refs,
        # R2: the resolved baseline run id (either --baseline-run-id or the run
        # submitted from --baseline-manifest), so the CLI ledger can cite it.
        "baseline_run_id": baseline_run_id,
        "trials": {str(rep): trials[rep] for rep in sorted(trials)},
        "abba_summary": summary,
        "preview": preview_info,
        "paired": stats.to_dict() if stats else None,
        "evidence_quality": evidence,
        "references": references_evidence,
        "pareto": pareto,
    }
    if not spec.gates:
        result["gates_note"] = "no guard declared"
    return result
