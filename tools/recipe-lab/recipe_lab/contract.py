"""Experiment contract: ExperimentSpec / Candidate / Gate / oracle policy.

The contract is the source of truth for one improvement campaign. It is
loaded from YAML (api_version: recipe-lab/v1alpha1) and validated eagerly:
invalid contracts are rejected at load time with actionable errors, never
mid-experiment.

Layout follows the recipe-lab design §5.3:
  objective:    metric_key / direction / min_improvement_pct or
                min_improvement_pp (M2, paired CI) / min_cases / replay_metric
                / constraints (E: economic conditions, distinct from gates)
  workload:     manifest / content_hash / profile
  attribution:  oracle_policy (label -> model mapping, or F2 rules/default)
  gates:        [{name, metric, floor}]  floor = "baseline" | number
  budget:       rounds / max_minutes_per_trial
  stack:        mode / endpoint / baseline_ref
  cost_model:   optional model -> relative-cost map (E2); the replay tier
                uses it to compute the routed_cost_index proxy and to warn
                on constraint-violating candidates

A5: the `evaluation` section is retired - replay and live are separate
commands with CLI-fixed tiers, not contracted per experiment.

Dataset identity (A4): `content_hash = "sha256:" + sha256(<dataset JSONL raw
bytes>)` and `manifest` is the path of that JSONL relative to the spec file.
`ExperimentSpec.verify_workload()` re-hashes the manifest file and rejects a
mismatch before any experiment command touches the data, so a drifted dataset
can never silently join two rounds of results.

X3: a split may additionally declare `srbench_dataset` + `srbench_sha256` -
the sr-bench format (id/messages/answer) rendering of the same cases that
live evidence binds to. Replay keeps reading `manifest`; live compares the
run documents' dataset sha256 against `srbench_sha256`. verify_workload
re-hashes that file too and checks its ids pair 1:1 with the cases manifest;
`cases to-srbench` generates it deterministically from the cases file.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import API_VERSION

DIRECTIONS = {"minimize", "maximize"}
PROFILES = {"smoke", "quick", "standard"}
STACK_MODES = {"external", "serve"}
# A1: the replay tier ranks candidates with exactly one contracted metric.
# `metric_key` stays the live sr-bench metric; `replay_metric` is what the
# replay ledger records and `ledger top --spec` ranks by.
REPLAY_METRICS = {"routed_accuracy"}
# M2: default minimum paired sample for a live verdict (objective.min_cases).
DEFAULT_MIN_CASES = 200
# M3: default output-quality ceilings for live evidence (evidence_quality).
DEFAULT_MAX_EVIDENCE_PCT = 2.0
# B: the frozen dataset is split three ways; frozen_test is quarantined from
# replay/propose work and may only be touched by live-verdict --final, once.
SPLIT_NAMES = ("calib", "dev", "frozen_test")
RUNNABLE_SPLITS = ("calib", "dev")
TIERS = {"replay", "preview", "live", "route_replay"}
CANDIDATE_KINDS = {"config", "artifact"}
VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE", "RANKED"}
# Numeric predicate ops for oracle-policy conditions and truth facts, with
# the router's semantics (pkg/decision engine numericPredicateMatches: e.g.
# gte fails when value < bound).
PREDICATE_OPS = ("gte", "lte", "gt", "lt")
LIVE_VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}
REPLAY_VERDICTS = {"RANKED"}
# AG3: how a case's truth label names evidence. "decision" labels ARE the
# intended decision (the legacy F1 mapping); "behavior" labels describe an
# observed outcome and only constrain the model route through the oracle
# mapping (or F2 rules) - a behavior label can never ground a
# decision_mismatch verdict.
TRUTH_SPACES = {"decision", "behavior"}
DEFAULT_TRUTH_SPACE = "decision"


class ContractError(ValueError):
    """Raised when a contract file or candidate violates the v1alpha1 schema."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_mapping(value: Any, where: str) -> dict:
    _require(isinstance(value, dict), f"{where} must be a mapping")
    return value


def _require_number(value: Any, where: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        f"{where} must be a finite number",
    )
    return float(value)


def dataset_content_hash(path: str | Path) -> str:
    """The frozen dataset identity: 'sha256:' + sha256 of the file's raw bytes."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_runnable_split(split: str) -> None:
    """frozen_test is quarantined (B): replay/propose/attribute may not touch
    it. live-verdict --final is the only consumer, via its own gate."""
    if split == "frozen_test":
        raise ContractError("frozen_test is quarantined; use live-verdict --final")


def _read_split_ids(manifest: Path, split: str) -> tuple[set[str], set[str]]:
    """case_ids and prompt hashes of one split manifest (disjointness input)."""
    case_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    with manifest.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"split {split!r} line {line_no} is not valid JSON: {exc}"
                ) from exc
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ContractError(
                    f"split {split!r} line {line_no} is missing a non-empty case_id"
                )
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ContractError(
                    f"split {split!r} line {line_no} is missing a non-empty prompt"
                )
            case_ids.add(case_id)
            prompt_hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    return case_ids, prompt_hashes


def _read_srbench_case_ids(path: Path) -> set[str]:
    """The `id` set of one sr-bench dataset JSONL (X3 pairing-check input)."""
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{line_no} is not valid JSON: {exc}"
                ) from exc
            case_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(case_id, str) or not case_id:
                raise ContractError(f"{path}:{line_no} is missing a non-empty id")
            ids.add(case_id)
    return ids


def _read_model_eval_task_ids(path: Path) -> set[str]:
    """The `task_id` set of one model_eval tasks JSONL (Y3 pairing input)."""
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{line_no} is not valid JSON: {exc}"
                ) from exc
            task_id = row.get("task_id") if isinstance(row, dict) else None
            if not isinstance(task_id, str) or not task_id:
                raise ContractError(f"{path}:{line_no} is missing a non-empty task_id")
            ids.add(task_id)
    return ids


def _check_split_disjointness(base: Path, workload: Workload) -> None:
    """The three-way split must be disjoint: no case_id and no prompt content
    shared between two splits (else calibration data leaks into the frozen
    test and the final verdict is no longer honest)."""
    seen: dict[str, tuple[set[str], set[str]]] = {
        name: _read_split_ids(base / split.manifest, name)
        for name, split in workload.splits.items()
    }

    names = sorted(seen)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared_ids = seen[left][0] & seen[right][0]
            shared_prompts = seen[left][1] & seen[right][1]
            if shared_ids or shared_prompts:
                detail = []
                if shared_ids:
                    detail.append(f"case_id(s) {sorted(shared_ids)[:3]}")
                if shared_prompts:
                    detail.append(f"{len(shared_prompts)} shared prompt(s)")
                raise ContractError(
                    f"workload splits overlap between {left!r} and {right!r}: {'; '.join(detail)}; "
                    "splits must be disjoint (case_id and prompt)"
                )


def _check_cross_workload_disjointness(
    base: Path, workloads: tuple[Workload, ...]
) -> None:
    """M5: no case_id or prompt may appear in two different workloads - a
    transfer workload sharing cases with a tune workload would be tuned on,
    which defeats the transfer quarantine."""
    seen: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    for workload in workloads:
        for name, split in workload.splits.items():
            seen[(workload.name, name)] = _read_split_ids(base / split.manifest, name)

    keys = sorted(seen)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            if left[0] == right[0]:
                continue  # splits within one workload: _check_split_disjointness
            shared_ids = seen[left][0] & seen[right][0]
            shared_prompts = seen[left][1] & seen[right][1]
            if shared_ids or shared_prompts:
                detail = []
                if shared_ids:
                    detail.append(f"case_id(s) {sorted(shared_ids)[:3]}")
                if shared_prompts:
                    detail.append(f"{len(shared_prompts)} shared prompt(s)")
                raise ContractError(
                    f"workloads {left[0]!r} and {right[0]!r} overlap (splits "
                    f"{left[1]!r} / {right[1]!r}): {'; '.join(detail)}; "
                    "different workloads must be disjoint (case_id and prompt)"
                )


CONSTRAINT_OPS = ("gte", "lte")
# M4: constraint bounds may pin a reference arm instead of the baseline;
# the roles name the single-model targets declared in `references`.
# C8: `ref:random_mix` pins the cost-matched random mix the model_eval
# driver computes from the candidate arm's routing distribution and the
# reference arms' per-case results (evidence-side, not a declared arm).
REF_BOUNDS = ("ref:strong", "ref:weak", "ref:random_mix")
# M4: where the verdict's cost numbers come from (cost_basis.source).
COST_BASIS_SOURCES = {"provider_pricing", "cost_model"}
# M5: workload vocabulary. `kind` selects the request shape, `driver` the
# evaluation harness (C7), `role` the quarantine (tune vs frozen_test rule).
WORKLOAD_KINDS = {"mcq", "open", "agentic"}
WORKLOAD_DRIVERS = {"srbench", "model_eval"}
# AG1: agentic workloads name the sr-bench adapter that grades them.
AGENTIC_BENCHMARKS = {"terminal-bench-2.1", "tau3"}
# AG7: agentic stability hard gates. Every agentic workload carries the
# full set (defaults below); a workload may override values, not the set.
# An exceeded ceiling vetoes the live verdict with FAIL; `baseline` as a
# value (only valid for max_tool_loop_interrupted) pins the baseline arm's
# measured interruption count. Z5: upstream #5567 releases the tool-loop
# lock when classification drifts to a different decision, so a raw
# tool-loop switch count cannot be a zero ceiling - the drift transfer is
# in-design. `max_unprotected_tool_loop_switches` caps the continuity
# tool-loop total minus the replay drift releases (the residual the
# identity-missing / bypassed switches live in), and
# `max_decision_drift_switches_per_task` caps the drift transfers per
# replay session; exceeding the drift ceiling points at threshold
# calibration (propose-calibrated), not a routing regression.
AGENTIC_GATE_NAMES = (
    "max_unprotected_tool_loop_switches",
    "max_selection_rejected",
    "max_tool_loop_interrupted",
    "max_mean_switches_per_task",
    "max_decision_drift_switches_per_task",
)
DEFAULT_AGENTIC_GATES: dict[str, float | str] = {
    "max_unprotected_tool_loop_switches": 0,
    "max_selection_rejected": 0,
    "max_tool_loop_interrupted": "baseline",
    "max_mean_switches_per_task": 1.0,
    "max_decision_drift_switches_per_task": 0.2,
}
WORKLOAD_ROLES = {"tune", "transfer"}
# M5: default auto "not worse" guard per workload - the workload's paired
# quality-delta CI lower bound must not fall below -guard_pp percentage
# points (the candidate may lose at most 1pp on any workload by default).
DEFAULT_GUARD_PP = 1.0


@dataclass(frozen=True)
class Constraint:
    """One economic constraint on the live verdict (E).

    Constraints are distinct from gates: a gate is a safety veto, a
    constraint is an economic condition (e.g. cost must not regress).
    `bound` is a number, the literal "baseline" (the baseline arm's
    measured value of the same metric), or a reference pin "ref:strong" /
    "ref:weak" / "ref:random_mix" (M4: the named single-model arm's
    measured value; C8: the model_eval driver's cost-matched random mix).
    `tolerance_pct` relaxes the bound by that percentage in the permissive
    direction; `tolerance_pp` (M4) relaxes it by absolute percentage
    points instead (for 0-1 fraction metrics, e.g. accuracy floors);
    declare at most one. `factor` (M4) multiplies a reference bound (e.g.
    cost <= 0.7 x the strong arm) and is only valid with a ref bound.
    """

    name: str
    metric: str
    op: str
    bound: (
        str | float
    )  # "baseline", "ref:strong"/"ref:weak"/"ref:random_mix", or a number
    tolerance_pct: float = 0.0
    # M4: absolute pp relaxation / reference multiplier (0 and None stay
    # out of the serialized form for round-trip stability).
    tolerance_pp: float = 0.0
    factor: float | None = None

    @classmethod
    def from_dict(cls, raw: dict, index: int) -> Constraint:
        name = raw.get("name")
        _require(
            isinstance(name, str) and name.strip(),
            f"constraints[{index}].name must be a non-empty string",
        )
        metric = raw.get("metric")
        _require(
            isinstance(metric, str) and all(part.strip() for part in metric.split(".")),
            f"constraints[{index}].metric must be a non-empty dotted JSON key",
        )
        op = raw.get("op")
        _require(
            op in CONSTRAINT_OPS,
            f"constraints[{index}].op must be one of {sorted(CONSTRAINT_OPS)}",
        )
        bound = raw.get("bound")
        if bound != "baseline" and bound not in REF_BOUNDS:
            bound = _require_number(bound, f"constraints[{index}].bound")
        tolerance = _require_number(
            raw.get("tolerance_pct", 0.0), f"constraints[{index}].tolerance_pct"
        )
        _require(tolerance >= 0, f"constraints[{index}].tolerance_pct must be >= 0")
        tolerance_pp = _require_number(
            raw.get("tolerance_pp", 0.0), f"constraints[{index}].tolerance_pp"
        )
        _require(tolerance_pp >= 0, f"constraints[{index}].tolerance_pp must be >= 0")
        _require(
            tolerance == 0.0 or tolerance_pp == 0.0,
            f"constraints[{index}]: declare tolerance_pct or tolerance_pp, not both",
        )
        factor_raw = raw.get("factor")
        factor: float | None = None
        if factor_raw is not None:
            factor = _require_number(factor_raw, f"constraints[{index}].factor")
            _require(factor > 0, f"constraints[{index}].factor must be > 0")
            _require(
                bound in REF_BOUNDS,
                f"constraints[{index}].factor is only valid with a reference bound "
                f"(one of {sorted(REF_BOUNDS)})",
            )
        return cls(
            name=name,
            metric=metric,
            op=op,
            bound=bound,
            tolerance_pct=tolerance,
            tolerance_pp=tolerance_pp,
            factor=factor,
        )

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "metric": self.metric,
            "op": self.op,
            "bound": self.bound,
            "tolerance_pct": self.tolerance_pct,
        }
        if self.tolerance_pp:
            out["tolerance_pp"] = self.tolerance_pp
        if self.factor is not None:
            out["factor"] = self.factor
        return out


@dataclass(frozen=True)
class Objective:
    metric_key: str
    direction: str
    min_improvement_pct: float
    replay_metric: str = "routed_accuracy"
    # M2: paired-decision thresholds. `min_cases` is the minimum paired
    # sample a live verdict needs (fewer -> INCONCLUSIVE). `min_improvement_pp`
    # is an absolute percentage-point bar applied to the paired confidence
    # interval; it is mutually exclusive with the relative `min_improvement_pct`,
    # which cannot be applied to an absolute interval.
    min_cases: int = 200
    min_improvement_pp: float | None = None
    constraints: tuple[Constraint, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict) -> Objective:
        metric_key = raw.get("metric_key")
        _require(
            isinstance(metric_key, str) and metric_key.strip(),
            "objective.metric_key must be a non-empty string",
        )
        _require(
            all(part.strip() for part in metric_key.split(".")),
            "objective.metric_key must be a non-empty dotted JSON key",
        )
        direction = raw.get("direction")
        _require(
            direction in DIRECTIONS,
            f"objective.direction must be one of {sorted(DIRECTIONS)}",
        )
        pct = _require_number(
            raw.get("min_improvement_pct", 0.0), "objective.min_improvement_pct"
        )
        _require(pct >= 0, "objective.min_improvement_pct must be >= 0")
        # M2: min_improvement_pp (absolute, applied to the paired CI) and
        # min_improvement_pct (relative, applied to point estimates) are
        # mutually exclusive.
        pp_raw = raw.get("min_improvement_pp")
        min_pp: float | None = None
        if pp_raw is not None:
            min_pp = _require_number(pp_raw, "objective.min_improvement_pp")
            _require(min_pp >= 0, "objective.min_improvement_pp must be >= 0")
            _require(
                "min_improvement_pct" not in raw,
                "objective.min_improvement_pp and min_improvement_pct are mutually "
                "exclusive; declare min_improvement_pp for the paired CI decision",
            )
        min_cases = raw.get("min_cases", DEFAULT_MIN_CASES)
        _require(
            isinstance(min_cases, int)
            and not isinstance(min_cases, bool)
            and min_cases >= 1,
            "objective.min_cases must be an integer >= 1",
        )
        replay_metric = raw.get("replay_metric", "routed_accuracy")
        _require(
            replay_metric in REPLAY_METRICS,
            f"objective.replay_metric must be one of {sorted(REPLAY_METRICS)} (got {replay_metric!r})",
        )
        constraints_raw = raw.get("constraints", [])
        _require(
            isinstance(constraints_raw, list), "objective.constraints must be a list"
        )
        constraints = tuple(
            Constraint.from_dict(c, i) for i, c in enumerate(constraints_raw)
        )
        return cls(
            metric_key=metric_key,
            direction=direction,
            min_improvement_pct=pct,
            replay_metric=replay_metric,
            min_cases=min_cases,
            min_improvement_pp=min_pp,
            constraints=constraints,
        )

    def to_dict(self) -> dict:
        out = {
            "metric_key": self.metric_key,
            "direction": self.direction,
            "min_improvement_pct": self.min_improvement_pct,
            "replay_metric": self.replay_metric,
        }
        if self.min_cases != DEFAULT_MIN_CASES:
            out["min_cases"] = self.min_cases
        if self.min_improvement_pp is not None:
            out["min_improvement_pp"] = self.min_improvement_pp
        if self.constraints:
            out["constraints"] = [c.to_dict() for c in self.constraints]
        return out


@dataclass(frozen=True)
class SplitSpec:
    """One named slice of the frozen dataset (B).

    X3: one slice carries two files of the same cases. `manifest` is the
    recipe-lab cases JSONL (case_id/prompt/truth) that replay reads;
    `srbench_dataset`/`srbench_sha256` optionally declare the sr-bench
    format (id/messages/answer) rendering of the same cases that the live
    tier binds its evidence to. Y3: `model_eval_tasks`/
    `model_eval_tasks_sha256` optionally declare the model_eval tasks
    rendering (task_id/prompt_user/answer, the deterministic `cases
    to-model-eval` output) that the model_eval driver binds its arms to.
    Each rendering pair is declared together or not at all; verify_workload
    re-hashes every declared rendering and checks its ids pair 1:1 with
    the cases manifest.
    """

    manifest: str
    content_hash: str
    srbench_dataset: str | None = None
    srbench_sha256: str | None = None
    model_eval_tasks: str | None = None
    model_eval_tasks_sha256: str | None = None

    @classmethod
    def from_dict(cls, raw: dict, name: str, prefix: str = "workload") -> SplitSpec:
        manifest = raw.get("manifest")
        _require(
            isinstance(manifest, str) and manifest.strip(),
            f"{prefix}.splits.{name}.manifest must be a non-empty string",
        )
        content_hash = raw.get("content_hash")
        _require(
            isinstance(content_hash, str) and content_hash.strip(),
            f"{prefix}.splits.{name}.content_hash is required (frozen dataset hash)",
        )
        srbench_dataset = raw.get("srbench_dataset")
        srbench_sha256 = raw.get("srbench_sha256")
        _require(
            (srbench_dataset is None) == (srbench_sha256 is None),
            f"{prefix}.splits.{name} must declare srbench_dataset and "
            "srbench_sha256 together (the sr-bench rendering of the same cases)",
        )
        if srbench_dataset is not None:
            _require(
                isinstance(srbench_dataset, str) and srbench_dataset.strip(),
                f"{prefix}.splits.{name}.srbench_dataset must be a non-empty string",
            )
            _require(
                isinstance(srbench_sha256, str) and srbench_sha256.strip(),
                f"{prefix}.splits.{name}.srbench_sha256 must be a non-empty string",
            )
        model_eval_tasks = raw.get("model_eval_tasks")
        model_eval_tasks_sha256 = raw.get("model_eval_tasks_sha256")
        _require(
            (model_eval_tasks is None) == (model_eval_tasks_sha256 is None),
            f"{prefix}.splits.{name} must declare model_eval_tasks and "
            "model_eval_tasks_sha256 together (the model_eval tasks rendering "
            "of the same cases)",
        )
        if model_eval_tasks is not None:
            _require(
                isinstance(model_eval_tasks, str) and model_eval_tasks.strip(),
                f"{prefix}.splits.{name}.model_eval_tasks must be a non-empty string",
            )
            _require(
                isinstance(model_eval_tasks_sha256, str)
                and model_eval_tasks_sha256.strip(),
                f"{prefix}.splits.{name}.model_eval_tasks_sha256 must be a "
                "non-empty string",
            )
        return cls(
            manifest=manifest,
            content_hash=content_hash,
            srbench_dataset=srbench_dataset,
            srbench_sha256=srbench_sha256,
            model_eval_tasks=model_eval_tasks,
            model_eval_tasks_sha256=model_eval_tasks_sha256,
        )

    def to_dict(self) -> dict:
        out = {"manifest": self.manifest, "content_hash": self.content_hash}
        if self.srbench_dataset is not None:
            out["srbench_dataset"] = self.srbench_dataset
            out["srbench_sha256"] = self.srbench_sha256
        if self.model_eval_tasks is not None:
            out["model_eval_tasks"] = self.model_eval_tasks
            out["model_eval_tasks_sha256"] = self.model_eval_tasks_sha256
        return out


@dataclass(frozen=True)
class Workload:
    """One evaluation workload (M5).

    The contract declares `workloads:` as a list; the legacy single
    `workload:` form loads as a one-item list whose entry keeps the
    historical defaults (name "default", kind "mcq", driver "srbench",
    role "tune", weight 1.0). `kind` selects the request shape (mcq /
    open / agentic), `driver` the evaluation harness (srbench or
    model_eval, C7), and `role` the quarantine: `tune` workloads feed
    calibration and replay, `transfer` workloads are final-verify only
    (the frozen_test rule applies to the whole workload). `weight` is the
    macro-average weight over tune workloads and `guard_pp` the auto
    "not worse" guard: the workload's paired quality-delta CI lower bound
    must not fall below -guard_pp percentage points.
    """

    profile: str
    split_rule: str | None
    splits: dict[str, SplitSpec]
    legacy: bool
    name: str = "default"
    kind: str = "mcq"
    weight: float = 1.0
    driver: str = "srbench"
    role: str = "tune"
    guard_pp: float = DEFAULT_GUARD_PP
    # AG1: agentic workloads declare the sr-bench benchmark adapter and the
    # per-run task repetitions used to enlarge the sample when a task tier
    # is too small; `profile` doubles as the sr-bench task profile.
    benchmark: str | None = None
    repetitions: int = 1
    # AG7: declared overrides for the stability hard gates (defaults apply
    # to every agentic workload; see DEFAULT_AGENTIC_GATES). None keeps the
    # defaults and round-trips nothing.
    agentic_gates: dict[str, float | str] | None = None

    def resolved_agentic_gates(self) -> dict[str, float | str]:
        """AG7: the workload's stability ceilings - defaults under the
        declared overrides. Only meaningful for kind: agentic workloads."""
        return {**DEFAULT_AGENTIC_GATES, **(self.agentic_gates or {})}

    @classmethod
    def from_dict(
        cls, raw: dict, prefix: str = "workload", entry: bool = False
    ) -> Workload:
        profile = raw.get("profile", "quick")
        _require(
            profile in PROFILES, f"{prefix}.profile must be one of {sorted(PROFILES)}"
        )
        # M5 identity fields. Entries must declare a name; the legacy
        # single-workload form keeps its anonymous historical default.
        name_raw = raw.get("name")
        if name_raw is None:
            _require(
                not entry,
                f"{prefix}.name is required (each workloads entry needs a "
                "unique non-empty name)",
            )
            name = "default"
        else:
            _require(
                isinstance(name_raw, str) and name_raw.strip(),
                f"{prefix}.name must be a non-empty string",
            )
            name = name_raw
        kind = raw.get("kind", "mcq")
        _require(
            kind in WORKLOAD_KINDS,
            f"{prefix}.kind must be one of {sorted(WORKLOAD_KINDS)}",
        )
        weight = _require_number(raw.get("weight", 1.0), f"{prefix}.weight")
        _require(weight > 0, f"{prefix}.weight must be > 0")
        driver = raw.get("driver", "srbench")
        _require(
            driver in WORKLOAD_DRIVERS,
            f"{prefix}.driver must be one of {sorted(WORKLOAD_DRIVERS)}",
        )
        benchmark = raw.get("benchmark")
        repetitions = raw.get("repetitions", 1)
        _require(
            isinstance(repetitions, int)
            and not isinstance(repetitions, bool)
            and repetitions >= 1,
            f"{prefix}.repetitions must be an integer >= 1",
        )
        if kind == "agentic":
            _require(
                driver != "model_eval",
                f"{prefix}: the model_eval MCQ driver cannot measure agentic stability gates",
            )
            _require(
                benchmark in AGENTIC_BENCHMARKS,
                f"{prefix}.benchmark must be one of {sorted(AGENTIC_BENCHMARKS)} "
                "for kind: agentic workloads",
            )
        else:
            _require(
                benchmark is None,
                f"{prefix}.benchmark is only valid for kind: agentic workloads",
            )
        role = raw.get("role", "tune")
        _require(
            role in WORKLOAD_ROLES,
            f"{prefix}.role must be one of {sorted(WORKLOAD_ROLES)}",
        )
        agentic_gates = raw.get("agentic_gates")
        if agentic_gates is not None:
            _require(
                kind == "agentic",
                f"{prefix}.agentic_gates is only valid for kind: agentic workloads",
            )
            _require(
                isinstance(agentic_gates, dict) and bool(agentic_gates),
                f"{prefix}.agentic_gates must be a non-empty mapping",
            )
            unknown = sorted(set(agentic_gates) - set(AGENTIC_GATE_NAMES))
            _require(
                not unknown,
                f"{prefix}.agentic_gates has unknown gates: {', '.join(unknown)}",
            )
            for gate_name, gate_value in agentic_gates.items():
                if gate_value == "baseline":
                    _require(
                        gate_name == "max_tool_loop_interrupted",
                        f'{prefix}.agentic_gates.{gate_name} may pin "baseline" '
                        "only when it is max_tool_loop_interrupted",
                    )
                else:
                    gate_number = _require_number(
                        gate_value, f"{prefix}.agentic_gates.{gate_name}"
                    )
                    _require(
                        gate_number >= 0,
                        f"{prefix}.agentic_gates.{gate_name} must be >= 0",
                    )
        guard_pp = _require_number(
            raw.get("guard_pp", DEFAULT_GUARD_PP), f"{prefix}.guard_pp"
        )
        _require(guard_pp >= 0, f"{prefix}.guard_pp must be >= 0")
        if entry:
            _require(
                "splits" in raw,
                f"{prefix} must declare splits (a mapping of split name -> "
                "manifest + content_hash)",
            )
        if "splits" in raw:
            split_rule = raw.get("split_rule")
            if not entry:
                _require(
                    isinstance(split_rule, str) and split_rule.strip(),
                    f"{prefix}.split_rule must be a non-empty string when {prefix}.splits is used "
                    "(free text like 'group by source_id; seed 42; 40/30/30'; it is written to provenance)",
                )
            splits_raw = raw.get("splits")
            _require(
                isinstance(splits_raw, dict) and bool(splits_raw),
                f"{prefix}.splits must be a non-empty mapping",
            )
            if not entry:
                _require(
                    set(splits_raw) == set(SPLIT_NAMES),
                    f"{prefix}.splits must declare exactly {sorted(SPLIT_NAMES)}",
                )
            splits = {
                split_name: SplitSpec.from_dict(
                    _require_mapping(v, f"{prefix}.splits.{split_name}"),
                    split_name,
                    prefix=prefix,
                )
                for split_name, v in splits_raw.items()
            }
            return cls(
                profile=profile,
                split_rule=split_rule,
                splits=splits,
                legacy=False,
                name=name,
                kind=kind,
                weight=weight,
                driver=driver,
                role=role,
                guard_pp=guard_pp,
                benchmark=benchmark,
                repetitions=repetitions,
                agentic_gates=agentic_gates,
            )
        # legacy single-dataset form: treated as a dev-only workload
        _require(
            not entry,
            f"{prefix} must declare splits (a mapping of split name -> "
            "manifest + content_hash)",
        )
        manifest = raw.get("manifest")
        _require(
            isinstance(manifest, str) and manifest.strip(),
            f"{prefix}.manifest must be a non-empty string",
        )
        content_hash = raw.get("content_hash")
        _require(
            isinstance(content_hash, str) and content_hash.strip(),
            f"{prefix}.content_hash is required (frozen dataset hash; cross-round comparability depends on it)",
        )
        srbench_dataset = raw.get("srbench_dataset")
        srbench_sha256 = raw.get("srbench_sha256")
        _require(
            (srbench_dataset is None) == (srbench_sha256 is None),
            f"{prefix} must declare srbench_dataset and srbench_sha256 "
            "together (the sr-bench rendering of the same cases)",
        )
        if srbench_dataset is not None:
            _require(
                isinstance(srbench_dataset, str) and srbench_dataset.strip(),
                f"{prefix}.srbench_dataset must be a non-empty string",
            )
            _require(
                isinstance(srbench_sha256, str) and srbench_sha256.strip(),
                f"{prefix}.srbench_sha256 must be a non-empty string",
            )
        model_eval_tasks = raw.get("model_eval_tasks")
        model_eval_tasks_sha256 = raw.get("model_eval_tasks_sha256")
        _require(
            (model_eval_tasks is None) == (model_eval_tasks_sha256 is None),
            f"{prefix} must declare model_eval_tasks and model_eval_tasks_sha256 "
            "together (the model_eval tasks rendering of the same cases)",
        )
        if model_eval_tasks is not None:
            _require(
                isinstance(model_eval_tasks, str) and model_eval_tasks.strip(),
                f"{prefix}.model_eval_tasks must be a non-empty string",
            )
            _require(
                isinstance(model_eval_tasks_sha256, str)
                and model_eval_tasks_sha256.strip(),
                f"{prefix}.model_eval_tasks_sha256 must be a non-empty string",
            )
        return cls(
            profile=profile,
            split_rule=None,
            splits={
                "dev": SplitSpec(
                    manifest=manifest,
                    content_hash=content_hash,
                    srbench_dataset=srbench_dataset,
                    srbench_sha256=srbench_sha256,
                    model_eval_tasks=model_eval_tasks,
                    model_eval_tasks_sha256=model_eval_tasks_sha256,
                )
            },
            legacy=True,
            name=name,
            kind=kind,
            weight=weight,
            driver=driver,
            role=role,
            guard_pp=guard_pp,
            benchmark=benchmark,
            repetitions=repetitions,
            agentic_gates=agentic_gates,
        )

    def check_live_split(self, split: str, *, final: bool) -> None:
        if self.role == "transfer" and not final:
            raise ContractError(
                f"workload {self.name!r} has role 'transfer'; pass --final"
            )
        if split == "frozen_test" and not final:
            raise ContractError(
                "frozen_test is quarantined; pass --final for the promotion verdict"
            )
        if (
            final
            and not self.legacy
            and self.role != "transfer"
            and split != "frozen_test"
        ):
            raise ContractError(
                "--final requires frozen_test or a quarantined transfer workload"
            )

    def split_hash(self, name: str) -> str:
        if name not in self.splits:
            raise ContractError(
                f"unknown workload split {name!r}; declared: {sorted(self.splits)}"
            )
        return self.splits[name].content_hash

    def split_manifest(self, name: str) -> str:
        if name not in self.splits:
            raise ContractError(
                f"unknown workload split {name!r}; declared: {sorted(self.splits)}"
            )
        return self.splits[name].manifest

    def split_srbench_dataset(self, name: str) -> str | None:
        """The split's sr-bench dataset path, or None when not declared (X3)."""
        if name not in self.splits:
            raise ContractError(
                f"unknown workload split {name!r}; declared: {sorted(self.splits)}"
            )
        return self.splits[name].srbench_dataset

    def split_srbench_hash(self, name: str) -> str | None:
        """The split's sr-bench dataset hash, or None when not declared (X3)."""
        if name not in self.splits:
            raise ContractError(
                f"unknown workload split {name!r}; declared: {sorted(self.splits)}"
            )
        return self.splits[name].srbench_sha256

    def split_model_eval_tasks(self, name: str) -> str | None:
        """The split's model_eval tasks rendering path, or None (Y3)."""
        if name not in self.splits:
            raise ContractError(
                f"unknown workload split {name!r}; declared: {sorted(self.splits)}"
            )
        return self.splits[name].model_eval_tasks

    def split_model_eval_tasks_hash(self, name: str) -> str | None:
        """The split's model_eval tasks rendering hash, or None (Y3)."""
        if name not in self.splits:
            raise ContractError(
                f"unknown workload split {name!r}; declared: {sorted(self.splits)}"
            )
        return self.splits[name].model_eval_tasks_sha256

    def live_dataset_hash(self, name: str) -> str:
        """X3: the dataset identity live evidence must bind to.

        sr-bench run documents project `manifest.dataset.sha256` - the
        identity of the dataset they actually scored. When the split
        declares `srbench_dataset`, that is its declared `srbench_sha256`;
        otherwise the legacy cases `content_hash` (the historical form
        where the cases file is fed to sr-bench directly).
        """
        srbench = self.split_srbench_hash(name)
        return srbench if srbench is not None else self.split_hash(name)

    def to_dict(self) -> dict:
        # M5 fields serialize only when they differ from the defaults, so a
        # legacy spec round-trips bit-identically through plan/next/report.
        extra: dict[str, Any] = {}
        if self.name != "default":
            extra["name"] = self.name
        if self.kind != "mcq":
            extra["kind"] = self.kind
        if self.weight != 1.0:
            extra["weight"] = self.weight
        if self.driver != "srbench":
            extra["driver"] = self.driver
        if self.role != "tune":
            extra["role"] = self.role
        if self.guard_pp != DEFAULT_GUARD_PP:
            extra["guard_pp"] = self.guard_pp
        if self.benchmark is not None:
            extra["benchmark"] = self.benchmark
        if self.repetitions != 1:
            extra["repetitions"] = self.repetitions
        if self.agentic_gates is not None:
            extra["agentic_gates"] = dict(self.agentic_gates)
        if self.legacy:
            split = self.splits["dev"]
            out = {
                "manifest": split.manifest,
                "content_hash": split.content_hash,
                "profile": self.profile,
                **extra,
            }
            if split.srbench_dataset is not None:
                out["srbench_dataset"] = split.srbench_dataset
                out["srbench_sha256"] = split.srbench_sha256
            if split.model_eval_tasks is not None:
                out["model_eval_tasks"] = split.model_eval_tasks
                out["model_eval_tasks_sha256"] = split.model_eval_tasks_sha256
            return out
        return {
            "profile": self.profile,
            "split_rule": self.split_rule,
            "splits": {name: split.to_dict() for name, split in self.splits.items()},
            **extra,
        }


def _parse_workloads(raw: dict) -> tuple[Workload, ...]:
    """M5: `workloads:` list (each entry named) or the legacy single
    `workload:` form (one entry with the historical defaults). Exactly one
    of the two keys may appear; names must be unique and at least one
    workload must be tunable."""
    workloads_raw = raw.get("workloads")
    workload_raw = raw.get("workload")
    if workloads_raw is not None:
        _require(
            workload_raw is None,
            "declare either workload or workloads, not both",
        )
        _require(
            isinstance(workloads_raw, list) and bool(workloads_raw),
            "workloads must be a non-empty list",
        )
        workloads = tuple(
            Workload.from_dict(
                _require_mapping(item, f"workloads[{i}]"),
                prefix=f"workloads[{i}]",
                entry=True,
            )
            for i, item in enumerate(workloads_raw)
        )
    elif workload_raw is not None:
        workloads = (Workload.from_dict(_require_mapping(workload_raw, "workload")),)
    else:
        raise ContractError(
            "spec must declare workload (legacy single dataset) or workloads (M5 list)"
        )
    names = [w.name for w in workloads]
    _require(
        len(set(names)) == len(names), f"workload names must be unique (got {names})"
    )
    _require(
        any(w.role == "tune" for w in workloads),
        "workloads must include at least one role=tune workload "
        "(transfer workloads are never tuned on)",
    )
    return workloads


def validate_signal_fact(value: Any, where: str) -> None:
    """Validate one truth-signal fact / condition value: a non-empty label
    string, a finite number, or a numeric predicate mapping over PREDICATE_OPS
    (F2)."""
    if isinstance(value, str):
        _require(bool(value.strip()), f"{where} must be a non-empty string")
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _require(math.isfinite(value), f"{where} must be a finite number")
        return
    if isinstance(value, dict):
        _require(bool(value), f"{where} predicate must not be empty")
        for op, bound in value.items():
            _require(
                op in PREDICATE_OPS,
                f"{where} predicate op must be one of {list(PREDICATE_OPS)}, got {op!r}",
            )
            _require_number(bound, f"{where} predicate bound {op!r}")
        return
    _require(
        False,
        f"{where} must be a label string, a number, or a predicate mapping, got {value!r}",
    )


def matches_condition(fact: Any, condition: Any) -> bool:
    """Does one truth fact satisfy one declared condition value?

    Label conditions match by string equality; number conditions match
    within float tolerance; predicate conditions apply every op with the
    router's numeric-predicate semantics. Anything else does not match
    (a missing fact never satisfies a condition)."""
    if isinstance(condition, str):
        return isinstance(fact, str) and fact == condition
    if isinstance(condition, (int, float)) and not isinstance(condition, bool):
        return (
            isinstance(fact, (int, float))
            and not isinstance(fact, bool)
            and math.isfinite(fact)
            and math.isclose(float(fact), float(condition), rel_tol=0.0, abs_tol=1e-9)
        )
    if isinstance(condition, dict):
        if (
            isinstance(fact, bool)
            or not isinstance(fact, (int, float))
            or not math.isfinite(fact)
        ):
            return False
        value = float(fact)
        for op, bound in condition.items():
            b = float(bound)
            if op == "gte" and value < b:
                return False
            if op == "lte" and value > b:
                return False
            if op == "gt" and value <= b:
                return False
            if op == "lt" and value >= b:
                return False
        return True
    return False


@dataclass(frozen=True)
class Expectation:
    """What the oracle policy expects for one case: the intended decision and
    the (optional) intended model set (F2; a decision's modelRefs usually
    has several models)."""

    decision: str
    models: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any, where: str) -> Expectation:
        _require(isinstance(raw, dict), f"{where} must be a mapping")
        decision = raw.get("decision")
        _require(
            isinstance(decision, str) and decision.strip(),
            f"{where}.decision must be a non-empty string",
        )
        models_raw = raw.get("models", [])
        _require(
            isinstance(models_raw, list),
            f"{where}.models must be a list of model names",
        )
        for model in models_raw:
            _require(
                isinstance(model, str) and model.strip(),
                f"{where}.models entries must be non-empty strings",
            )
        return cls(decision=decision, models=tuple(models_raw))

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"decision": self.decision}
        if self.models:
            out["models"] = list(self.models)
        return out


@dataclass(frozen=True)
class OracleRule:
    """One oracle-policy rule: signal-fact conditions plus the expected
    route. Rules are evaluated top-down; the first whose `when` matches the
    case's truth facts wins (F2)."""

    when: dict[str, Any]
    expect: Expectation


@dataclass(frozen=True)
class OraclePolicy:
    """Truth -> intended route, declared explicitly and never recomputed
    from the router config: attribution depends on this policy, not on
    re-implementing router logic (design §6).

    Two declared forms (F2; mixing them is rejected at load time):

    - label mapping (legacy): {"<decision label>": "<model>"} — a string
      truth looks up the intended model directly.
    - rules: {"rules": [{"when": {<signal>: <condition>}, "expect":
      {decision, models?}}, ...], "default": {decision, models?}} —
      evaluated top-down against the case's truth signal facts; the first
      matching rule wins, the optional default covers the rest.

    `when` conditions and truth-signal facts share one value grammar
    (validate_signal_fact): label string, finite number, or numeric
    predicate mapping (PREDICATE_OPS)."""

    mapping: dict[str, str] = field(default_factory=dict)
    rules: tuple[OracleRule, ...] = ()
    default: Expectation | None = None
    # AG3: the truth namespace the mapping is written in (decision labels are
    # intended decisions; behavior labels constrain the model route only).
    truth_space: str = DEFAULT_TRUTH_SPACE

    @classmethod
    def from_dict(
        cls, raw: Any, truth_space: str = DEFAULT_TRUTH_SPACE
    ) -> OraclePolicy:
        _require(
            isinstance(raw, dict) and bool(raw),
            "attribution.oracle_policy must be a non-empty mapping",
        )
        if "rules" in raw or "default" in raw:
            unknown = [k for k in raw if k not in ("rules", "default")]
            _require(
                not unknown,
                "oracle policy must be either a label -> model mapping or a rules/default mapping, not both "
                f"(unexpected keys: {unknown})",
            )
            rules_raw = raw.get("rules", [])
            _require(isinstance(rules_raw, list), "oracle policy rules must be a list")
            _require(
                bool(rules_raw) or raw.get("default") is not None,
                "oracle policy rules must not be empty",
            )
            rules = tuple(cls._rule_from_dict(r, i) for i, r in enumerate(rules_raw))
            default_raw = raw.get("default")
            default = (
                Expectation.from_dict(default_raw, "oracle policy default")
                if default_raw is not None
                else None
            )
            return cls(rules=rules, default=default, truth_space=truth_space)
        for label, model in raw.items():
            _require(
                isinstance(label, str) and label.strip(),
                f"oracle policy label must be a non-empty string: {label!r}",
            )
            _require(
                isinstance(model, str) and model.strip(),
                f"oracle policy model for label {label!r} must be a non-empty string",
            )
        return cls(mapping=dict(raw), truth_space=truth_space)

    @staticmethod
    def _rule_from_dict(raw: Any, index: int) -> OracleRule:
        where = f"oracle policy rules[{index}]"
        _require(isinstance(raw, dict), f"{where} must be a mapping")
        unexpected = [k for k in raw if k not in ("when", "expect")]
        _require(not unexpected, f"{where} has unexpected keys: {unexpected}")
        when = raw.get("when")
        _require(
            isinstance(when, dict) and bool(when),
            f"{where}.when must be a non-empty mapping",
        )
        for signal, condition in when.items():
            _require(
                isinstance(signal, str) and signal.strip(),
                f"{where}.when signal names must be non-empty strings",
            )
            validate_signal_fact(condition, f"{where}.when[{signal!r}]")
        expect = Expectation.from_dict(raw.get("expect"), f"{where}.expect")
        return OracleRule(when=dict(when), expect=expect)

    def expected(self, label: str) -> str | None:
        """Legacy label-mapping lookup; rules-form policies have no label
        mapping and resolve only signal-fact truths (evaluate)."""
        return self.mapping.get(label)

    def evaluate(self, facts: dict[str, Any]) -> Expectation | None:
        """Resolve the expected route for signal facts: first matching rule,
        else the default, else None (uncovered truth -> policy_gap)."""
        for rule in self.rules:
            if all(
                matches_condition(facts.get(signal), condition)
                for signal, condition in rule.when.items()
            ):
                return rule.expect
        return self.default

    def covers(self, labels: list[str]) -> list[str]:
        """Labels the policy cannot resolve. Rules-form policies resolve
        signal-fact truths, so every bare string label counts as uncovered."""
        if self.rules or self.default is not None:
            return list(labels)
        return [label for label in labels if label not in self.mapping]

    def to_dict(self) -> dict:
        if self.rules or self.default is not None:
            out: dict[str, Any] = {
                "rules": [
                    {"when": dict(rule.when), "expect": rule.expect.to_dict()}
                    for rule in self.rules
                ]
            }
            if self.default is not None:
                out["default"] = self.default.to_dict()
            return out
        return dict(self.mapping)


@dataclass(frozen=True)
class Gate:
    name: str
    metric: str
    floor: str | float  # "baseline" or a numeric floor

    @classmethod
    def from_dict(cls, raw: dict, index: int) -> Gate:
        name = raw.get("name")
        _require(
            isinstance(name, str) and name.strip(),
            f"gates[{index}].name must be a non-empty string",
        )
        metric = raw.get("metric")
        _require(
            isinstance(metric, str) and all(part.strip() for part in metric.split(".")),
            f"gates[{index}].metric must be a non-empty dotted JSON key",
        )
        floor = raw.get("floor")
        if floor != "baseline":
            floor = _require_number(floor, f"gates[{index}].floor")
        return cls(name=name, metric=metric, floor=floor)

    def to_dict(self) -> dict:
        return {"name": self.name, "metric": self.metric, "floor": self.floor}


@dataclass(frozen=True)
class Budget:
    rounds: int
    max_minutes_per_trial: float

    @classmethod
    def from_dict(cls, raw: dict) -> Budget:
        rounds = raw.get("rounds", 1)
        _require(
            isinstance(rounds, int) and not isinstance(rounds, bool) and rounds >= 1,
            "budget.rounds must be an integer >= 1",
        )
        minutes = _require_number(
            raw.get("max_minutes_per_trial", 120.0), "budget.max_minutes_per_trial"
        )
        _require(minutes > 0, "budget.max_minutes_per_trial must be > 0")
        return cls(rounds=rounds, max_minutes_per_trial=minutes)

    def to_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "max_minutes_per_trial": self.max_minutes_per_trial,
        }


def _is_cost_metric(metric: str) -> bool:
    """M4: a cost/spend metric (benefit constraints price against a basis)."""
    lowered = metric.lower()
    return "cost" in lowered or "spend" in lowered


def _check_cost_model(raw: Any) -> dict[str, float] | None:
    """E2: optional top-level `cost_model` - model name -> relative cost.

    The replay tier's cost proxy (routed_cost_index) averages these relative
    numbers over answered samples, so every value must be a finite positive
    number (a zero or negative cost would corrupt the index, and an empty
    map cannot proxy anything)."""
    if raw is None:
        return None
    _require(
        isinstance(raw, dict),
        "cost_model must be a mapping of model name -> relative cost",
    )
    _require(
        bool(raw),
        "cost_model must not be empty (omit the key when there is no cost model)",
    )
    out: dict[str, float] = {}
    for model, value in raw.items():
        _require(
            isinstance(model, str) and model.strip(),
            f"cost_model keys must be non-empty model names (got {model!r})",
        )
        cost = _require_number(value, f"cost_model[{model!r}]")
        _require(cost > 0, f"cost_model[{model!r}] must be > 0 (relative cost)")
        out[model] = cost
    return out


def _derived_cost_model(rates: dict[str, dict[str, float]]) -> dict[str, float]:
    """C6: derive the replay cost model from provider pricing rates.

    One model's unit rate is the sum of its prompt and completion rates
    (per-1M tokens); the relative cost is that sum divided by the cheapest
    positive sum, so the cheapest priced model sits at 1.0 and a free model
    stays at 0. Deterministic and token-statistics-free: it assumes direct
    model evals spend a comparable token mix per question, which is exactly
    the mcq workload shape the cold-start lane runs. All-zero rates cannot
    be normalized and are a contract error.
    """
    totals = {
        model: float(rates_["prompt_per_1m"]) + float(rates_["completion_per_1m"])
        for model, rates_ in rates.items()
    }
    positives = [value for value in totals.values() if value > 0]
    if not positives:
        raise ContractError(
            "cost_basis.assumed_pricing rates are all zero: cannot derive a "
            "cost model (declare cost_model directly instead)"
        )
    floor = min(positives)
    return {model: total / floor for model, total in totals.items()}


@dataclass(frozen=True)
class Stack:
    mode: str
    endpoint: str
    baseline_ref: str

    @classmethod
    def from_dict(cls, raw: dict) -> Stack:
        mode = raw.get("mode", "external")
        _require(
            mode in STACK_MODES, f"stack.mode must be one of {sorted(STACK_MODES)}"
        )
        endpoint = raw.get("endpoint")
        _require(
            isinstance(endpoint, str) and endpoint.strip(),
            "stack.endpoint must be a non-empty string",
        )
        baseline_ref = raw.get("baseline_ref")
        _require(
            isinstance(baseline_ref, str) and baseline_ref.strip(),
            "stack.baseline_ref must be a non-empty string",
        )
        return cls(mode=mode, endpoint=endpoint, baseline_ref=baseline_ref)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "endpoint": self.endpoint,
            "baseline_ref": self.baseline_ref,
        }


@dataclass(frozen=True)
class EvidenceQuality:
    """M3: output-quality ceilings for live evidence.

    Truncated outputs and missing answers make an arm's score unreliable
    evidence rather than a quality loss: a rate beyond its ceiling turns
    the live verdict INCONCLUSIVE. sr-bench reports the per-target counts
    (target summary `output_diagnostics`), and the verdict reads them
    from the compare payload's `baseline_targets`/`candidate_targets`.
    """

    max_truncation_pct: float = DEFAULT_MAX_EVIDENCE_PCT
    max_no_answer_pct: float = DEFAULT_MAX_EVIDENCE_PCT

    @classmethod
    def from_dict(cls, raw: dict) -> EvidenceQuality:
        unknown = sorted(set(raw) - {"max_truncation_pct", "max_no_answer_pct"})
        _require(
            not unknown,
            f"evidence_quality has unknown keys: {', '.join(unknown)}",
        )
        truncation = _require_number(
            raw.get("max_truncation_pct", DEFAULT_MAX_EVIDENCE_PCT),
            "evidence_quality.max_truncation_pct",
        )
        _require(truncation >= 0, "evidence_quality.max_truncation_pct must be >= 0")
        no_answer = _require_number(
            raw.get("max_no_answer_pct", DEFAULT_MAX_EVIDENCE_PCT),
            "evidence_quality.max_no_answer_pct",
        )
        _require(no_answer >= 0, "evidence_quality.max_no_answer_pct must be >= 0")
        return cls(max_truncation_pct=truncation, max_no_answer_pct=no_answer)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        if self.max_truncation_pct != DEFAULT_MAX_EVIDENCE_PCT:
            out["max_truncation_pct"] = self.max_truncation_pct
        if self.max_no_answer_pct != DEFAULT_MAX_EVIDENCE_PCT:
            out["max_no_answer_pct"] = self.max_no_answer_pct
        return out


@dataclass(frozen=True)
class References:
    """M4: the two single-model reference arms (模型别名 = sr-bench target id).

    `strong` is the quality reference (the best candidate model),
    `weak` the cheapest one; constraint bounds may pin either via
    `bound: ref:strong` / `ref:weak`. Both must be registered single-model
    target ids present in the compared runs. C8: `ref:random_mix` is also
    a valid bound but resolves against computed driver evidence (the
    cost-matched random mix), not against a declared arm here.
    """

    strong: str
    weak: str

    @classmethod
    def from_dict(cls, raw: dict) -> References:
        unknown = sorted(set(raw) - {"strong", "weak"})
        _require(not unknown, f"references has unknown keys: {', '.join(unknown)}")
        strong = raw.get("strong")
        _require(
            isinstance(strong, str) and strong.strip(),
            "references.strong must be a non-empty string (the strong single-model target id)",
        )
        weak = raw.get("weak")
        _require(
            isinstance(weak, str) and weak.strip(),
            "references.weak must be a non-empty string (the weak single-model target id)",
        )
        return cls(strong=strong, weak=weak)

    def to_dict(self) -> dict:
        return {"strong": self.strong, "weak": self.weak}


@dataclass(frozen=True)
class CostBasis:
    """M4: where the verdict's cost numbers come from.

    `source` is "provider_pricing" (billed/token rates from the provider)
    or "cost_model" (the replay cost proxy). `assumed_pricing` carries
    per-model rates ONLY for models the stack cannot price; any cost
    estimated from it is annotated "assumed" in the evidence and report,
    never presented as measured spend.
    """

    source: str
    assumed_pricing: dict[str, dict[str, float]] | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> CostBasis:
        unknown = sorted(set(raw) - {"source", "assumed_pricing"})
        _require(not unknown, f"cost_basis has unknown keys: {', '.join(unknown)}")
        source = raw.get("source")
        _require(
            source in COST_BASIS_SOURCES,
            f"cost_basis.source must be one of {sorted(COST_BASIS_SOURCES)} (got {source!r})",
        )
        assumed_raw = raw.get("assumed_pricing")
        assumed: dict[str, dict[str, float]] | None = None
        if assumed_raw is not None:
            _require(
                isinstance(assumed_raw, dict) and bool(assumed_raw),
                "cost_basis.assumed_pricing must be a non-empty mapping of model -> rates",
            )
            assumed = {}
            for model, rates in assumed_raw.items():
                _require(
                    isinstance(rates, dict),
                    f"cost_basis.assumed_pricing.{model} must be a mapping",
                )
                unknown_rate = sorted(
                    set(rates) - {"prompt_per_1m", "completion_per_1m"}
                )
                _require(
                    not unknown_rate,
                    f"cost_basis.assumed_pricing.{model} has unknown keys: "
                    + ", ".join(unknown_rate),
                )
                prompt = _require_number(
                    rates.get("prompt_per_1m"),
                    f"cost_basis.assumed_pricing.{model}.prompt_per_1m",
                )
                completion = _require_number(
                    rates.get("completion_per_1m"),
                    f"cost_basis.assumed_pricing.{model}.completion_per_1m",
                )
                _require(
                    prompt >= 0 and completion >= 0,
                    f"cost_basis.assumed_pricing.{model} rates must be >= 0",
                )
                assumed[model] = {
                    "prompt_per_1m": prompt,
                    "completion_per_1m": completion,
                }
        return cls(source=source, assumed_pricing=assumed)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"source": self.source}
        if self.assumed_pricing is not None:
            out["assumed_pricing"] = {
                model: dict(rates) for model, rates in self.assumed_pricing.items()
            }
        return out


# Cold start (C3): profile/grouping setting. Labels with fewer observed
# cases than this merge into `other` before the profile is computed.
DEFAULT_MIN_LABEL_CASES = 20
# Cold start (C3 lane mode): a lane observed on fewer cases than this is
# reported as "not measured" (sub-issue-8 §2.1 bootstrap.min_lane_cases);
# its binding follows the default rule and is marked untested.
DEFAULT_MIN_LANE_CASES = 20
# Cold start (bindings): below this advantage (percentage points) between a
# lane's best model and the cheapest acceptable one, the routing-space check
# reports "no benefit to routing" (sub-issue-8 §2.2 step 5).
DEFAULT_MIN_ADVANTAGE_PP = 5.0
# Cold start (bindings): a lane whose requirement is a context window needs
# at least this many tokens (MoM V1 long_context lane: 200K, recipe.dsl L127).
LONG_CONTEXT_MIN_TOKENS = 200_000
# Cold start (C3): generation params shared by every direct-model eval and
# the later live arms; keys mirror multi_benchmark_eval.py's load_params.
GENERATION_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "chat_template_kwargs",
    "extra_body",
)


@dataclass(frozen=True)
class ModelCapabilities:
    """Cold start (bindings): one model's user-declared capabilities.

    recipe-lab never guesses capabilities (sub-issue-8 §2.1): lane
    requirements filter against exactly what the user declared here.
    """

    vision: bool = False
    tools: bool = False
    reasoning: bool = False
    context_window: int | None = None

    def is_default(self) -> bool:
        return self == ModelCapabilities()

    @classmethod
    def from_dict(cls, raw: dict) -> ModelCapabilities:
        unknown = sorted(set(raw) - {"vision", "tools", "reasoning", "context_window"})
        _require(not unknown, f"capabilities has unknown keys: {', '.join(unknown)}")
        for key in ("vision", "tools", "reasoning"):
            if key in raw:
                _require(
                    isinstance(raw[key], bool),
                    f"capabilities.{key} must be a boolean",
                )
        window = raw.get("context_window")
        if window is not None:
            _require(
                isinstance(window, int)
                and not isinstance(window, bool)
                and window >= 1,
                "capabilities.context_window must be an integer >= 1",
            )
        return cls(
            vision=bool(raw.get("vision", False)),
            tools=bool(raw.get("tools", False)),
            reasoning=bool(raw.get("reasoning", False)),
            context_window=window,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        if self.vision:
            out["vision"] = True
        if self.tools:
            out["tools"] = True
        if self.reasoning:
            out["reasoning"] = True
        if self.context_window is not None:
            out["context_window"] = self.context_window
        return out


@dataclass(frozen=True)
class TemplateConfig:
    """Cold start (C2/C4): the built-in recipe template the config is
    generated from (sub-issue-8 §2.1 `template`).

    `bundle` is a built-in model-bundle name (e.g. `mom-v1`) and `recipe`
    the recipe inside it (e.g. `balance`); both are validated against
    `vllm-sr recipe builtin list` at runtime, and the bundle's sha256 is
    written into the provenance.
    """

    bundle: str
    recipe: str

    @classmethod
    def from_dict(cls, raw: dict) -> TemplateConfig:
        unknown = sorted(set(raw) - {"bundle", "recipe"})
        _require(not unknown, f"template has unknown keys: {', '.join(unknown)}")
        bundle = raw.get("bundle")
        recipe = raw.get("recipe")
        _require(
            isinstance(bundle, str) and bundle.strip(),
            "template.bundle must be a non-empty string",
        )
        _require(
            isinstance(recipe, str) and recipe.strip(),
            "template.recipe must be a non-empty string",
        )
        return cls(bundle=bundle, recipe=recipe)

    def to_dict(self) -> dict:
        return {"bundle": self.bundle, "recipe": self.recipe}


@dataclass(frozen=True)
class BootstrapConfig:
    """Cold start (bindings, sub-issue-8 §2.1 `bootstrap`): generation-side
    settings shared by every lambda."""

    min_lane_cases: int = DEFAULT_MIN_LANE_CASES
    min_advantage_pp: float = DEFAULT_MIN_ADVANTAGE_PP
    lambdas: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> BootstrapConfig:
        unknown = sorted(set(raw) - {"min_lane_cases", "min_advantage_pp", "lambdas"})
        _require(not unknown, f"bootstrap has unknown keys: {', '.join(unknown)}")
        min_cases = raw.get("min_lane_cases", DEFAULT_MIN_LANE_CASES)
        _require(
            isinstance(min_cases, int)
            and not isinstance(min_cases, bool)
            and min_cases >= 1,
            "bootstrap.min_lane_cases must be an integer >= 1",
        )
        advantage = raw.get("min_advantage_pp", DEFAULT_MIN_ADVANTAGE_PP)
        _require(
            isinstance(advantage, (int, float))
            and not isinstance(advantage, bool)
            and advantage > 0,
            "bootstrap.min_advantage_pp must be a number > 0",
        )
        lambdas_raw = raw.get("lambdas")
        lambdas: tuple[float, ...] | None = None
        if lambdas_raw is not None:
            _require(
                isinstance(lambdas_raw, list) and bool(lambdas_raw),
                "bootstrap.lambdas must be a non-empty list of numbers",
            )
            values: list[float] = []
            seen: set[str] = set()
            for i, entry in enumerate(lambdas_raw):
                _require(
                    isinstance(entry, (int, float))
                    and not isinstance(entry, bool)
                    and entry >= 0,
                    f"bootstrap.lambdas[{i}] must be a number >= 0",
                )
                tag = lambda_tag(float(entry))
                if tag not in seen:
                    seen.add(tag)
                    values.append(float(entry))
            lambdas = tuple(values)
        return cls(
            min_lane_cases=min_cases,
            min_advantage_pp=float(advantage),
            lambdas=lambdas,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        if self.min_lane_cases != DEFAULT_MIN_LANE_CASES:
            out["min_lane_cases"] = self.min_lane_cases
        if self.min_advantage_pp != DEFAULT_MIN_ADVANTAGE_PP:
            out["min_advantage_pp"] = self.min_advantage_pp
        if self.lambdas is not None:
            out["lambdas"] = list(self.lambdas)
        return out


def lambda_tag(lam: float) -> str:
    """Deterministic filename-safe tag for a lambda value.

    Lives at module level so BootstrapConfig and the bindings CLI share one
    tagging convention.
    """
    lam = float(lam)
    if lam == int(lam):
        return str(int(lam))
    return repr(lam)


@dataclass(frozen=True)
class ModelEntry:
    """Cold start (C3/C5): one candidate model and its direct endpoint.

    `name` is the alias used in result dirnames, token-cost files and the
    generated decisions. It must round-trip through result_to_config's
    directory convention (§3.2-1: the first `_` becomes `/`, plus the
    `_direct`/`_cot` suffix stripping), so it either contains exactly one
    `/` or no `_` at all - and it must not be `auto` (collect_model_accuracies
    skips that name).
    """

    name: str
    endpoint: str
    served_name: str
    api_key_env: str | None = None
    # Cold start (bindings): user-declared capabilities; lane requirements
    # filter against exactly these (sub-issue-8 §2.1 - never guessed).
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    @classmethod
    def from_dict(cls, raw: dict) -> ModelEntry:
        unknown = sorted(
            set(raw)
            - {"name", "endpoint", "served_name", "api_key_env", "capabilities"}
        )
        _require(not unknown, f"model entry has unknown keys: {', '.join(unknown)}")
        name = raw.get("name")
        _require(
            isinstance(name, str) and name.strip(),
            "model name must be a non-empty string",
        )
        _require(
            name != "auto",
            "model name must not be 'auto' (collect_model_accuracies skips it)",
        )
        _require(
            "/" not in name or name.count("/") == 1,
            f"model name {name!r} must contain at most one '/' "
            "(the dirname convention maps it back)",
        )
        _require(
            "/" in name or "_" not in name,
            f"model name {name!r} cannot round-trip through the result_to_config "
            "dirname convention: it must contain '/' or no '_' at all",
        )
        endpoint = raw.get("endpoint")
        _require(
            isinstance(endpoint, str) and endpoint.startswith(("http://", "https://")),
            f"model {name!r} endpoint must be an http(s) URL (got {endpoint!r})",
        )
        served = raw.get("served_name")
        if served is None:
            served = name
        _require(
            isinstance(served, str) and served.strip(),
            f"model {name!r} served_name must be a non-empty string",
        )
        api_key_env = raw.get("api_key_env")
        if api_key_env is not None:
            _require(
                isinstance(api_key_env, str) and api_key_env.strip(),
                f"model {name!r} api_key_env must be a non-empty string",
            )
        capabilities_raw = raw.get("capabilities")
        capabilities = (
            ModelCapabilities.from_dict(
                _require_mapping(capabilities_raw, f"model {name!r} capabilities")
            )
            if capabilities_raw is not None
            else ModelCapabilities()
        )
        return cls(
            name=name,
            endpoint=endpoint,
            served_name=served,
            api_key_env=api_key_env,
            capabilities=capabilities,
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "name": self.name,
            "endpoint": self.endpoint,
            "served_name": self.served_name,
        }
        if self.api_key_env is not None:
            out["api_key_env"] = self.api_key_env
        if not self.capabilities.is_default():
            out["capabilities"] = self.capabilities.to_dict()
        return out


def _check_generation(raw: dict) -> dict[str, Any]:
    """Validate the shared generation params (multi_benchmark_eval shape)."""
    unknown = sorted(set(raw) - set(GENERATION_KEYS))
    _require(not unknown, f"generation has unknown keys: {', '.join(unknown)}")
    if "temperature" in raw:
        temp = _require_number(raw["temperature"], "generation.temperature")
        _require(temp >= 0, "generation.temperature must be >= 0")
    if "max_tokens" in raw:
        tokens = raw["max_tokens"]
        _require(
            isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 1,
            "generation.max_tokens must be an integer >= 1",
        )
    if "top_p" in raw and raw["top_p"] is not None:
        top_p = _require_number(raw["top_p"], "generation.top_p")
        _require(0 < top_p <= 1, "generation.top_p must be in (0, 1]")
    for key in ("chat_template_kwargs", "extra_body"):
        if key in raw and raw[key] is not None:
            _require_mapping(raw[key], f"generation.{key}")
    return dict(raw)


@dataclass(frozen=True)
class ProfileConfig:
    """Cold start (C3): router-label profile settings."""

    min_label_cases: int = DEFAULT_MIN_LABEL_CASES

    @classmethod
    def from_dict(cls, raw: dict) -> ProfileConfig:
        unknown = sorted(set(raw) - {"min_label_cases"})
        _require(not unknown, f"profile has unknown keys: {', '.join(unknown)}")
        min_cases = raw.get("min_label_cases", DEFAULT_MIN_LABEL_CASES)
        _require(
            isinstance(min_cases, int)
            and not isinstance(min_cases, bool)
            and min_cases >= 1,
            "profile.min_label_cases must be an integer >= 1",
        )
        return cls(min_label_cases=min_cases)

    def to_dict(self) -> dict:
        if self.min_label_cases == DEFAULT_MIN_LABEL_CASES:
            return {}
        return {"min_label_cases": self.min_label_cases}


@dataclass(frozen=True)
class ExperimentSpec:
    api_version: str
    objective: Objective
    # M5: one or more evaluation workloads; the legacy single `workload:`
    # form loads as a one-item list. `workload` (below) resolves to the
    # first entry for the historical single-workload API.
    workloads: tuple[Workload, ...]
    oracle_policy: OraclePolicy
    gates: tuple[Gate, ...] = field(default=())
    budget: Budget = field(
        default_factory=lambda: Budget(rounds=1, max_minutes_per_trial=120.0)
    )
    stack: Stack | None = None
    # E2: optional model -> relative-cost map for the replay cost proxy.
    cost_model: dict[str, float] | None = None
    # M3: output-quality ceilings; rates beyond them are unreliable evidence.
    evidence_quality: EvidenceQuality = field(default_factory=EvidenceQuality)
    # M4: the two single-model reference arms (target ids) constraint bounds
    # may pin, and where the verdict's cost numbers come from.
    references: References | None = None
    cost_basis: CostBasis | None = None
    # Cold start (C3/C5): the candidate models and their direct endpoints,
    # the generation params shared by every eval arm, and profile settings.
    models: tuple[ModelEntry, ...] = field(default=())
    generation: dict[str, Any] | None = None
    profile_config: ProfileConfig = field(default_factory=ProfileConfig)
    # Cold start (C2/C4, sub-issue-8 §2.1): the built-in template the config
    # is generated from, the provider config `recipe builtin init --config`
    # consumes (endpoints/credentials come only from there), and the
    # generation-side bootstrap settings. All three are lane-mode only; the
    # legacy domain-template path ignores them.
    template: TemplateConfig | None = None
    providers_config: str | None = None
    bootstrap_config: BootstrapConfig = field(default_factory=BootstrapConfig)
    # AG5: the models privacy/security traffic stays on (the agentic
    # bootstrap pins those lanes to them with adaptation bypassed).
    local_models: tuple[str, ...] = field(default=())
    # directory of the YAML file this spec was loaded from (from_yaml only);
    # workload.manifest resolves against it. Not part of the contract.
    spec_dir: str | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> ExperimentSpec:
        raw = _require_mapping(raw, "spec")
        api_version = raw.get("api_version")
        _require(
            api_version == API_VERSION,
            f"api_version must be {API_VERSION!r} (got {api_version!r})",
        )
        objective = Objective.from_dict(
            _require_mapping(raw.get("objective"), "objective")
        )
        workloads = _parse_workloads(raw)
        attribution = _require_mapping(raw.get("attribution", {}), "attribution")
        truth_space = attribution.get("truth_space", DEFAULT_TRUTH_SPACE)
        _require(
            truth_space in TRUTH_SPACES,
            f"attribution.truth_space must be one of {sorted(TRUTH_SPACES)}",
        )
        oracle_policy = OraclePolicy.from_dict(
            attribution.get("oracle_policy"), truth_space=truth_space
        )
        local_models_raw = raw.get("local_models", [])
        _require(
            isinstance(local_models_raw, list)
            and all(
                isinstance(model, str) and model.strip() for model in local_models_raw
            ),
            "local_models must be a list of non-empty model names",
        )
        local_models = tuple(local_models_raw)
        gates_raw = raw.get("gates", [])
        _require(isinstance(gates_raw, list), "gates must be a list")
        gates = tuple(Gate.from_dict(g, i) for i, g in enumerate(gates_raw))
        budget = Budget.from_dict(_require_mapping(raw.get("budget", {}), "budget"))
        stack_raw = raw.get("stack")
        stack = (
            Stack.from_dict(_require_mapping(stack_raw, "stack"))
            if stack_raw is not None
            else None
        )
        cost_model = _check_cost_model(raw.get("cost_model"))
        # M3: optional output-quality ceilings; absent means the defaults.
        evidence_raw = raw.get("evidence_quality")
        evidence_quality = (
            EvidenceQuality.from_dict(
                _require_mapping(evidence_raw, "evidence_quality")
            )
            if evidence_raw is not None
            else EvidenceQuality()
        )
        # M4: optional reference arms and cost basis.
        references_raw = raw.get("references")
        references = (
            References.from_dict(_require_mapping(references_raw, "references"))
            if references_raw is not None
            else None
        )
        cost_basis_raw = raw.get("cost_basis")
        cost_basis = (
            CostBasis.from_dict(_require_mapping(cost_basis_raw, "cost_basis"))
            if cost_basis_raw is not None
            else None
        )
        # C6: the replay cost model is derived from the cost basis, so a spec
        # declares one economic source, never two. A provider-pricing basis
        # with declared rates derives the cost model (relative to the
        # cheapest model); rates are optional so a basis can still be
        # declared for live-tier pricing alone. A cost_model basis must be
        # backed by the top-level section it names.
        if cost_basis is not None and cost_basis.source == "provider_pricing":
            _require(
                cost_model is None,
                "cost_basis.source=provider_pricing derives the replay cost model "
                "from assumed_pricing; remove the explicit cost_model section "
                "(declare one or the other, not both)",
            )
            if cost_basis.assumed_pricing:
                cost_model = _derived_cost_model(cost_basis.assumed_pricing)
        elif cost_basis is not None and cost_basis.source == "cost_model":
            _require(
                cost_model is not None,
                "cost_basis.source=cost_model requires the top-level cost_model "
                "section it derives from",
            )
        # Cold start (C3): candidate models, shared generation params and
        # profile settings.
        models_raw = raw.get("models", [])
        _require(isinstance(models_raw, list), "models must be a list")
        models = tuple(
            ModelEntry.from_dict(_require_mapping(m, f"models[{i}]"))
            for i, m in enumerate(models_raw)
        )
        names = [model.name for model in models]
        _require(
            len(names) == len(set(names)),
            "models names must be unique: " + ", ".join(sorted(names)),
        )
        generation_raw = raw.get("generation")
        generation = (
            _check_generation(_require_mapping(generation_raw, "generation"))
            if generation_raw is not None
            else None
        )
        profile_config = ProfileConfig.from_dict(
            _require_mapping(raw.get("profile", {}), "profile")
        )
        # Cold start (sub-issue-8 §2.1): the lane-mode template, provider
        # config and bootstrap settings; all optional (the legacy
        # domain-template path declares none of them).
        template_raw = raw.get("template")
        template = (
            TemplateConfig.from_dict(_require_mapping(template_raw, "template"))
            if template_raw is not None
            else None
        )
        providers_config = raw.get("providers_config")
        if providers_config is not None:
            _require(
                isinstance(providers_config, str) and providers_config.strip(),
                "providers_config must be a non-empty path string",
            )
        bootstrap_config = BootstrapConfig.from_dict(
            _require_mapping(raw.get("bootstrap", {}), "bootstrap")
        )
        if template is not None:
            _require(
                bool(models),
                "template requires the models section (the template path binds "
                "its lanes from the declared candidates)",
            )
            _require(
                providers_config is not None,
                "template requires providers_config (recipe builtin init consumes "
                "it; endpoints and credentials come only from there)",
            )
        # M4: a benefit constraint (cost bounded against a reference arm)
        # is meaningless without a declared cost basis - refuse it; a ref
        # bound also needs the references section to resolve against.
        for i, objective_constraint in enumerate(objective.constraints):
            if objective_constraint.bound in REF_BOUNDS:
                _require(
                    references is not None,
                    f"constraints[{i}] pins {objective_constraint.bound} but the spec "
                    "declares no references section",
                )
                if _is_cost_metric(objective_constraint.metric):
                    _require(
                        cost_basis is not None,
                        f"benefit constraint {objective_constraint.name!r} (cost bound "
                        "against a reference arm) requires a cost_basis section",
                    )
        # A5: `evaluation` is retired - replay and live are separate commands,
        # tiers are fixed by the CLI, not contracted per experiment.
        _require(
            "evaluation" not in raw,
            "spec key 'evaluation' is retired: remove it (replay and live are separate commands now)",
        )
        return cls(
            api_version=api_version,
            objective=objective,
            workloads=workloads,
            oracle_policy=oracle_policy,
            gates=gates,
            budget=budget,
            stack=stack,
            cost_model=cost_model,
            evidence_quality=evidence_quality,
            references=references,
            cost_basis=cost_basis,
            models=models,
            generation=generation,
            profile_config=profile_config,
            template=template,
            providers_config=providers_config,
            bootstrap_config=bootstrap_config,
            local_models=local_models,
        )

    @property
    def workload(self) -> Workload:
        """The default (first declared) workload - the historical
        single-workload accessor; multi-workload commands use
        select_workload / require_tunable_workload instead."""
        return self.workloads[0]

    def select_workload(self, name: str | None = None) -> Workload:
        """M5: resolve a --workload name (default: the first workload)."""
        if name is None:
            return self.workloads[0]
        for workload in self.workloads:
            if workload.name == name:
                return workload
        raise ContractError(
            f"unknown workload {name!r}; declared: {[w.name for w in self.workloads]}"
        )

    def require_tunable_workload(self, name: str | None = None) -> Workload:
        """The workload a tuning command (attribute/replay/propose) may read:
        role=transfer workloads are quarantined like frozen_test (M5)."""
        workload = self.select_workload(name)
        _require(
            workload.role == "tune",
            f"workload {workload.name!r} has role 'transfer' (the frozen_test "
            "rule applies to the whole workload): only the final verify may read it",
        )
        return workload

    def workload_guards(self) -> dict[str, float]:
        """M5: the auto per-workload "not worse" guards (name -> guard_pp).
        Every workload carries one; workload_guard_results (verdict) turns
        them into per-workload guard rows from the paired evidence."""
        return {workload.name: workload.guard_pp for workload in self.workloads}

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentSpec:
        path = Path(path)
        _require(path.is_file(), f"spec file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            try:
                raw = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                raise ContractError(
                    f"spec file is not valid YAML: {path} ({exc})"
                ) from exc
        spec = cls.from_dict(raw)
        return cls(**{**spec.__dict__, "spec_dir": str(path.parent)})

    def verify_workload(
        self,
        base_dir: str | Path | None = None,
        split: str | None = None,
        workload: str | None = None,
    ) -> Any:
        """Re-hash the declared split manifests and compare with declared hashes.

        Returns the verified content hash of the selected split (all declared
        splits when `split` is None). `workload` (M5) names one declared
        workload (default: the first); `workloads[*].splits[*].manifest`
        resolves against `base_dir`, defaulting to the spec file's directory.
        This is the A4 contract check: attribute / run / propose-* call it
        before touching any data, so a drifted dataset can never silently
        join two rounds of results. With the full three-way form, also checks
        the calib/dev/frozen_test slices are disjoint (case_id + prompt
        hash); with multiple workloads, checks that different workloads do
        not share cases either (the transfer quarantine).
        """
        base = (
            Path(base_dir)
            if base_dir is not None
            else (Path(self.spec_dir) if self.spec_dir else None)
        )
        if base is None:
            raise ContractError(
                "cannot locate the workload manifest: the spec was not loaded from a file; pass base_dir"
            )
        selected = self.select_workload(workload)
        names = [split] if split is not None else sorted(selected.splits)
        verified: dict[str, str] = {}
        for name in names:
            manifest = base / selected.split_manifest(name)
            if not manifest.is_file():
                raise ContractError(
                    f"dataset manifest not found: {manifest} (workload {selected.name!r} "
                    f"split {name!r} manifest is relative to the spec file)"
                )
            actual = dataset_content_hash(manifest)
            if actual != selected.split_hash(name):
                raise ContractError(
                    f"dataset hash mismatch for split {name!r}: spec={selected.split_hash(name)} file={actual}"
                )
            verified[name] = actual
            case_ids: set[str] | None = None
            # X3: the declared sr-bench rendering is verified against its own
            # hash and paired 1:1 with the cases manifest, so live evidence
            # can never bind to a rendering of a different case set than the
            # one replay ranked on.
            srbench_path = selected.split_srbench_dataset(name)
            if srbench_path is not None:
                srbench_file = base / srbench_path
                if not srbench_file.is_file():
                    raise ContractError(
                        f"srbench dataset not found: {srbench_file} (workload "
                        f"{selected.name!r} split {name!r} srbench_dataset is "
                        "relative to the spec file)"
                    )
                srbench_actual = dataset_content_hash(srbench_file)
                srbench_expected = selected.split_srbench_hash(name)
                if srbench_actual != srbench_expected:
                    raise ContractError(
                        f"srbench dataset hash mismatch for split {name!r}: "
                        f"spec={srbench_expected} file={srbench_actual}"
                    )
                case_ids, _ = _read_split_ids(manifest, name)
                srbench_ids = _read_srbench_case_ids(srbench_file)
                if case_ids != srbench_ids:
                    missing = sorted(case_ids - srbench_ids)
                    extra = sorted(srbench_ids - case_ids)
                    raise ContractError(
                        f"srbench dataset for split {name!r} does not pair 1:1 "
                        f"with the cases manifest (missing: {missing}, extra: {extra})"
                    )
            # Y3: the declared model_eval tasks rendering is verified the same
            # way, so the model_eval driver can never bind its arms to a
            # rendering of a different case set than the one replay ranked on.
            tasks_path = selected.split_model_eval_tasks(name)
            if tasks_path is not None:
                tasks_file = base / tasks_path
                if not tasks_file.is_file():
                    raise ContractError(
                        f"model_eval tasks file not found: {tasks_file} (workload "
                        f"{selected.name!r} split {name!r} model_eval_tasks is "
                        "relative to the spec file)"
                    )
                tasks_actual = dataset_content_hash(tasks_file)
                tasks_expected = selected.split_model_eval_tasks_hash(name)
                if tasks_actual != tasks_expected:
                    raise ContractError(
                        f"model_eval tasks hash mismatch for split {name!r}: "
                        f"spec={tasks_expected} file={tasks_actual}"
                    )
                if case_ids is None:
                    case_ids, _ = _read_split_ids(manifest, name)
                task_ids = _read_model_eval_task_ids(tasks_file)
                if case_ids != task_ids:
                    missing = sorted(case_ids - task_ids)
                    extra = sorted(task_ids - case_ids)
                    raise ContractError(
                        f"model_eval tasks for split {name!r} do not pair 1:1 "
                        f"with the cases manifest (missing: {missing}, extra: {extra})"
                    )
        if split is None:
            if not selected.legacy:
                _check_split_disjointness(base, selected)
            if len(self.workloads) > 1:
                _check_cross_workload_disjointness(base, self.workloads)
        if split is not None:
            return verified[split]
        return verified

    def require_dataset_file(
        self, path: str | Path, what: str, expected_hash: str | None = None
    ) -> str:
        """Assert a CLI-provided data file (--cases) is byte-identical to the
        frozen workload (or an explicit split hash). Raises ContractError
        (exit 2 via main) on drift or a missing file."""
        if expected_hash is None:
            expected_hash = self.workload.split_hash("dev")
        try:
            actual = dataset_content_hash(path)
        except OSError as exc:
            raise ContractError(f"cannot read {what}: {exc}") from exc
        if actual != expected_hash:
            raise ContractError(
                f"{what} hash mismatch: spec={expected_hash} file={actual}"
            )
        return actual

    def validate_policy_covers(self, labels: list[str]) -> list[str]:
        """Return oracle-policy coverage gaps for a dataset's label set."""
        return self.oracle_policy.covers(labels)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "api_version": self.api_version,
            "objective": self.objective.to_dict(),
            # M5: single workload keeps the historical `workload` key (and a
            # legacy spec round-trips its exact legacy shape); multi-workload
            # specs serialize the list form.
            "workload" if len(self.workloads) == 1 else "workloads": (
                self.workloads[0].to_dict()
                if len(self.workloads) == 1
                else [w.to_dict() for w in self.workloads]
            ),
            "attribution": {
                "oracle_policy": self.oracle_policy.to_dict(),
                # round-trip stability: the default namespace serializes nothing
                **(
                    {"truth_space": self.oracle_policy.truth_space}
                    if self.oracle_policy.truth_space != DEFAULT_TRUTH_SPACE
                    else {}
                ),
            },
            **({"local_models": list(self.local_models)} if self.local_models else {}),
            "gates": [g.to_dict() for g in self.gates],
            "budget": self.budget.to_dict(),
        }
        if self.stack is not None:
            out["stack"] = self.stack.to_dict()
        # C6: a cost_model derived from a provider-pricing basis is not
        # written back - re-parsing the output would re-derive it, and an
        # explicit key would trip the one-economic-source check.
        derived = None
        if (
            self.cost_basis is not None
            and self.cost_basis.source == "provider_pricing"
            and self.cost_basis.assumed_pricing
        ):
            derived = _derived_cost_model(self.cost_basis.assumed_pricing)
        if self.cost_model is not None and self.cost_model != derived:
            out["cost_model"] = dict(self.cost_model)
        if self.evidence_quality != EvidenceQuality():
            out["evidence_quality"] = self.evidence_quality.to_dict()
        if self.references is not None:
            out["references"] = self.references.to_dict()
        if self.cost_basis is not None:
            out["cost_basis"] = self.cost_basis.to_dict()
        if self.models:
            out["models"] = [model.to_dict() for model in self.models]
        if self.generation is not None:
            out["generation"] = dict(self.generation)
        if self.profile_config != ProfileConfig():
            out["profile"] = self.profile_config.to_dict()
        if self.template is not None:
            out["template"] = self.template.to_dict()
        if self.providers_config is not None:
            out["providers_config"] = self.providers_config
        if self.bootstrap_config != BootstrapConfig():
            out["bootstrap"] = self.bootstrap_config.to_dict()
        return out


@dataclass(frozen=True)
class Candidate:
    kind: str
    ref: str
    change_summary: str = ""
    proposer: str = "manual"
    insight_refs: tuple[str, ...] = field(default=())

    @classmethod
    def from_dict(cls, raw: dict) -> Candidate:
        kind = raw.get("kind")
        _require(
            kind in CANDIDATE_KINDS,
            f"candidate.kind must be one of {sorted(CANDIDATE_KINDS)}",
        )
        ref = raw.get("ref")
        _require(
            isinstance(ref, str) and ref.strip(),
            "candidate.ref must be a non-empty string",
        )
        provenance = raw.get("provenance", {})
        provenance = _require_mapping(provenance, "candidate.provenance")
        insight_refs = provenance.get("insight_refs", [])
        _require(
            isinstance(insight_refs, list),
            "candidate.provenance.insight_refs must be a list",
        )
        return cls(
            kind=kind,
            ref=ref,
            change_summary=str(raw.get("change_summary", "")),
            proposer=str(provenance.get("proposer", "manual")),
            insight_refs=tuple(str(r) for r in insight_refs),
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "ref": self.ref,
            "change_summary": self.change_summary,
            "provenance": {
                "proposer": self.proposer,
                "insight_refs": list(self.insight_refs),
            },
        }
