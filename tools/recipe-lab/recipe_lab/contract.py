"""Experiment contract: ExperimentSpec / Candidate / Gate / oracle policy.

The contract is the source of truth for one improvement campaign. It is
loaded from YAML (api_version: recipe-lab/v1alpha1) and validated eagerly:
invalid contracts are rejected at load time with actionable errors, never
mid-experiment.

Layout follows the recipe-lab design §5.3:
  objective:    metric_key / direction / min_improvement_pct / replay_metric
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
# B: the frozen dataset is split three ways; frozen_test is quarantined from
# replay/propose work and may only be touched by live-verdict --final, once.
SPLIT_NAMES = ("calib", "dev", "frozen_test")
RUNNABLE_SPLITS = ("calib", "dev")
TIERS = {"replay", "preview", "live"}
CANDIDATE_KINDS = {"config", "artifact"}
VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE", "RANKED"}
# Numeric predicate ops for oracle-policy conditions and truth facts, with
# the router's semantics (pkg/decision engine numericPredicateMatches: e.g.
# gte fails when value < bound).
PREDICATE_OPS = ("gte", "lte", "gt", "lt")
LIVE_VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}
REPLAY_VERDICTS = {"RANKED"}


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


def _check_split_disjointness(base: Path, workload: Workload) -> None:
    """The three-way split must be disjoint: no case_id and no prompt content
    shared between two splits (else calibration data leaks into the frozen
    test and the final verdict is no longer honest)."""
    seen: dict[str, tuple[set[str], set[str]]] = {}
    for name, split in workload.splits.items():
        manifest = base / split.manifest
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
                        f"split {name!r} line {line_no} is not valid JSON: {exc}"
                    ) from exc
                case_id = row.get("case_id")
                if not isinstance(case_id, str) or not case_id:
                    raise ContractError(
                        f"split {name!r} line {line_no} is missing a non-empty case_id"
                    )
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise ContractError(
                        f"split {name!r} line {line_no} is missing a non-empty prompt"
                    )
                case_ids.add(case_id)
                prompt_hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        seen[name] = (case_ids, prompt_hashes)

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


CONSTRAINT_OPS = ("gte", "lte")


@dataclass(frozen=True)
class Constraint:
    """One economic constraint on the live verdict (E).

    Constraints are distinct from gates: a gate is a safety veto, a
    constraint is an economic condition (e.g. cost must not regress).
    `bound` is either a number or the literal "baseline" (the baseline
    arm's measured value of the same metric); `tolerance_pct` relaxes the
    bound by that percentage in the permissive direction.
    """

    name: str
    metric: str
    op: str
    bound: str | float  # "baseline" or a numeric bound
    tolerance_pct: float = 0.0

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
        if bound != "baseline":
            bound = _require_number(bound, f"constraints[{index}].bound")
        tolerance = _require_number(
            raw.get("tolerance_pct", 0.0), f"constraints[{index}].tolerance_pct"
        )
        _require(tolerance >= 0, f"constraints[{index}].tolerance_pct must be >= 0")
        return cls(
            name=name, metric=metric, op=op, bound=bound, tolerance_pct=tolerance
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metric": self.metric,
            "op": self.op,
            "bound": self.bound,
            "tolerance_pct": self.tolerance_pct,
        }


@dataclass(frozen=True)
class Objective:
    metric_key: str
    direction: str
    min_improvement_pct: float
    replay_metric: str = "routed_accuracy"
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
            constraints=constraints,
        )

    def to_dict(self) -> dict:
        out = {
            "metric_key": self.metric_key,
            "direction": self.direction,
            "min_improvement_pct": self.min_improvement_pct,
            "replay_metric": self.replay_metric,
        }
        if self.constraints:
            out["constraints"] = [c.to_dict() for c in self.constraints]
        return out


@dataclass(frozen=True)
class SplitSpec:
    """One named slice of the frozen dataset (B)."""

    manifest: str
    content_hash: str

    @classmethod
    def from_dict(cls, raw: dict, name: str) -> SplitSpec:
        manifest = raw.get("manifest")
        _require(
            isinstance(manifest, str) and manifest.strip(),
            f"workload.splits.{name}.manifest must be a non-empty string",
        )
        content_hash = raw.get("content_hash")
        _require(
            isinstance(content_hash, str) and content_hash.strip(),
            f"workload.splits.{name}.content_hash is required (frozen dataset hash)",
        )
        return cls(manifest=manifest, content_hash=content_hash)

    def to_dict(self) -> dict:
        return {"manifest": self.manifest, "content_hash": self.content_hash}


@dataclass(frozen=True)
class Workload:
    profile: str
    split_rule: str | None
    splits: dict[str, SplitSpec]
    legacy: bool

    @classmethod
    def from_dict(cls, raw: dict) -> Workload:
        profile = raw.get("profile", "quick")
        _require(
            profile in PROFILES, f"workload.profile must be one of {sorted(PROFILES)}"
        )
        if "splits" in raw:
            split_rule = raw.get("split_rule")
            _require(
                isinstance(split_rule, str) and split_rule.strip(),
                "workload.split_rule must be a non-empty string when workload.splits is used "
                "(free text like 'group by source_id; seed 42; 40/30/30'; it is written to provenance)",
            )
            splits_raw = raw.get("splits")
            _require(isinstance(splits_raw, dict), "workload.splits must be a mapping")
            _require(
                set(splits_raw) == set(SPLIT_NAMES),
                f"workload.splits must declare exactly {sorted(SPLIT_NAMES)}",
            )
            splits = {
                name: SplitSpec.from_dict(
                    _require_mapping(v, f"workload.splits.{name}"), name
                )
                for name, v in splits_raw.items()
            }
            return cls(
                profile=profile, split_rule=split_rule, splits=splits, legacy=False
            )
        # legacy single-dataset form: treated as a dev-only workload
        manifest = raw.get("manifest")
        _require(
            isinstance(manifest, str) and manifest.strip(),
            "workload.manifest must be a non-empty string",
        )
        content_hash = raw.get("content_hash")
        _require(
            isinstance(content_hash, str) and content_hash.strip(),
            "workload.content_hash is required (frozen dataset hash; cross-round comparability depends on it)",
        )
        return cls(
            profile=profile,
            split_rule=None,
            splits={"dev": SplitSpec(manifest=manifest, content_hash=content_hash)},
            legacy=True,
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

    def to_dict(self) -> dict:
        if self.legacy:
            split = self.splits["dev"]
            return {
                "manifest": split.manifest,
                "content_hash": split.content_hash,
                "profile": self.profile,
            }
        return {
            "profile": self.profile,
            "split_rule": self.split_rule,
            "splits": {name: split.to_dict() for name, split in self.splits.items()},
        }


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

    @classmethod
    def from_dict(cls, raw: Any) -> OraclePolicy:
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
            return cls(rules=rules, default=default)
        for label, model in raw.items():
            _require(
                isinstance(label, str) and label.strip(),
                f"oracle policy label must be a non-empty string: {label!r}",
            )
            _require(
                isinstance(model, str) and model.strip(),
                f"oracle policy model for label {label!r} must be a non-empty string",
            )
        return cls(mapping=dict(raw))

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


def resolve_constraint(constraint: dict[str, Any]) -> dict[str, Any]:
    """Status and resolved bound for one constraint evidence entry (E).

    Shared by the live verdict and the replay cost proxy (E2): the replay
    tier feeds the same evidence shape with the routed_cost_index proxy
    substituted for the live sr-bench measurement.

    `constraint` carries the declared fields (name/metric/op/bound/
    tolerance_pct) plus the measured `baseline` and `candidate` values
    (None = unknown, e.g. sr-bench serializes unknown spend as null -
    unknown usage must never count as zero). A "baseline" bound resolves
    against the measured baseline value; tolerance_pct relaxes the bound
    in the permissive direction. Not-measurable constraints are
    unevaluated, never failed.
    """
    name = constraint["name"]
    op = constraint["op"]
    bound = constraint["bound"]
    tolerance = float(constraint.get("tolerance_pct", 0.0))
    baseline = constraint.get("baseline")
    candidate = constraint.get("candidate")
    out = dict(constraint)
    out["bound_value"] = None
    out["status"] = "not_evaluated"
    if candidate is None or (bound == "baseline" and baseline is None):
        return out
    bound_value = float(baseline) if bound == "baseline" else float(bound)
    # lte allows a worst case above the bound, gte below it
    bound_value = (
        bound_value * (1 + tolerance / 100.0)
        if op == "lte"
        else bound_value * (1 - tolerance / 100.0)
    )
    out["bound_value"] = bound_value
    violated = candidate > bound_value if op == "lte" else candidate < bound_value
    out["status"] = "violated" if violated else "pass"
    if violated:
        out["reason"] = (
            f"constraint {name} violated (candidate {candidate:.6f} {'>' if op == 'lte' else '<'} bound {bound_value:.6f})"
        )
    return out


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
class ExperimentSpec:
    api_version: str
    objective: Objective
    workload: Workload
    oracle_policy: OraclePolicy
    gates: tuple[Gate, ...] = field(default=())
    budget: Budget = field(
        default_factory=lambda: Budget(rounds=1, max_minutes_per_trial=120.0)
    )
    stack: Stack | None = None
    # E2: optional model -> relative-cost map for the replay cost proxy.
    cost_model: dict[str, float] | None = None
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
        workload = Workload.from_dict(_require_mapping(raw.get("workload"), "workload"))
        attribution = _require_mapping(raw.get("attribution", {}), "attribution")
        oracle_policy = OraclePolicy.from_dict(attribution.get("oracle_policy"))
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
        # A5: `evaluation` is retired - replay and live are separate commands,
        # tiers are fixed by the CLI, not contracted per experiment.
        _require(
            "evaluation" not in raw,
            "spec key 'evaluation' is retired: remove it (replay and live are separate commands now)",
        )
        return cls(
            api_version=api_version,
            objective=objective,
            workload=workload,
            oracle_policy=oracle_policy,
            gates=gates,
            budget=budget,
            stack=stack,
            cost_model=cost_model,
        )

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
        self, base_dir: str | Path | None = None, split: str | None = None
    ) -> str:
        """Re-hash the declared split manifests and compare with declared hashes.

        Returns the verified content hash of the selected split (all declared
        splits when `split` is None). `workload.splits[*].manifest` resolves
        against `base_dir`, defaulting to the spec file's directory. This is
        the A4 contract check: attribute / run / propose-* call it before
        touching any data, so a drifted dataset can never silently join two
        rounds of results. With the full three-way form, also checks the
        calib/dev/frozen_test slices are disjoint (case_id + prompt hash).
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
        names = [split] if split is not None else sorted(self.workload.splits)
        verified: dict[str, str] = {}
        for name in names:
            manifest = base / self.workload.split_manifest(name)
            if not manifest.is_file():
                raise ContractError(
                    f"dataset manifest not found: {manifest} (workload.splits.{name}.manifest is relative to the spec file)"
                )
            actual = dataset_content_hash(manifest)
            if actual != self.workload.split_hash(name):
                raise ContractError(
                    f"dataset hash mismatch for split {name!r}: spec={self.workload.split_hash(name)} file={actual}"
                )
            verified[name] = actual
        if split is None and not self.workload.legacy:
            _check_split_disjointness(base, self.workload)
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
            "workload": self.workload.to_dict(),
            "attribution": {"oracle_policy": self.oracle_policy.to_dict()},
            "gates": [g.to_dict() for g in self.gates],
            "budget": self.budget.to_dict(),
        }
        if self.stack is not None:
            out["stack"] = self.stack.to_dict()
        if self.cost_model is not None:
            out["cost_model"] = dict(self.cost_model)
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
