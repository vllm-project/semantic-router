"""Experiment contract slice for misroute attribution (sub-issue 1).

Loaded from YAML (api_version: recipe-lab/v1alpha1) and validated eagerly:
invalid contracts are rejected at load time with actionable errors, never
mid-experiment.

This first slice carries the objective, the frozen workload identity
(manifest + content hash) and the oracle policy - the legacy label -> model
mapping or the multi-signal rules/default form (first match wins; an
expectation names the intended decision plus an optional model set) - so
attribution never reimplements router logic.

The remaining contract surface (experiment/budget/candidate/gates/stack,
split disjointness and workload re-verification) arrives with the replay
loop (sub-issue 2).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import API_VERSION

DIRECTIONS = {"minimize", "maximize"}
PROFILES = {"smoke", "quick", "standard"}
REPLAY_METRICS = {"routed_accuracy"}
# B: the frozen dataset is split three ways; frozen_test is quarantined from
# replay/propose work and may only be touched by live-verdict --final, once.
SPLIT_NAMES = ("calib", "dev", "frozen_test")
PREDICATE_OPS = ("gte", "lte", "gt", "lt")


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


@dataclass(frozen=True)
class Objective:
    metric_key: str
    direction: str
    min_improvement_pct: float
    replay_metric: str = "routed_accuracy"

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
        return cls(
            metric_key=metric_key,
            direction=direction,
            min_improvement_pct=pct,
            replay_metric=replay_metric,
        )

    def to_dict(self) -> dict:
        out = {
            "metric_key": self.metric_key,
            "direction": self.direction,
            "min_improvement_pct": self.min_improvement_pct,
            "replay_metric": self.replay_metric,
        }
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
class ExperimentSpec:
    api_version: str
    objective: Objective
    workload: Workload
    oracle_policy: OraclePolicy
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

    def validate_policy_covers(self, labels: list[str]) -> list[str]:
        """Return oracle-policy coverage gaps for a dataset's label set."""
        return self.oracle_policy.covers(labels)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "api_version": self.api_version,
            "objective": self.objective.to_dict(),
            "workload": self.workload.to_dict(),
            "attribution": {"oracle_policy": self.oracle_policy.to_dict()},
        }
        return out
