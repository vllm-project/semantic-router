#!/usr/bin/env python3
"""Fail-closed semantic contract for Go complexity lint configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass

import yaml

from go_complexity_identity import (
    COMPLEXITY_LINTERS,
    ComplexityLimits,
    complexity_limits_from_config,
)


class ComplexityConfigError(ValueError):
    """Raised when complexity lint configuration cannot be compared safely."""


_THRESHOLD_KEYS = {
    "cyclop": frozenset({"max-complexity"}),
    "funlen": frozenset({"lines", "statements"}),
    "gocognit": frozenset({"min-complexity"}),
    "interfacebloat": frozenset({"max"}),
    "nestif": frozenset({"min-complexity"}),
}
_EXCLUSION_KEYS = frozenset(
    {"rules", "paths", "paths-except", "presets", "generated", "warn-unused"}
)
_GENERATED_STRICTNESS = {"lax": 2, "strict": 1, "disable": 0}


@dataclass(frozen=True)
class ComplexityConfigContract:
    limits: ComplexityLimits
    enabled: frozenset[str]
    disabled: frozenset[str]
    linter_control: str
    run_analysis: str
    extra_settings: tuple[tuple[str, str], ...]
    exclusion_rules: tuple[tuple[str, str], ...]
    exclusion_paths: frozenset[str]
    exclusion_paths_except: frozenset[str]
    exclusion_presets: frozenset[str]
    generated_exclusions: str
    unsupported_exclusions: tuple[str, ...]
    issue_filters: tuple[str, ...]
    max_issues_per_linter: object
    max_same_issues: object
    uniq_by_line: object


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ComplexityConfigError(f"{label} must be an object")
    return value


def _string_set(value: object, label: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ComplexityConfigError(f"{label} must be a string list")
    return frozenset(value)


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ComplexityConfigError(
            "complexity configuration contains a non-canonical value"
        ) from exc


def _extra_linter_settings(settings: dict) -> tuple[tuple[str, str], ...]:
    extras = []
    for linter in sorted(COMPLEXITY_LINTERS):
        raw = _mapping(settings.get(linter), f"linters.settings.{linter}")
        extra = {
            key: value
            for key, value in raw.items()
            if key not in _THRESHOLD_KEYS[linter]
        }
        extras.append((linter, _canonical(extra)))
    return tuple(extras)


def _exclusion_rules(exclusions: dict) -> tuple[tuple[str, str], ...]:
    rules = exclusions.get("rules", [])
    if not isinstance(rules, list):
        raise ComplexityConfigError("linters.exclusions.rules must be a list")
    normalized: set[tuple[str, str]] = set()
    for index, raw_rule in enumerate(rules):
        rule = _mapping(raw_rule, f"linters.exclusions.rules[{index}]")
        targets = _string_set(rule.get("linters"), f"exclusion rule {index} linters")
        if not targets:
            targets = COMPLEXITY_LINTERS
        predicate = {key: value for key, value in rule.items() if key != "linters"}
        encoded = _canonical(predicate)
        for linter in targets & COMPLEXITY_LINTERS:
            normalized.add((linter, encoded))
    return tuple(sorted(normalized))


def _issue_filters(issues: dict) -> tuple[str, ...]:
    return tuple(
        sorted(
            key
            for key, value in issues.items()
            if value
            and (
                key.startswith("new")
                or key.startswith("exclude")
                or key == "whole-files"
            )
        )
    )


def parse_complexity_config(raw_bytes: bytes, source: str) -> ComplexityConfigContract:
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ComplexityConfigError(f"cannot parse {source}: {exc}") from exc
    config = _mapping(raw, source)
    linters = _mapping(config.get("linters"), f"{source} linters")
    run = _mapping(config.get("run", {}), f"{source} run")
    settings = _mapping(linters.get("settings"), f"{source} linters.settings")
    exclusions = _mapping(linters.get("exclusions", {}), f"{source} linters.exclusions")
    issues = _mapping(config.get("issues", {}), f"{source} issues")
    generated = exclusions.get("generated", "lax")
    if generated not in _GENERATED_STRICTNESS:
        raise ComplexityConfigError(
            f"{source} has unsupported generated exclusion mode {generated!r}"
        )
    return ComplexityConfigContract(
        limits=complexity_limits_from_config(config, source),
        enabled=_string_set(linters.get("enable"), f"{source} linters.enable"),
        disabled=_string_set(linters.get("disable"), f"{source} linters.disable"),
        linter_control=_canonical(
            {
                key: value
                for key, value in linters.items()
                if key not in {"enable", "disable", "settings", "exclusions"}
            }
        ),
        run_analysis=_canonical(
            {
                key: value
                for key, value in run.items()
                if key
                not in {
                    "timeout",
                    "concurrency",
                    "allow-parallel-runners",
                    "allow-serial-runners",
                }
            }
        ),
        extra_settings=_extra_linter_settings(settings),
        exclusion_rules=_exclusion_rules(exclusions),
        exclusion_paths=_string_set(
            exclusions.get("paths"), f"{source} linters.exclusions.paths"
        ),
        exclusion_paths_except=_string_set(
            exclusions.get("paths-except"),
            f"{source} linters.exclusions.paths-except",
        ),
        exclusion_presets=_string_set(
            exclusions.get("presets"), f"{source} linters.exclusions.presets"
        ),
        generated_exclusions=generated,
        unsupported_exclusions=tuple(sorted(set(exclusions) - _EXCLUSION_KEYS)),
        issue_filters=_issue_filters(issues),
        max_issues_per_linter=issues.get("max-issues-per-linter"),
        max_same_issues=issues.get("max-same-issues"),
        uniq_by_line=issues.get("uniq-by-line"),
    )


def validate_active_contract(contract: ComplexityConfigContract) -> list[str]:
    errors = []
    missing = COMPLEXITY_LINTERS - contract.enabled
    if missing:
        errors.append(f"complexity linters are disabled: {sorted(missing)}")
    disabled = COMPLEXITY_LINTERS & contract.disabled
    if disabled:
        errors.append(f"complexity linters are explicitly disabled: {sorted(disabled)}")
    if contract.uniq_by_line is not False:
        errors.append("issues.uniq-by-line must remain false")
    if contract.max_issues_per_linter != 0:
        errors.append("issues.max-issues-per-linter must remain unlimited (0)")
    if contract.max_same_issues != 0:
        errors.append("issues.max-same-issues must remain unlimited (0)")
    if contract.issue_filters:
        errors.append(
            f"complexity issue filtering is not allowed: {list(contract.issue_filters)}"
        )
    if contract.unsupported_exclusions:
        errors.append(
            "unsupported complexity exclusion settings: "
            f"{list(contract.unsupported_exclusions)}"
        )
    if contract.generated_exclusions != "disable":
        errors.append("generated-code complexity exclusions must remain disabled")
    return errors


def validate_not_looser(
    current: ComplexityConfigContract,
    baseline: ComplexityConfigContract,
) -> list[str]:
    errors = validate_active_contract(current)
    current_limits = vars(current.limits)
    baseline_limits = vars(baseline.limits)
    for name in sorted(current_limits):
        if current_limits[name] > baseline_limits[name]:
            errors.append(
                f"complexity threshold was loosened: {name} "
                f"{baseline_limits[name]} -> {current_limits[name]}"
            )
    if current.linter_control != baseline.linter_control:
        errors.append("Go complexity linter selection controls may not change")
    if current.run_analysis != baseline.run_analysis:
        errors.append("Go complexity analysis coverage controls may not change")
    if current.extra_settings != baseline.extra_settings:
        errors.append("unproven complexity linter setting changes are not allowed")
    new_rules = set(current.exclusion_rules) - set(baseline.exclusion_rules)
    if new_rules:
        errors.append("new or broadened complexity exclusion rules are not allowed")
    if current.exclusion_paths - baseline.exclusion_paths:
        errors.append("new global complexity exclusion paths are not allowed")
    if baseline.exclusion_paths_except - current.exclusion_paths_except:
        errors.append("removing complexity exclusion path exceptions is not allowed")
    if current.exclusion_presets - baseline.exclusion_presets:
        errors.append("new complexity exclusion presets are not allowed")
    if (
        _GENERATED_STRICTNESS[current.generated_exclusions]
        > _GENERATED_STRICTNESS[baseline.generated_exclusions]
    ):
        errors.append("generated-code complexity exclusions were loosened")
    return errors
