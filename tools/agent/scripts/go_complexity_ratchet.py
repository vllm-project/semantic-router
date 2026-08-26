#!/usr/bin/env python3
"""Evaluation and reporting for the changed-file Go complexity debt ratchet."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from go_complexity_baseline import resolve_frozen_baseline
from go_complexity_config import parse_complexity_config, validate_not_looser
from go_complexity_identity import (
    ComplexityFinding,
    ComplexityFindingNormalizer,
    ComplexityIdentity,
    ComplexityIdentityError,
)
from go_complexity_manifest import (
    ComplexityRatchetError,
    DebtEntry,
    DebtManifest,
    identity_sort_key,
    load_manifest,
    validate_manifest_contract,
    validate_manifest_contract_bytes,
    validate_manifest_delta,
)
from go_complexity_source_policy import validate_changed_source_policy


@dataclass(frozen=True)
class MetricChange:
    entry: DebtEntry
    finding: ComplexityFinding | None


@dataclass
class RatchetResult:
    known: list[ComplexityFinding] = field(default_factory=list)
    new: list[ComplexityFinding] = field(default_factory=list)
    worsened: list[MetricChange] = field(default_factory=list)
    improved: list[MetricChange] = field(default_factory=list)
    stale: list[DebtEntry] = field(default_factory=list)
    contract_errors: list[str] = field(default_factory=list)
    bootstrap: bool = False

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.new,
                self.worsened,
                self.improved,
                self.stale,
                self.contract_errors,
            )
        )


def evaluate_ratchet(
    findings: list[ComplexityFinding],
    manifest: DebtManifest,
    changed_paths: set[str],
    evaluated_paths: set[str] | None = None,
    require_complete_manifest: bool = False,
) -> RatchetResult:
    result = RatchetResult()
    finding_by_identity: dict[ComplexityIdentity, ComplexityFinding] = {}
    for finding in findings:
        previous = finding_by_identity.get(finding.identity)
        if previous is not None:
            result.contract_errors.append(
                f"multiple diagnostics resolved to {finding.identity.path} "
                f"{finding.identity.declaration} ({finding.identity.linter})"
            )
            continue
        finding_by_identity[finding.identity] = finding

    entry_by_identity = manifest.by_identity()
    for identity, finding in sorted(
        finding_by_identity.items(), key=lambda item: identity_sort_key(item[0])
    ):
        entry = entry_by_identity.get(identity)
        if entry is None:
            result.new.append(finding)
            continue
        if finding.limit != entry.limit:
            result.contract_errors.append(
                f"configured limit changed for {identity.path} "
                f"{identity.declaration} ({identity.linter})"
            )
        if finding.observed > entry.observed:
            result.worsened.append(MetricChange(entry, finding))
        elif finding.observed < entry.observed:
            result.improved.append(MetricChange(entry, finding))
        else:
            result.known.append(finding)

    stale_scope = evaluated_paths if evaluated_paths is not None else changed_paths
    if require_complete_manifest:
        manifest_paths = {entry.identity.path for entry in manifest.entries}
        for path in sorted(manifest_paths - stale_scope):
            result.contract_errors.append(
                f"complexity debt path is outside the evaluated scope: {path}"
            )
    for identity, entry in sorted(
        entry_by_identity.items(), key=lambda item: identity_sort_key(item[0])
    ):
        if identity.path not in stale_scope:
            continue
        if identity not in finding_by_identity:
            result.stale.append(entry)
    return result


def run_complexity_ratchet(
    records: list[dict],
    repo_root: Path,
    config_path: Path,
    manifest_path: Path,
    changed_paths: set[str],
    base_ref: str,
    tool_version: str,
    covered_paths: set[str] | None = None,
    evaluated_paths: set[str] | None = None,
    require_complete_manifest: bool = False,
) -> RatchetResult:
    result = RatchetResult()
    try:
        manifest = load_manifest(manifest_path)
    except ComplexityRatchetError as exc:
        result.contract_errors.append(str(exc))
        return result

    result.contract_errors.extend(
        validate_manifest_contract(manifest, config_path, tool_version)
    )
    try:
        baseline = resolve_frozen_baseline(repo_root, base_ref)
        result.bootstrap = baseline.bootstrap
        result.contract_errors.extend(
            validate_manifest_contract_bytes(
                baseline.manifest,
                baseline.config_bytes,
                f"{baseline.commit}:Go complexity config",
                baseline.manifest.tool_version,
            )
        )
        if baseline.manifest.tool_version != tool_version:
            result.contract_errors.append(
                "golangci-lint version differs from the frozen complexity baseline"
            )
        current_contract = parse_complexity_config(
            config_path.read_bytes(), str(config_path)
        )
        baseline_contract = parse_complexity_config(
            baseline.config_bytes, f"{baseline.commit}:Go complexity config"
        )
        if baseline.bootstrap:
            target_contract = parse_complexity_config(
                baseline.target_config_bytes,
                f"{base_ref}:target Go complexity config",
            )
            result.contract_errors.extend(
                validate_not_looser(baseline_contract, target_contract)
            )
        result.contract_errors.extend(
            validate_not_looser(current_contract, baseline_contract)
        )
        delta_errors, bootstrap = validate_manifest_delta(
            manifest, baseline.manifest, changed_paths
        )
        result.contract_errors.extend(delta_errors)
        if bootstrap:
            result.contract_errors.append(
                "complexity baseline resolution returned an unfrozen manifest"
            )
        result.contract_errors.extend(
            validate_changed_source_policy(
                repo_root,
                base_ref,
                changed_paths,
                covered_paths,
            )
        )
    except (ComplexityRatchetError, OSError, ValueError) as exc:
        result.contract_errors.append(str(exc))

    normalizer = ComplexityFindingNormalizer(repo_root, config_path)
    findings: list[ComplexityFinding] = []
    for record in records:
        try:
            findings.append(normalizer.normalize(record))
        except (ComplexityIdentityError, OSError, TypeError, ValueError) as exc:
            result.contract_errors.append(str(exc))

    evaluated = evaluate_ratchet(
        findings,
        manifest,
        changed_paths,
        evaluated_paths,
        require_complete_manifest,
    )
    result.known.extend(evaluated.known)
    result.new.extend(evaluated.new)
    result.worsened.extend(evaluated.worsened)
    result.improved.extend(evaluated.improved)
    result.stale.extend(evaluated.stale)
    result.contract_errors.extend(evaluated.contract_errors)
    return result


def print_ratchet_result(result: RatchetResult) -> None:
    print(
        "Go complexity debt: "
        f"known={len(result.known)} "
        f"new={len(result.new)} "
        f"worsened={len(result.worsened)} "
        f"improved={len(result.improved)} "
        f"stale={len(result.stale)} "
        f"contract_errors={len(result.contract_errors)} "
        f"bootstrap={'yes' if result.bootstrap else 'no'}"
    )
    for message in result.contract_errors:
        print(f"complexity contract: {message}")
    for finding in result.new:
        _print_finding("new", finding)
    for change in result.worsened:
        assert change.finding is not None
        _print_finding(
            f"worsened from {change.entry.observed}",
            change.finding,
        )
    for change in result.improved:
        assert change.finding is not None
        _print_finding(
            f"improved from {change.entry.observed}; lower the debt manifest",
            change.finding,
        )
    for entry in result.stale:
        print(
            f"{entry.identity.path}: {entry.identity.declaration} "
            f"({entry.identity.linter}) no longer violates its limit; "
            "remove the stale debt entry"
        )


def _print_finding(label: str, finding: ComplexityFinding) -> None:
    print(
        f"{finding.identity.path}:{finding.line}:{finding.column}: "
        f"{label}: {finding.identity.declaration} "
        f"{finding.observed} > {finding.limit} ({finding.identity.linter})"
    )
