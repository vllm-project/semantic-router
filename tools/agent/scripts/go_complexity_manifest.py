#!/usr/bin/env python3
"""Manifest contract for the declaration-level Go complexity debt ratchet."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml
from go_complexity_config import parse_complexity_config, validate_active_contract
from go_complexity_identity import (
    COMPLEXITY_LINTERS,
    ComplexityFinding,
    ComplexityIdentity,
)

SCHEMA_VERSION = 1
IDENTITY_SCHEMA_VERSION = 1
IDENTITY_PARSER = "tree-sitter-go"
IDENTITY_CORE = "tree-sitter"
IDENTITY_IMPLEMENTATION_PATH = Path(__file__).with_name("go_complexity_identity.py")
MANIFEST_RELATIVE_PATH = Path("tools/linter/go/complexity-debt.yaml")
_TOOL_VERSION_PATTERN = re.compile(r"\bversion (?P<version>\d+\.\d+\.\d+)\b")


class ComplexityRatchetError(ValueError):
    """Raised when the ratchet contract cannot be evaluated safely."""


@dataclass(frozen=True)
class DebtEntry:
    identity: ComplexityIdentity
    observed: int
    limit: int
    owner: str
    debt: str


@dataclass(frozen=True)
class DebtManifest:
    tool_version: str
    config_sha256: str
    entries: tuple[DebtEntry, ...]
    additions: str = "deny"
    baseline: str = "committed-freeze-marker"
    identity_schema_version: int = IDENTITY_SCHEMA_VERSION
    identity_parser: str = IDENTITY_PARSER
    identity_parser_version: str = ""
    identity_core_version: str = ""
    identity_implementation_sha256: str = ""

    def by_identity(self) -> dict[ComplexityIdentity, DebtEntry]:
        return {entry.identity: entry for entry in self.entries}


def config_digest(config_path: Path) -> str:
    return config_digest_bytes(config_path.read_bytes())


def config_digest_bytes(config_bytes: bytes) -> str:
    return f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"


def detect_tool_version(command: str, repo_root: Path) -> str:
    result = subprocess.run(
        [command, "version"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ComplexityRatchetError(
            f"cannot resolve golangci-lint version: {result.stderr.strip()}"
        )
    match = _TOOL_VERSION_PATTERN.search(result.stdout)
    if match is None:
        raise ComplexityRatchetError(
            f"cannot parse golangci-lint version output: {result.stdout.strip()}"
        )
    return match.group("version")


def detect_identity_parser_version() -> str:
    try:
        return version(IDENTITY_PARSER)
    except PackageNotFoundError as exc:
        raise ComplexityRatchetError(
            f"cannot resolve {IDENTITY_PARSER} package version"
        ) from exc


def detect_identity_core_version() -> str:
    try:
        return version(IDENTITY_CORE)
    except PackageNotFoundError as exc:
        raise ComplexityRatchetError(
            f"cannot resolve {IDENTITY_CORE} package version"
        ) from exc


def detect_identity_implementation_digest() -> str:
    try:
        return config_digest_bytes(IDENTITY_IMPLEMENTATION_PATH.read_bytes())
    except OSError as exc:
        raise ComplexityRatchetError(
            f"cannot read complexity identity implementation: {exc}"
        ) from exc


def identity_sort_key(identity: ComplexityIdentity) -> tuple[str, str, str, str]:
    return (
        identity.path,
        identity.declaration,
        identity.linter,
        identity.site,
    )


def _entry_string(value: object, label: str, index: int) -> str:
    if not isinstance(value, str) or not value:
        raise ComplexityRatchetError(f"manifest entry {index} has invalid {label}")
    return value


def _parse_identity(raw: dict, index: int) -> ComplexityIdentity:
    path = _entry_string(raw["path"], "path", index)
    declaration = _entry_string(raw["declaration"], "declaration", index)
    linter = _entry_string(raw["linter"], "linter", index)
    site_raw = raw.get("site", "")
    if not isinstance(site_raw, str):
        raise ComplexityRatchetError(f"manifest entry {index} has invalid site")
    if not path.endswith(".go") or path.startswith("/"):
        raise ComplexityRatchetError(f"manifest entry {index} has invalid path")
    if ".." in Path(path).parts:
        raise ComplexityRatchetError(f"manifest entry {index} escapes the repository")
    if linter not in COMPLEXITY_LINTERS:
        raise ComplexityRatchetError(
            f"manifest entry {index} has unsupported linter {linter!r}"
        )
    if linter == "nestif" and not site_raw.startswith("sha256:"):
        raise ComplexityRatchetError(
            f"manifest entry {index} requires a normalized nestif site hash"
        )
    if linter == "funlen" and site_raw not in {"lines", "statements"}:
        raise ComplexityRatchetError(
            f"manifest entry {index} requires a funlen metric discriminator"
        )
    if linter not in {"nestif", "funlen"} and site_raw:
        raise ComplexityRatchetError(
            f"manifest entry {index} has an unexpected site discriminator"
        )
    return ComplexityIdentity(path, declaration, linter, site_raw)


def _entry_metric(raw: dict, label: str, index: int) -> int:
    value = raw[label]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ComplexityRatchetError(
            f"manifest entry {index} requires integer observed and limit"
        )
    return value


def _parse_entry(raw_value: object, index: int) -> DebtEntry:
    if not isinstance(raw_value, dict):
        raise ComplexityRatchetError(f"manifest entry {index} must be an object")
    required_keys = {
        "path",
        "declaration",
        "linter",
        "observed",
        "limit",
        "owner",
        "debt",
    }
    allowed_keys = required_keys | {"site"}
    missing = required_keys - raw_value.keys()
    unknown = raw_value.keys() - allowed_keys
    if missing or unknown:
        raise ComplexityRatchetError(
            f"manifest entry {index} has missing keys {sorted(missing)} "
            f"and unknown keys {sorted(unknown)}"
        )

    identity = _parse_identity(raw_value, index)
    owner = _entry_string(raw_value["owner"], "owner", index)
    debt = _entry_string(raw_value["debt"], "debt", index)
    observed = _entry_metric(raw_value, "observed", index)
    limit = _entry_metric(raw_value, "limit", index)
    if observed <= limit:
        raise ComplexityRatchetError(
            f"manifest entry {index} does not exceed its configured limit"
        )
    return DebtEntry(
        identity=identity,
        observed=observed,
        limit=limit,
        owner=owner,
        debt=debt,
    )


def parse_manifest_text(raw_text: str, source: str) -> DebtManifest:
    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ComplexityRatchetError(f"cannot parse {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ComplexityRatchetError(f"{source} must contain a manifest object")
    if set(raw) != {"schema_version", "identity", "tool", "policy", "entries"}:
        raise ComplexityRatchetError(
            f"{source} must contain exactly schema_version, identity, tool, policy, "
            "and entries"
        )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ComplexityRatchetError(
            f"{source} has unsupported schema version {raw['schema_version']!r}"
        )
    tool = raw["tool"]
    identity = raw["identity"]
    policy = raw["policy"]
    entries_raw = raw["entries"]
    if not isinstance(tool, dict) or set(tool) != {
        "name",
        "version",
        "config_sha256",
    }:
        raise ComplexityRatchetError(f"{source} has an invalid tool contract")
    if tool["name"] != "golangci-lint":
        raise ComplexityRatchetError(f"{source} has an unsupported lint tool")
    if not isinstance(tool["version"], str) or not isinstance(
        tool["config_sha256"], str
    ):
        raise ComplexityRatchetError(f"{source} has invalid tool metadata")
    if not isinstance(identity, dict) or set(identity) != {
        "schema_version",
        "parser",
        "parser_version",
        "core_version",
        "implementation_sha256",
    }:
        raise ComplexityRatchetError(f"{source} has an invalid identity contract")
    if (
        identity["schema_version"] != IDENTITY_SCHEMA_VERSION
        or identity["parser"] != IDENTITY_PARSER
        or not isinstance(identity["parser_version"], str)
        or not identity["parser_version"]
        or not isinstance(identity["core_version"], str)
        or not identity["core_version"]
        or not isinstance(identity["implementation_sha256"], str)
        or not identity["implementation_sha256"].startswith("sha256:")
    ):
        raise ComplexityRatchetError(f"{source} has an unsupported identity contract")
    expected_policy = {
        "additions": "deny",
        "baseline": "committed-freeze-marker",
    }
    if not isinstance(policy, dict) or policy != expected_policy:
        raise ComplexityRatchetError(
            f"{source} must use the committed complexity freeze marker"
        )
    if not isinstance(entries_raw, list):
        raise ComplexityRatchetError(f"{source} entries must be a list")

    entries = tuple(_parse_entry(item, index) for index, item in enumerate(entries_raw))
    identities = [entry.identity for entry in entries]
    if identities != sorted(identities, key=identity_sort_key):
        raise ComplexityRatchetError(
            f"{source} entries are not deterministically sorted"
        )
    if len(set(identities)) != len(identities):
        raise ComplexityRatchetError(f"{source} contains duplicate debt identities")
    return DebtManifest(
        tool_version=tool["version"],
        config_sha256=tool["config_sha256"],
        entries=entries,
        additions=policy["additions"],
        baseline=policy["baseline"],
        identity_schema_version=identity["schema_version"],
        identity_parser=identity["parser"],
        identity_parser_version=identity["parser_version"],
        identity_core_version=identity["core_version"],
        identity_implementation_sha256=identity["implementation_sha256"],
    )


def load_manifest(path: Path) -> DebtManifest:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComplexityRatchetError(
            f"cannot read complexity debt manifest: {exc}"
        ) from exc
    return parse_manifest_text(raw_text, str(path))


def validate_manifest_contract(
    manifest: DebtManifest,
    config_path: Path,
    tool_version: str,
) -> list[str]:
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        return [f"cannot read active Go complexity config: {exc}"]
    return validate_manifest_contract_bytes(
        manifest, config_bytes, str(config_path), tool_version
    )


def validate_manifest_contract_bytes(
    manifest: DebtManifest,
    config_bytes: bytes,
    source: str,
    tool_version: str,
) -> list[str]:
    errors: list[str] = []
    current_digest = config_digest_bytes(config_bytes)
    if manifest.config_sha256 != current_digest:
        errors.append(
            "complexity debt config digest does not match the active Go lint config"
        )
    if manifest.tool_version != tool_version:
        errors.append(
            "complexity debt tool version does not match the active golangci-lint"
        )
    try:
        parser_version = detect_identity_parser_version()
    except ComplexityRatchetError as exc:
        errors.append(str(exc))
        parser_version = ""
    if manifest.identity_parser_version != parser_version:
        errors.append(
            "complexity debt identity parser version does not match the active parser"
        )
    try:
        core_version = detect_identity_core_version()
        implementation_digest = detect_identity_implementation_digest()
    except ComplexityRatchetError as exc:
        errors.append(str(exc))
        core_version = ""
        implementation_digest = ""
    if manifest.identity_core_version != core_version:
        errors.append(
            "complexity debt identity core version does not match the active parser"
        )
    if manifest.identity_implementation_sha256 != implementation_digest:
        errors.append(
            "complexity debt identity implementation digest does not match"
        )
    try:
        contract = parse_complexity_config(config_bytes, source)
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    errors.extend(validate_active_contract(contract))
    limits = contract.limits
    expected_limits = {
        "cyclop": limits.cyclop,
        "gocognit": limits.gocognit,
        "interfacebloat": limits.interfacebloat,
        "nestif": limits.nestif,
    }
    for entry in manifest.entries:
        if entry.identity.linter == "funlen":
            expected = (
                limits.funlen_lines
                if entry.identity.site == "lines"
                else limits.funlen_statements
            )
        else:
            expected = expected_limits[entry.identity.linter]
        if entry.limit != expected:
            errors.append(
                f"complexity debt limit does not match active config: "
                f"{entry.identity.path} {entry.identity.declaration} "
                f"({entry.identity.linter})"
            )
    return errors


def validate_manifest_delta(
    manifest: DebtManifest,
    base_manifest: DebtManifest | None,
    changed_paths: set[str],
) -> tuple[list[str], bool]:
    if base_manifest is None:
        errors = [
            f"complexity bootstrap debt is outside the changed-file set: "
            f"{entry.identity.path} {entry.identity.declaration} "
            f"({entry.identity.linter})"
            for entry in manifest.entries
            if entry.identity.path not in changed_paths
        ]
        return errors, True
    errors: list[str] = []
    current = manifest.by_identity()
    baseline = base_manifest.by_identity()
    for identity in sorted(current.keys() - baseline.keys(), key=identity_sort_key):
        errors.append(
            f"complexity debt addition is frozen: {identity.path} "
            f"{identity.declaration} ({identity.linter})"
        )
    for identity in sorted(current.keys() & baseline.keys(), key=identity_sort_key):
        if current[identity].limit > baseline[identity].limit:
            errors.append(
                f"complexity debt limit widened: {identity.path} "
                f"{identity.declaration} ({identity.linter}) "
                f"{baseline[identity].limit} -> {current[identity].limit}"
            )
        if current[identity].observed > baseline[identity].observed:
            errors.append(
                f"complexity debt allowance widened: {identity.path} "
                f"{identity.declaration} ({identity.linter}) "
                f"{baseline[identity].observed} -> {current[identity].observed}"
            )
        if (
            current[identity].observed < baseline[identity].observed
            and identity.path not in changed_paths
        ):
            errors.append(
                f"complexity debt reduction requires a changed source file: "
                f"{identity.path} {identity.declaration} ({identity.linter})"
            )
        if (
            current[identity].owner != baseline[identity].owner
            or current[identity].debt != baseline[identity].debt
        ):
            errors.append(
                f"complexity debt ownership changed: {identity.path} "
                f"{identity.declaration} ({identity.linter})"
            )
    for identity in sorted(baseline.keys() - current.keys(), key=identity_sort_key):
        if identity.path not in changed_paths:
            errors.append(
                f"complexity debt removal requires a changed source file: "
                f"{identity.path} {identity.declaration} ({identity.linter})"
            )
    return errors, False


def render_manifest(
    findings: list[ComplexityFinding],
    tool_version: str,
    config_path: Path,
    owner: str,
    debt: str,
) -> str:
    entries = []
    for finding in sorted(findings, key=lambda item: identity_sort_key(item.identity)):
        entry = {
            "path": finding.identity.path,
            "declaration": finding.identity.declaration,
            "linter": finding.identity.linter,
        }
        if finding.identity.site:
            entry["site"] = finding.identity.site
        entry.update(
            {
                "observed": finding.observed,
                "limit": finding.limit,
                "owner": owner,
                "debt": debt,
            }
        )
        entries.append(entry)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "parser": IDENTITY_PARSER,
            "parser_version": detect_identity_parser_version(),
            "core_version": detect_identity_core_version(),
            "implementation_sha256": detect_identity_implementation_digest(),
        },
        "tool": {
            "name": "golangci-lint",
            "version": tool_version,
            "config_sha256": config_digest(config_path),
        },
        "policy": {
            "additions": "deny",
            "baseline": "committed-freeze-marker",
        },
        "entries": entries,
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
