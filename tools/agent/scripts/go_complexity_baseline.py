#!/usr/bin/env python3
"""Resolve the immutable Git-history baseline for Go complexity debt."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from go_complexity_manifest import (
    IDENTITY_PARSER,
    IDENTITY_SCHEMA_VERSION,
    MANIFEST_RELATIVE_PATH,
    ComplexityRatchetError,
    DebtManifest,
    config_digest_bytes,
    parse_manifest_text,
)


CONFIG_RELATIVE_PATH = Path("tools/linter/go/.golangci.agent.yml")
FREEZE_RELATIVE_PATH = Path("tools/linter/go/complexity-baseline.freeze.yaml")
FREEZE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrozenBaseline:
    commit: str
    manifest: DebtManifest
    config_bytes: bytes
    target_config_bytes: bytes
    bootstrap: bool


@dataclass(frozen=True)
class FreezeMarker:
    manifest_sha256: str
    config_sha256: str
    tool_version: str
    identity_schema_version: int
    identity_parser: str
    identity_parser_version: str
    identity_core_version: str
    identity_implementation_sha256: str


def _digest(raw_bytes: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"


def parse_freeze_marker(raw_bytes: bytes, source: str) -> FreezeMarker:
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ComplexityRatchetError(f"cannot parse {source}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "manifest_sha256",
        "config_sha256",
        "tool",
        "identity",
    }:
        raise ComplexityRatchetError(f"{source} has an invalid freeze contract")
    tool = raw["tool"]
    identity = raw["identity"]
    if raw["schema_version"] != FREEZE_SCHEMA_VERSION:
        raise ComplexityRatchetError(f"{source} has an unsupported schema version")
    if not isinstance(tool, dict) or set(tool) != {"name", "version"}:
        raise ComplexityRatchetError(f"{source} has an invalid tool contract")
    if tool["name"] != "golangci-lint" or not isinstance(tool["version"], str):
        raise ComplexityRatchetError(f"{source} has an unsupported lint tool")
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
    for key in ("manifest_sha256", "config_sha256"):
        if not isinstance(raw[key], str) or not raw[key].startswith("sha256:"):
            raise ComplexityRatchetError(f"{source} has an invalid {key}")
    return FreezeMarker(
        manifest_sha256=raw["manifest_sha256"],
        config_sha256=raw["config_sha256"],
        tool_version=tool["version"],
        identity_schema_version=identity["schema_version"],
        identity_parser=identity["parser"],
        identity_parser_version=identity["parser_version"],
        identity_core_version=identity["core_version"],
        identity_implementation_sha256=identity["implementation_sha256"],
    )


def render_freeze_marker(
    manifest_bytes: bytes,
    config_bytes: bytes,
    manifest: DebtManifest,
) -> str:
    payload = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "manifest_sha256": _digest(manifest_bytes),
        "config_sha256": config_digest_bytes(config_bytes),
        "tool": {"name": "golangci-lint", "version": manifest.tool_version},
        "identity": {
            "schema_version": manifest.identity_schema_version,
            "parser": manifest.identity_parser,
            "parser_version": manifest.identity_parser_version,
            "core_version": manifest.identity_core_version,
            "implementation_sha256": manifest.identity_implementation_sha256,
        },
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def _git(repo_root: Path, arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )


def _merge_base(repo_root: Path, base_ref: str) -> str:
    result = _git(repo_root, ["merge-base", "HEAD", base_ref])
    commit = result.stdout.decode().strip()
    if result.returncode != 0 or not commit:
        raise ComplexityRatchetError(
            f"cannot resolve complexity debt base revision from {base_ref!r}"
        )
    return commit


def resolve_target_tip(repo_root: Path, base_ref: str) -> str:
    result = _git(repo_root, ["rev-parse", "--verify", f"{base_ref}^{{commit}}"])
    commit = result.stdout.decode().strip()
    if result.returncode != 0 or not commit:
        raise ComplexityRatchetError(
            f"cannot resolve complexity debt target tip from {base_ref!r}"
        )
    return commit


def _git_blob(repo_root: Path, commit: str, path: Path) -> bytes | None:
    exists = _git(
        repo_root,
        ["cat-file", "-e", f"{commit}:{path.as_posix()}"],
    )
    if exists.returncode != 0:
        return None
    result = _git(repo_root, ["show", f"{commit}:{path.as_posix()}"])
    if result.returncode != 0:
        raise ComplexityRatchetError(
            f"cannot read {path} at complexity baseline {commit}"
        )
    return result.stdout


def _first_added_commit(repo_root: Path, merge_base: str, path: Path) -> str:
    result = _git(
        repo_root,
        [
            "log",
            "--reverse",
            "--format=%H",
            "--diff-filter=A",
            f"{merge_base}..HEAD",
            "--",
            path.as_posix(),
        ],
    )
    commits = result.stdout.decode().splitlines()
    if result.returncode != 0 or not commits:
        raise ComplexityRatchetError(
            f"complexity debt bootstrap must add {path} in a committed freeze"
        )
    return commits[0]


def _verify_freeze(
    marker: FreezeMarker,
    manifest_bytes: bytes,
    config_bytes: bytes,
    manifest: DebtManifest,
    source: str,
) -> None:
    expected = FreezeMarker(
        manifest_sha256=_digest(manifest_bytes),
        config_sha256=config_digest_bytes(config_bytes),
        tool_version=manifest.tool_version,
        identity_schema_version=manifest.identity_schema_version,
        identity_parser=manifest.identity_parser,
        identity_parser_version=manifest.identity_parser_version,
        identity_core_version=manifest.identity_core_version,
        identity_implementation_sha256=manifest.identity_implementation_sha256,
    )
    if marker != expected:
        raise ComplexityRatchetError(
            f"{source} does not commit to its baseline manifest/config contract"
        )


def _current_marker(repo_root: Path) -> bytes:
    try:
        return (repo_root / FREEZE_RELATIVE_PATH).read_bytes()
    except OSError as exc:
        raise ComplexityRatchetError(
            f"cannot read immutable complexity freeze marker: {exc}"
        ) from exc


def resolve_frozen_baseline(repo_root: Path, base_ref: str) -> FrozenBaseline:
    target_tip = resolve_target_tip(repo_root, base_ref)
    target_config_bytes = _git_blob(repo_root, target_tip, CONFIG_RELATIVE_PATH)
    if target_config_bytes is None:
        raise ComplexityRatchetError(
            f"complexity target config is missing at {target_tip}"
        )
    manifest_bytes = _git_blob(repo_root, target_tip, MANIFEST_RELATIVE_PATH)
    bootstrap = manifest_bytes is None
    if bootstrap:
        merge_base = _merge_base(repo_root, target_tip)
        baseline_commit = _first_added_commit(
            repo_root, merge_base, FREEZE_RELATIVE_PATH
        )
        manifest_commit = _first_added_commit(
            repo_root, merge_base, MANIFEST_RELATIVE_PATH
        )
        if manifest_commit != baseline_commit:
            raise ComplexityRatchetError(
                "complexity manifest and freeze marker must be introduced together"
            )
        manifest_bytes = _git_blob(repo_root, baseline_commit, MANIFEST_RELATIVE_PATH)
    else:
        baseline_commit = target_tip
    if manifest_bytes is None:
        raise ComplexityRatchetError(
            f"complexity baseline manifest is missing at {baseline_commit}"
        )
    config_bytes = _git_blob(repo_root, baseline_commit, CONFIG_RELATIVE_PATH)
    if config_bytes is None:
        raise ComplexityRatchetError(
            f"complexity baseline config is missing at {baseline_commit}"
        )
    marker_bytes = _git_blob(repo_root, baseline_commit, FREEZE_RELATIVE_PATH)
    if marker_bytes is None:
        raise ComplexityRatchetError(
            f"complexity freeze marker is missing at {baseline_commit}"
        )
    if _current_marker(repo_root) != marker_bytes:
        raise ComplexityRatchetError(
            "complexity freeze marker changed after its canonical commit"
        )
    manifest = parse_manifest_text(
        manifest_bytes.decode("utf-8"),
        f"{baseline_commit}:{MANIFEST_RELATIVE_PATH}",
    )
    marker = parse_freeze_marker(
        marker_bytes, f"{baseline_commit}:{FREEZE_RELATIVE_PATH}"
    )
    if bootstrap:
        _verify_freeze(
            marker,
            manifest_bytes,
            config_bytes,
            manifest,
            f"{baseline_commit}:{FREEZE_RELATIVE_PATH}",
        )
    return FrozenBaseline(
        commit=baseline_commit,
        manifest=manifest,
        config_bytes=config_bytes,
        target_config_bytes=target_config_bytes,
        bootstrap=bootstrap,
    )
