"""Load the repository's single ownership and validation registry."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "tools" / "agent" / "domains.yaml"
PROFILE_SELECTIONS = frozenset({"pr", "manual"})
REGISTRY_VERSION = 2


@lru_cache(maxsize=1)
def load_domain_registry() -> dict[str, Any]:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(registry, dict):
        raise TypeError(f"{REGISTRY_PATH} must contain a mapping")
    return registry


def _records(name: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    value = (registry or load_domain_registry()).get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"domain registry {name} must be a mapping")
    return value


def job_records(registry: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    return _records("jobs", registry)


def domain_records(
    registry: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return _records("domains", registry)


def profile_records(
    registry: dict[str, Any] | None = None,
    *,
    selection: str | None = None,
) -> dict[str, dict[str, Any]]:
    profiles = _records("profiles", registry)
    if selection is None:
        return profiles
    return {
        name: data
        for name, data in profiles.items()
        if data.get("selection") == selection
    }


def image_records(
    registry: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return _records("images", registry)


def path_matches(path: str, pattern: str) -> bool:
    return bool(_glob_regex(pattern).fullmatch(path))


@cache
def _glob_regex(pattern: str) -> re.Pattern[str]:
    pieces: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            pieces.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            pieces.append(".*")
            index += 2
        elif pattern[index] == "*":
            pieces.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            pieces.append("[^/]")
            index += 1
        else:
            pieces.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(pieces))


def any_matches(paths: Iterable[str], patterns: Iterable[str]) -> bool:
    return any(path_matches(path, pattern) for path in paths for pattern in patterns)


def matching_domains(
    paths: Iterable[str], registry: dict[str, Any] | None = None
) -> tuple[str, ...]:
    path_list = tuple(paths)
    return tuple(
        name
        for name, data in domain_records(registry).items()
        if any_matches(path_list, data.get("paths", []))
    )


def commands_for_domains(
    domains: Iterable[str],
    field: str,
    registry: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    records = domain_records(registry)
    commands: list[str] = []
    for name in domains:
        for raw_command in records.get(name, {}).get(field, []):
            normalized = str(raw_command)
            if normalized not in commands:
                commands.append(normalized)
    return tuple(commands)


def profile_paths(
    selection: str, registry: dict[str, Any] | None = None
) -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(str(path) for path in data.get("paths", []))
        for name, data in profile_records(registry, selection=selection).items()
    }


def registry_schema_errors(
    registry: dict[str, Any] | None = None,
) -> list[str]:
    data = registry or load_domain_registry()
    errors: list[str] = []
    if data.get("version") != REGISTRY_VERSION:
        errors.append(f"domain registry version must be {REGISTRY_VERSION}")

    jobs = job_records(data)
    outputs: list[str] = []
    for name, job in jobs.items():
        if not isinstance(job, dict) or not isinstance(job.get("workflow"), str):
            errors.append(f"CI job {name!r} must declare one workflow")
            continue
        output = job.get("output")
        if output:
            outputs.append(str(output))
    if len(outputs) != len(set(outputs)):
        errors.append("CI job output names must be unique")

    for name, domain in domain_records(data).items():
        if not isinstance(domain, dict):
            errors.append(f"domain {name!r} must be a mapping")
            continue
        for field in ("owner", "paths", "checks", "ci_jobs"):
            if field not in domain:
                errors.append(f"domain {name!r} is missing {field!r}")
        _validate_string_list(domain, "paths", f"domain {name!r}", errors)
        _validate_string_list(domain, "checks", f"domain {name!r}", errors)
        _validate_string_list(domain, "verify", f"domain {name!r}", errors)
        _validate_job_names(domain.get("ci_jobs", []), jobs, f"domain {name!r}", errors)
        escalation = domain.get("escalation", {})
        if escalation:
            if not isinstance(escalation, dict):
                errors.append(f"domain {name!r} escalation must be a mapping")
            else:
                _validate_string_list(
                    escalation, "paths", f"domain {name!r} escalation", errors
                )
                _validate_job_names(
                    escalation.get("jobs", []),
                    jobs,
                    f"domain {name!r} escalation",
                    errors,
                )

    for name, image in image_records(data).items():
        if not isinstance(image, dict):
            errors.append(f"image {name!r} must be a mapping")
            continue
        _validate_string_list(image, "pr_paths", f"image {name!r}", errors)
        _validate_string_list(image, "publish_paths", f"image {name!r}", errors)

    for name, profile in profile_records(data).items():
        if not isinstance(profile, dict):
            errors.append(f"E2E profile {name!r} must be a mapping")
            continue
        for field in ("owner", "coverage_role", "selection", "paths"):
            if not profile.get(field):
                errors.append(f"E2E profile {name!r} is missing {field!r}")
        if profile.get("selection") not in PROFILE_SELECTIONS:
            errors.append(
                f"E2E profile {name!r} has invalid selection "
                f"{profile.get('selection')!r}"
            )
        _validate_string_list(profile, "paths", f"E2E profile {name!r}", errors)
    return errors


def _validate_string_list(
    record: dict[str, Any], field: str, label: str, errors: list[str]
) -> None:
    value = record.get(field, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{label} {field} must be a string list")


def _validate_job_names(
    names: Any,
    jobs: dict[str, dict[str, Any]],
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(names, list):
        errors.append(f"{label} jobs must be a string list")
        return
    unknown = sorted(name for name in names if name not in jobs)
    if unknown:
        errors.append(f"{label} references unknown jobs: {', '.join(unknown)}")
