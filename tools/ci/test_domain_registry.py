"""Load and project the repository test-domain registry."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "tools" / "agent" / "test-domain-registry.yaml"
DOMAIN_FIELDS = frozenset(
    {
        "owner",
        "contract",
        "layer",
        "selector",
        "pr_job",
        "workflow",
        "cadence",
        "resource_class",
        "local",
        "image_receipts",
        "reporting",
    }
)
PROFILE_FIELDS = frozenset({"owner", "coverage_role", "selection", "paths"})
RESOURCE_CLASSES = frozenset({"light", "medium", "heavy"})
PROFILE_SELECTIONS = frozenset({"pr", "manual"})


@lru_cache(maxsize=1)
def load_test_domain_registry() -> dict[str, Any]:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(registry, dict):
        raise TypeError(f"{REGISTRY_PATH} must contain a mapping")
    return registry


def domain_records(
    registry: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    value = (registry or load_test_domain_registry()).get("domains", {})
    if not isinstance(value, dict):
        raise TypeError("test-domain registry domains must be a mapping")
    return value


def profile_records(
    registry: dict[str, Any] | None = None,
    *,
    selection: str | None = None,
) -> dict[str, dict[str, Any]]:
    value = (registry or load_test_domain_registry()).get("profiles", {})
    if not isinstance(value, dict):
        raise TypeError("test-domain registry profiles must be a mapping")
    if selection is None:
        return value
    return {
        name: data
        for name, data in value.items()
        if isinstance(data, dict) and data.get("selection") == selection
    }


def domain_paths(name: str, registry: dict[str, Any] | None = None) -> tuple[str, ...]:
    raw_paths = domain_records(registry).get(name, {}).get("paths", [])
    return tuple(str(path) for path in raw_paths)


def profile_paths(
    selection: str, registry: dict[str, Any] | None = None
) -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(str(path) for path in data.get("paths", []))
        for name, data in profile_records(registry, selection=selection).items()
    }


def suite_domains(
    registry: dict[str, Any] | None = None,
) -> dict[str, tuple[str, dict[str, Any]]]:
    return {
        str(data["suite"]): (name, data)
        for name, data in domain_records(registry).items()
        if isinstance(data, dict) and data.get("suite")
    }


def pr_job_order(registry: dict[str, Any] | None = None) -> tuple[str, ...]:
    value = (registry or load_test_domain_registry()).get("pr_job_order", [])
    if not isinstance(value, list):
        raise TypeError("test-domain registry pr_job_order must be a list")
    return tuple(str(job) for job in value)


def local_full_ci_paths(
    registry: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    value = (registry or load_test_domain_registry()).get("local_full_ci_paths", [])
    if not isinstance(value, list):
        raise TypeError("test-domain registry local_full_ci_paths must be a list")
    return tuple(str(path) for path in value)


def registry_schema_errors(
    registry: dict[str, Any] | None = None,
) -> list[str]:
    data = registry or load_test_domain_registry()
    errors: list[str] = []
    if data.get("version") != 1:
        errors.append("test-domain registry version must be 1")
    _validate_domains(domain_records(data), errors)
    _validate_profiles(profile_records(data), errors)
    contexts = data.get("compatibility_contexts")
    if not isinstance(contexts, list) or not contexts:
        errors.append("test-domain registry compatibility_contexts must be non-empty")
    order = pr_job_order(data)
    expected_jobs = {domain["pr_job"] for domain in domain_records(data).values()}
    if len(order) != len(set(order)) or set(order) != expected_jobs:
        errors.append(
            "test-domain registry pr_job_order must contain every unique PR job"
        )
    if not local_full_ci_paths(data):
        errors.append("test-domain registry local_full_ci_paths must be non-empty")
    return errors


def _validate_domains(domains: dict[str, dict[str, Any]], errors: list[str]) -> None:
    for name, data in domains.items():
        if not isinstance(data, dict):
            errors.append(f"test domain {name!r} must be a mapping")
            continue
        missing = sorted(field for field in DOMAIN_FIELDS if field not in data)
        if missing:
            errors.append(
                f"test domain {name!r} is missing fields: {', '.join(missing)}"
            )
        if data.get("resource_class") not in RESOURCE_CLASSES:
            errors.append(
                f"test domain {name!r} has invalid resource_class "
                f"{data.get('resource_class')!r}"
            )
        local = data.get("local")
        if not isinstance(local, dict) or not all(
            isinstance(local.get(key), list) for key in ("fast", "feature")
        ):
            errors.append(
                f"test domain {name!r} local commands require fast/feature lists"
            )


def _validate_profiles(profiles: dict[str, dict[str, Any]], errors: list[str]) -> None:
    for name, data in profiles.items():
        if not isinstance(data, dict):
            errors.append(f"test profile {name!r} must be a mapping")
            continue
        missing = sorted(field for field in PROFILE_FIELDS if not data.get(field))
        if missing:
            errors.append(
                f"test profile {name!r} is missing fields: {', '.join(missing)}"
            )
        if data.get("selection") not in PROFILE_SELECTIONS:
            errors.append(
                f"test profile {name!r} has invalid selection {data.get('selection')!r}"
            )
