"""Resolve reviewed CI placement and observe its producer environment.

The CPU adapter establishes host placement, not model-device attestation.
Accelerator qualification needs a real adapter with device/model observations
and a hardware-specific required case inventory; labels cannot opt into it.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = ROOT / "tools/ci/execution_targets.yaml"
TARGET_FIELDS = (
    "execution_target",
    "execution_target_sha256",
    "execution_adapter",
    "runner",
    "runner_environment",
    "platform",
)
RUNNER_ENVIRONMENTS = frozenset({"github-hosted", "self-hosted"})


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def load_execution_targets() -> dict:
    return yaml.safe_load(TARGETS_PATH.read_text())


def resolve_execution_target(record: dict, *, registry: dict | None = None) -> dict:
    registry = load_execution_targets() if registry is None else registry
    if registry.get("version") != 1:
        raise ValueError("unsupported execution target registry version")
    name = record.get("execution_target")
    target = registry.get("targets", {}).get(name)
    if not isinstance(target, dict) or target.get("enabled") is not True:
        raise ValueError(f"execution target {name!r} is unknown or disabled")
    if target.get("adapter") != "cpu-v1":
        raise ValueError(f"execution target {name}: unsupported execution adapter")
    if not target.get("devices") or set(target["devices"]) - {"none", "cpu"}:
        raise ValueError(f"execution target {name}: CPU adapter cannot qualify GPUs")
    if not target.get("runtimes") or set(target["runtimes"]) - {
        "none",
        "model-runtime",
    }:
        raise ValueError(f"execution target {name}: unsupported runtime adapter")
    runner = target.get("runner")
    labels = [runner] if isinstance(runner, str) else runner
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or target.get("runner_environment") not in RUNNER_ENVIRONMENTS
        or target.get("platform") not in {"linux/amd64", "linux/arm64"}
    ):
        raise ValueError(f"execution target {name}: invalid runner placement")
    for field, supported in (
        ("executor", "executors"),
        ("runtime", "runtimes"),
        ("device", "devices"),
    ):
        if record.get(field) not in target.get(supported, []):
            raise ValueError(
                f"execution target {name}: unsupported {field} {record.get(field)!r}"
            )
    return {
        **record,
        "execution_target_sha256": digest(target),
        "execution_adapter": target["adapter"],
        "runner": runner,
        "runner_environment": target["runner_environment"],
        "platform": target["platform"],
    }


def target_errors(record: dict, *, registry: dict | None = None) -> list[str]:
    try:
        resolved = resolve_execution_target(record, registry=registry)
    except (ValueError, KeyError, TypeError) as error:
        return [str(error)]
    return [
        f"execution target {field} differs from registry"
        for field in TARGET_FIELDS
        if record.get(field) != resolved[field]
    ]


def observation_errors(
    record: dict, observation: dict, *, require_actions: bool = False
) -> list[str]:
    errors = target_errors(record)
    if not isinstance(observation, dict):
        return [*errors, "missing execution target observation"]
    for field in (
        "execution_target",
        "execution_target_sha256",
        "execution_adapter",
        "platform",
    ):
        if observation.get(field) != record.get(field):
            errors.append(f"observed {field} differs from planned target")
    environment = observation.get("runner_environment")
    if environment == "local":
        if require_actions:
            errors.append("local execution cannot qualify a GitHub Actions run")
    elif environment != record.get("runner_environment"):
        errors.append("observed runner environment differs from planned target")
    return errors


def execution_observation(
    record: dict, execution_platform: str, *, environ: dict | None = None
) -> dict:
    env = os.environ if environ is None else environ
    # Workflows bind this value directly from runner.environment. Local checks
    # remain possible, but their receipts are explicitly local and cannot pass
    # an Actions gate. Never infer runner trust from caller-supplied labels.
    actions = env.get("GITHUB_ACTIONS") == "true"
    environment = env.get("CI_RUNNER_ENVIRONMENT") if actions else "local"
    observation = {
        key: record.get(key)
        for key in (
            "execution_target",
            "execution_target_sha256",
            "execution_adapter",
        )
    }
    observation.update(platform=execution_platform, runner_environment=environment)
    errors = observation_errors(record, observation, require_actions=actions)
    if errors:
        raise ValueError("; ".join(errors))
    return observation
