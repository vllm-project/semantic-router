"""Shared helpers for the meta-routing learned-policy artifact contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACT_VERSION = "meta-routing-policy/v1alpha1"
FEATURE_SCHEMA_NAME = "feedback_record_flattened"
FEATURE_SCHEMA_VERSION = "v1"
SUPPORTED_PROVIDER_KINDS = {"calibrated_policy", "learned_policy"}


def load_policy_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_policy_artifact(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_artifact_version(artifact))

    provider = artifact.get("provider")
    errors.extend(_validate_provider(provider))

    feature_schema = artifact.get("feature_schema")
    errors.extend(_validate_feature_schema(feature_schema))

    evaluation = artifact.get("evaluation")
    errors.extend(_validate_evaluation(evaluation))

    policy = artifact.get("policy")
    errors.extend(_validate_policy(policy))

    rollout = artifact.get("rollout")
    if isinstance(rollout, dict) and isinstance(evaluation, dict):
        errors.extend(validate_rollout_gate(rollout, evaluation))
    return errors


def validate_rollout_gate(
    rollout: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    replay_records = int(evaluation.get("replay_records") or 0)
    min_replay_records = int(rollout.get("min_replay_records") or 0)
    if min_replay_records and replay_records < min_replay_records:
        errors.append(
            f"replay_records {replay_records} below minimum {min_replay_records}"
        )

    errors.extend(
        _validate_min_metric(
            "trigger_precision",
            rollout.get("min_trigger_precision"),
            evaluation.get("trigger_precision"),
        )
    )
    errors.extend(
        _validate_min_metric(
            "action_precision",
            rollout.get("min_action_precision"),
            evaluation.get("action_precision"),
        )
    )
    errors.extend(
        _validate_min_metric(
            "overturn_gain",
            rollout.get("min_overturn_gain"),
            evaluation.get("overturn_gain"),
        )
    )
    errors.extend(
        _validate_max_metric(
            "p95_latency_delta_ms",
            rollout.get("max_p95_latency_delta_ms"),
            evaluation.get("p95_latency_delta_ms"),
        )
    )
    return errors


def summarize_policy_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    provider = artifact.get("provider") or {}
    evaluation = artifact.get("evaluation") or {}
    rollout = artifact.get("rollout") or {}
    return {
        "artifact_id": artifact.get("artifact_id"),
        "version": artifact.get("version"),
        "provider": {
            "kind": provider.get("kind"),
            "name": provider.get("name"),
            "version": provider.get("version"),
        },
        "feature_schema": artifact.get("feature_schema") or {},
        "accepted": bool(evaluation.get("accepted")),
        "replay_records": evaluation.get("replay_records", 0),
        "rollout": rollout,
        "evaluation": evaluation,
        "has_trigger_policy": bool(
            (artifact.get("policy") or {}).get("trigger_policy")
        ),
        "allowed_action_count": len(
            (artifact.get("policy") or {}).get("allowed_actions") or []
        ),
    }


def _validate_artifact_version(artifact: dict[str, Any]) -> list[str]:
    if artifact.get("version") == ARTIFACT_VERSION:
        return []
    return [f"unsupported version: {artifact.get('version')!r}"]


def _validate_provider(provider: Any) -> list[str]:
    if not isinstance(provider, dict):
        return ["provider is required"]

    errors: list[str] = []
    if provider.get("kind") not in SUPPORTED_PROVIDER_KINDS:
        errors.append(f"unsupported provider kind: {provider.get('kind')!r}")
    if not provider.get("name"):
        errors.append("provider.name is required")
    if not provider.get("version"):
        errors.append("provider.version is required")
    return errors


def _validate_feature_schema(feature_schema: Any) -> list[str]:
    if not isinstance(feature_schema, dict):
        return ["feature_schema is required"]

    errors: list[str] = []
    if feature_schema.get("name") != FEATURE_SCHEMA_NAME:
        errors.append(
            f"unsupported feature_schema.name: {feature_schema.get('name')!r}"
        )
    if feature_schema.get("version") != FEATURE_SCHEMA_VERSION:
        errors.append(
            f"unsupported feature_schema.version: {feature_schema.get('version')!r}"
        )
    return errors


def _validate_evaluation(evaluation: Any) -> list[str]:
    if not isinstance(evaluation, dict):
        return ["evaluation is required"]
    if bool(evaluation.get("accepted")):
        return []
    return ["evaluation.accepted must be true"]


def _validate_policy(policy: Any) -> list[str]:
    if not isinstance(policy, dict):
        return ["policy is required"]
    if policy.get("trigger_policy") or policy.get("allowed_actions"):
        return []
    return ["policy must define trigger_policy or allowed_actions"]


def _validate_min_metric(
    name: str,
    minimum: Any,
    actual: Any,
) -> list[str]:
    if minimum is None:
        return []
    if actual is None:
        return [f"{name} is required when rollout gate is configured"]
    if float(actual) < float(minimum):
        return [f"{name} {float(actual):.4f} below minimum {float(minimum):.4f}"]
    return []


def _validate_max_metric(
    name: str,
    maximum: Any,
    actual: Any,
) -> list[str]:
    if maximum is None:
        return []
    if actual is None:
        return [f"{name} is required when rollout gate is configured"]
    if float(actual) > float(maximum):
        return [f"{name} {float(actual):.4f} above maximum {float(maximum):.4f}"]
    return []
