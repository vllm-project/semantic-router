"""Bind fitted actions to explicit deployment stage and provider aliases."""

from __future__ import annotations

import copy

from .metrics import FEATURE_NAMES


def bind_policy(policy: dict, bindings: dict[str, dict[str, str]]) -> dict:
    """Rename actions without removing, adding or repointing measured candidates."""
    if (
        policy.get("schema_version") != "systemone-policy/v1"
        or policy.get("feature_names") != FEATURE_NAMES
    ):
        raise ValueError("unsupported policy feature contract")
    actions = policy.get("actions", {})
    if not actions or set(bindings) != set(actions):
        raise ValueError(
            "bind every measured action; candidate subsets require a separate experiment"
        )
    for value in bindings.values():
        if set(value) != {"stage", "model"} or any(
            not isinstance(v, str) or not v.strip() for v in value.values()
        ):
            raise ValueError(
                "each binding requires explicit nonempty stage and model aliases"
            )
    if len({v["stage"] for v in bindings.values()}) != len(bindings):
        raise ValueError("stage aliases must be unique")
    if set(policy["heads"]) != set(actions) or any(
        set(heads) != set(actions) - {source}
        for source, heads in policy["heads"].items()
    ):
        raise ValueError("policy heads do not cover the measured native actions")
    result = copy.deepcopy(policy)
    result["actions"] = {
        bindings[name]["stage"]: {**value, "model": bindings[name]["model"]}
        for name, value in result["actions"].items()
    }
    result["heads"] = {
        bindings[source]["stage"]: {
            bindings[target]["stage"]: head for target, head in heads.items()
        }
        for source, heads in result["heads"].items()
    }
    result["training"]["deployment_bindings"] = copy.deepcopy(bindings)
    return result
