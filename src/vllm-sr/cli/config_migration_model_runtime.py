"""Explicit migration of role-selected model backends to named contracts."""

from __future__ import annotations

from typing import Any


def migrate_prompt_guard_backend(canonical: dict[str, Any]) -> None:
    """Replace the retired prompt_guard.protocol without guessing a service."""

    catalog = canonical.get("global", {}).get("model_catalog", {})
    guard = catalog.get("modules", {}).get("prompt_guard", {})
    if not isinstance(guard, dict) or "protocol" not in guard:
        return
    protocol = guard["protocol"]
    if not protocol:
        guard.pop("protocol")
        return
    field = "global.model_catalog.modules.prompt_guard"
    if protocol not in {"http_chat", "http_classify"}:
        raise ValueError(f"{field}.protocol: unsupported legacy protocol {protocol!r}")
    if guard.get("backend") or guard.get("variant"):
        raise ValueError(f"{field}.protocol conflicts with backend or variant")

    external = catalog.get("external", [])
    candidates = [
        model
        for model in external
        if isinstance(model, dict) and model.get("model_role") == "guardrail"
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{field}.protocol migration requires exactly one guardrail external "
            "model; name the intended service and configure backend explicitly"
        )
    model = candidates[0]
    name = model.get("name") or "guardrail_classifier"
    if any(
        other is not model and other.get("name") == name
        for other in external
        if isinstance(other, dict)
    ):
        raise ValueError(
            f"{field}.protocol migration: external model name {name!r} conflicts; "
            "assign a unique name to the guardrail service"
        )
    model["name"] = name
    guard["backend"] = {
        "protocol": protocol,
        "contract": (
            "label_decision.v1" if protocol == "http_chat" else "label_distribution.v1"
        ),
        "model": name,
    }
    guard.pop("protocol")
