"""Backends for the models a decision calls in process.

The Router makes a Looper algorithm's model calls, a prompt helper's and
context recovery's itself, through each model's providers.models[].backend_refs,
in either gateway mode. It refuses at load a decision that calls a model with
no backend (pkg/config/validator_looper_backends.go); this mirrors that check
with the same message.
"""

import json
from collections.abc import Iterator
from typing import Any

from cli.config_schema import routing_surface_catalog
from cli.models import UserConfig
from cli.validation_error import ValidationError

LOOPER_ALGORITHM_TYPES = frozenset(
    surface["type"]
    for surface in routing_surface_catalog()["algorithms"]
    if surface.get("execution") == "looper"
)


def validate_inprocess_model_backends(config: UserConfig) -> list[ValidationError]:
    served = _served_models(config)
    errors: list[ValidationError] = []
    for field_prefix, routing in _routing_profiles(config):
        for decision in routing.decisions:
            missing = [
                model for model in _called_models(decision) if model not in served
            ]
            for model in dict.fromkeys(missing):
                errors.append(
                    ValidationError(
                        f"decision {_quote(decision.name)}: the Router calls model "
                        f"{_quote(model)} in process, but it has no backend; "
                        "give it providers.models[].backend_refs",
                        field=f"{field_prefix}.{decision.name}",
                    )
                )
    return errors


def _routing_profiles(config: UserConfig) -> list[tuple[str, Any]]:
    profiles = [("decisions", config.routing)]
    profiles.extend(
        (f"recipes.{recipe.name}.decisions", recipe.routing)
        for recipe in config.recipes
    )
    return profiles


def _served_models(config: UserConfig) -> set[str]:
    """Models with a backend, and the LoRA adapters their model cards declare."""
    served = {model.name for model in config.providers.models if model.backend_refs}
    for card in config.routing.model_cards:
        if card.name in served:
            served.update(lora.name for lora in card.loras or [])
    return served


def _called_models(decision: Any) -> Iterator[str]:
    """The models a decision calls in process, beyond the one it routes to."""
    refs = [ref.model for ref in decision.modelRefs]
    algorithm = decision.algorithm
    if algorithm is not None and algorithm.type in LOOPER_ALGORITHM_TYPES:
        yield from refs
        fusion = algorithm.fusion
        if fusion is not None:
            yield from fusion.analysis_models or []
            if fusion.model:
                yield fusion.model
        planner = algorithm.workflows.planner if algorithm.workflows else None
        if planner is not None and (planner.model or "").strip():
            yield planner.model.strip()
    elif algorithm is not None and algorithm.prompt is not None:
        yield algorithm.prompt.model
    if _recovers_context(decision):
        yield from refs


def _recovers_context(decision: Any) -> bool:
    for plugin in decision.plugins or []:
        if str(getattr(plugin.type, "value", plugin.type)) != "context_compression":
            continue
        recovery = (plugin.configuration or {}).get("recovery")
        if isinstance(recovery, dict) and recovery.get("enabled"):
            return True
    return False


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
