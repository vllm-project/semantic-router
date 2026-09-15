"""Validate decision reasoning controls against the canonical model families."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from cli.config_contract import iter_routing_profiles
from cli.model_catalog import DEFAULT_CHANNEL, _load_catalog_document
from cli.models import UserConfig
from cli.validation_error import ValidationError


@lru_cache(maxsize=1)
def _reasoning_catalog():
    _, catalog = _load_catalog_document(DEFAULT_CHANNEL)
    return (
        {item["id"]: item for item in catalog.get("reasoning_families", [])},
        {item["id"]: item for item in catalog.get("models", [])},
    )


def validate_reasoning_controls(config: UserConfig) -> list[ValidationError]:
    """Match Go's model-ref reasoning checks in default and named recipes.

    Family definitions come from the installed catalog, including catalog-bound
    physical models. Custom inline families keep their authored mode contract.
    """

    families, cards = _reasoning_catalog()
    model_families: dict[str, dict[str, Any] | None] = {}
    for model in config.providers.models:
        if model.reasoning is not None:
            family = (
                families.get(model.reasoning.family)
                if model.reasoning.family
                else model.reasoning.model_dump(exclude_none=True)
            )
        else:
            card = cards.get(model.catalog or model.name, {})
            family = families.get(card.get("reasoning_family"))
        model_families[model.name] = family

    errors = []
    for name, routing in iter_routing_profiles(config):
        prefix = "routing" if name == "default" else f"recipes.{name}.routing"
        for decision in routing.decisions:
            for index, model_ref in enumerate(decision.modelRefs):
                message = _model_ref_error(
                    model_ref,
                    model_families.get(model_ref.model),
                    has_registry=bool(config.providers.models),
                )
                if message:
                    errors.append(
                        ValidationError(
                            message,
                            field=f"{prefix}.decisions.{decision.name}.modelRefs[{index}]",
                        )
                    )
    return errors


def _model_ref_error(
    ref: Any, family: dict[str, Any] | None, *, has_registry: bool
) -> str | None:
    effort = ref.reasoning_effort or ""
    mode = ref.reasoning_mode or ""
    if effort != effort.strip():
        return "reasoning_effort must not contain surrounding whitespace"
    if family is None:
        if mode and has_registry:
            return "reasoning_mode requires a reasoning family"
        return None

    # The Router defaults an omitted pointer to false before validation.
    enabled = ref.use_reasoning is True
    modes = family.get("modes") or []
    if mode and mode not in modes:
        return (
            f"reasoning_mode '{mode}' is not supported by the model's reasoning family"
        )
    if mode and enabled != (mode != "disabled"):
        return f"use_reasoning conflicts with reasoning_mode '{mode}'; set use_reasoning explicitly"
    if not enabled:
        if not (
            family.get("activation_parameter")
            or family.get("disabled")
            or "disabled" in modes
        ):
            return "use_reasoning=false is not supported by this always-on reasoning family"
        if effort:
            return "reasoning_effort cannot be set while reasoning is disabled"
        return None
    if effort and family.get("type") in {"reasoning_mode", "chat_template_kwargs"}:
        return "reasoning_effort cannot be used with this mode-only reasoning family"
    if effort and effort not in family.get("levels", []):
        return f"reasoning_effort '{effort}' is not supported by the model's reasoning family"
    return None
