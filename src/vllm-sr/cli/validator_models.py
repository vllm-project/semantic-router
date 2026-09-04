"""Catalog-aware validation for the public v0.3 model configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from cli.model_catalog import DEFAULT_CHANNEL, _load_catalog_document
from cli.models import UserConfig
from cli.validation_error import ValidationError


@lru_cache(maxsize=1)
def _catalog_ids() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    _, document = _load_catalog_document(DEFAULT_CHANNEL)
    return (
        _ids(document.get("providers")),
        _ids(document.get("models")),
        _ids(document.get("reasoning_families")),
    )


def _ids(values: Any) -> frozenset[str]:
    if not isinstance(values, list):
        return frozenset()
    return frozenset(
        value["id"]
        for value in values
        if isinstance(value, dict) and isinstance(value.get("id"), str)
    )


def validate_model_references(config: UserConfig) -> list[ValidationError]:
    """Validate aliases, canonical card identities, providers, and LoRAs."""

    provider_ids, built_in_models, reasoning_families = _catalog_ids()
    aliases = {model.name for model in config.providers.models}
    cards = {card.name: card for card in config.routing.model_cards}
    catalogs_by_alias = {
        model.name: model.catalog or model.name for model in config.providers.models
    }
    lora_aliases = {
        adapter.name
        for card in config.routing.model_cards
        for adapter in (card.loras or [])
    }
    errors = _duplicate_identity_errors(config, aliases, cards)
    errors.extend(
        _provider_model_errors(
            config,
            cards,
            catalogs_by_alias,
            provider_ids,
            built_in_models,
            reasoning_families,
        )
    )
    errors.extend(
        _model_card_errors(
            cards, set(catalogs_by_alias.values()), aliases, lora_aliases
        )
    )
    errors.extend(_decision_errors(config, aliases, cards, catalogs_by_alias))
    errors.extend(_default_model_errors(config, aliases, lora_aliases))
    return errors


def _duplicate_identity_errors(
    config: UserConfig, aliases: set[str], cards: dict[str, Any]
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if len(aliases) != len(config.providers.models):
        errors.append(
            ValidationError(
                "providers.models[].name values must be unique",
                field="providers.models",
            )
        )
    if len(cards) != len(config.routing.model_cards):
        errors.append(
            ValidationError(
                "routing.modelCards[].name values must be unique",
                field="routing.modelCards",
            )
        )
    return errors


def _provider_model_errors(
    config: UserConfig,
    cards: dict[str, Any],
    catalogs_by_alias: dict[str, str],
    provider_ids: frozenset[str],
    built_in_models: frozenset[str],
    reasoning_families: frozenset[str],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for model in config.providers.models:
        catalog = catalogs_by_alias[model.name]
        if model.catalog and catalog not in built_in_models:
            errors.append(
                ValidationError(
                    f"Provider model '{model.name}' references unknown built-in catalog model '{catalog}'",
                    field=f"providers.models.{model.name}.catalog",
                )
            )
        if model.catalog and model.name in cards and model.name != catalog:
            errors.append(
                ValidationError(
                    f"Model card override for '{model.name}' must use canonical catalog name '{catalog}'",
                    field=f"routing.modelCards.{model.name}.name",
                )
            )
        if model.catalog and model.reasoning:
            errors.append(
                ValidationError(
                    f"Provider model '{model.name}' inherits reasoning from catalog model '{catalog}'",
                    field=f"providers.models.{model.name}.reasoning",
                )
            )
        if (
            model.reasoning
            and model.reasoning.family
            and model.reasoning.family not in reasoning_families
        ):
            errors.append(
                ValidationError(
                    f"Provider model '{model.name}' references unknown reasoning family '{model.reasoning.family}'",
                    field=f"providers.models.{model.name}.reasoning.family",
                )
            )
        for index, backend in enumerate(model.backend_refs):
            if backend.provider not in provider_ids:
                errors.append(
                    ValidationError(
                        f"Provider model '{model.name}' backend references unknown provider ID '{backend.provider}'",
                        field=f"providers.models.{model.name}.backend_refs[{index}].provider",
                    )
                )
    return errors


def _model_card_errors(
    cards: dict[str, Any],
    referenced_cards: set[str],
    aliases: set[str],
    lora_aliases: set[str],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for card_name in cards:
        if (
            card_name not in referenced_cards
            and card_name not in aliases
            and card_name not in lora_aliases
        ):
            errors.append(
                ValidationError(
                    f"Model card '{card_name}' does not match a providers.models catalog identity",
                    field=f"routing.modelCards.{card_name}.name",
                )
            )
    return errors


def _decision_errors(
    config: UserConfig,
    aliases: set[str],
    cards: dict[str, Any],
    catalogs_by_alias: dict[str, str],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for field_prefix, decision in _all_decisions(config):
        for model_ref in decision.modelRefs:
            if model_ref.model not in aliases:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references unknown model '{model_ref.model}'",
                        field=f"{field_prefix}.{decision.name}.modelRefs",
                    )
                )
                continue
            if not model_ref.lora_name:
                continue
            card = cards.get(catalogs_by_alias[model_ref.model])
            declared_loras = {
                adapter.name for adapter in ((card.loras if card else None) or [])
            }
            if model_ref.lora_name not in declared_loras:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references unknown LoRA '{model_ref.lora_name}' for model '{model_ref.model}'",
                        field=f"{field_prefix}.{decision.name}.modelRefs",
                    )
                )
    return errors


def _default_model_errors(
    config: UserConfig, aliases: set[str], lora_aliases: set[str]
) -> list[ValidationError]:
    default_model = config.providers.defaults.model
    if (
        default_model
        and default_model not in aliases
        and default_model not in lora_aliases
    ):
        return [
            ValidationError(
                f"Default model '{default_model}' not found in providers.models or model-card LoRAs",
                field="providers.defaults.model",
            )
        ]
    return []


def _all_decisions(config: UserConfig):
    yield from (
        ("routing.decisions", decision) for decision in config.routing.decisions
    )
    for recipe in config.recipes:
        yield from (
            (f"recipes.{recipe.name}.decisions", decision)
            for decision in recipe.routing.decisions
        )
