"""Catalog-aware validation for the public v0.5 model configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from cli.config_contract import iter_routing_profiles
from cli.model_catalog import DEFAULT_CHANNEL, _load_catalog_document
from cli.models import UserConfig
from cli.validation_error import ValidationError


@lru_cache(maxsize=1)
def _catalog_index() -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    frozenset[str],
    dict[str, dict[str, Any]],
]:
    _, document = _load_catalog_document(DEFAULT_CHANNEL)
    models = document.get("models")
    return (
        _ids(document.get("providers")),
        _ids(models, kind="physical"),
        _ids(models, kind="virtual"),
        _ids(document.get("reasoning_families")),
        {
            model["id"]: model
            for model in models or []
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        },
    )


def _ids(values: Any, *, kind: str | None = None) -> frozenset[str]:
    if not isinstance(values, list):
        return frozenset()
    return frozenset(
        value["id"]
        for value in values
        if isinstance(value, dict)
        and isinstance(value.get("id"), str)
        and (kind is None or value.get("kind") == kind)
    )


def validate_model_references(config: UserConfig) -> list[ValidationError]:
    """Validate aliases, canonical card identities, providers, and LoRAs."""

    (
        provider_ids,
        physical_models,
        virtual_models,
        reasoning_families,
        built_in_cards,
    ) = _catalog_index()
    built_in_models = physical_models | virtual_models
    aliases = {model.name for model in config.providers.models}
    cards = _model_cards(config)
    catalogs_by_alias = {
        model.name: model.catalog or model.name for model in config.providers.models
    }
    lora_aliases = {
        adapter.name for card in cards.values() for adapter in (card.loras or [])
    }
    errors = _blank_identity_errors(config)
    errors.extend(_duplicate_identity_errors(config, aliases, cards))
    errors.extend(
        _provider_model_errors(
            config,
            cards,
            catalogs_by_alias,
            provider_ids,
            built_in_models,
            virtual_models,
            reasoning_families,
        )
    )
    errors.extend(_built_in_card_claim_errors(config, built_in_cards))
    errors.extend(_decision_errors(config, aliases, cards, catalogs_by_alias))
    errors.extend(_default_model_errors(config, aliases, cards, lora_aliases))
    return errors


def _blank_identity_errors(config: UserConfig) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for index, model in enumerate(config.providers.models):
        if not model.name.strip():
            errors.append(
                ValidationError(
                    "Provider model name cannot be empty",
                    field=f"providers.models[{index}].name",
                )
            )
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
    declared_by: dict[str, str] = {}
    for model in config.providers.models:
        if not model.declares_facts():
            continue
        card = model.catalog or model.name
        if card in declared_by:
            errors.append(
                ValidationError(
                    f"the facts of model card '{card}' are already declared by "
                    f"providers.models[{declared_by[card]}]",
                    field=f"providers.models[{model.name}]",
                )
            )
        declared_by.setdefault(card, model.name)
    return errors


def _provider_model_errors(
    config: UserConfig,
    cards: dict[str, Any],
    catalogs_by_alias: dict[str, str],
    provider_ids: frozenset[str],
    built_in_models: frozenset[str],
    virtual_models: frozenset[str],
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
        if (
            config.listeners
            and not model.backend_refs
            and not model.deployment
            and catalog not in virtual_models
            and not model.states_only_facts()
        ):
            errors.append(
                ValidationError(
                    f"Provider model '{model.name}' is a physical model used by a "
                    "router-owned listener and must define backend_refs with an "
                    "explicit Provider ID",
                    field=f"providers.models.{model.name}.backend_refs",
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


def _built_in_card_claim_errors(
    config: UserConfig, built_in_cards: dict[str, dict[str, Any]]
) -> list[ValidationError]:
    """Match the Router catalog compiler's built-in claim widening rule.

    A provider model's facts cannot carry verification.status=reproduced, so
    additions to a catalog-backed card must be rejected before serve starts.
    Custom cards, even ones sharing a built-in name, remain operator-owned.
    """

    catalog_backed = {
        model.catalog for model in config.providers.models if model.catalog
    }
    errors: list[ValidationError] = []
    for card in config.providers.models:
        if not card.declares_facts() or card.catalog not in catalog_backed:
            continue
        built_in = built_in_cards.get(card.catalog)
        if built_in is None:
            continue  # The unknown catalog identity is reported by provider validation.
        if card.capabilities is not None:
            added = set(card.capabilities) - set(built_in.get("capabilities") or [])
            if added:
                errors.append(
                    ValidationError(
                        "capabilities widen a built-in claim without "
                        "verification.status=reproduced: " + ", ".join(sorted(added)),
                        field=f"providers.models[{card.name}].capabilities",
                    )
                )
        if card.modalities is not None:
            built_in_modalities = built_in.get("modalities") or {}
            added = {
                modality
                for direction in ("input", "output")
                for modality in set(card.modalities.get(direction) or [])
                - set(built_in_modalities.get(direction) or [])
            }
            if added:
                errors.append(
                    ValidationError(
                        "modalities widen a built-in claim without "
                        "verification.status=reproduced: " + ", ".join(sorted(added)),
                        field=f"providers.models[{card.name}].modalities",
                    )
                )
    return errors


def _decision_errors(
    config: UserConfig,
    aliases: set[str],
    cards: dict[str, Any],
    catalogs_by_alias: dict[str, str],
) -> list[ValidationError]:
    # Match the canonical Go contract: routing-only metadata can describe
    # external-gateway targets before any Router-owned provider alias exists.
    if not aliases:
        return []
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
    config: UserConfig,
    aliases: set[str],
    cards: dict[str, Any],
    lora_aliases: set[str],
) -> list[ValidationError]:
    default_model = config.providers.defaults.model
    if (
        default_model
        and default_model not in aliases
        and default_model not in lora_aliases
        and (aliases or default_model not in cards)
    ):
        return [
            ValidationError(
                f"Default model '{default_model}' not found in providers.models or model-card LoRAs",
                field="providers.defaults.model",
            )
        ]
    return []


def _all_decisions(config: UserConfig):
    for name, recipe in iter_routing_profiles(config):
        yield from (
            (f"recipes.{name}.decisions", decision) for decision in recipe.decisions
        )


def _model_cards(config: UserConfig) -> dict[str, Any]:
    """The provider models that state facts, by the card they describe: the
    entry's catalog identity, else its name."""
    return {
        model.catalog or model.name: model
        for model in config.providers.models
        if model.declares_facts()
    }
