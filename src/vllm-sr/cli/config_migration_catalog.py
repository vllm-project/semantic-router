"""Migration helpers for the catalog-backed v0.3 model contract."""

from copy import deepcopy
from typing import Any


def migrate_v03_catalog_contract(canonical: dict[str, Any]) -> None:
    """Migrate legacy model metadata into the compact catalog-backed surface."""

    providers = _as_dict(canonical.get("providers"))
    defaults = _as_dict(providers.get("defaults"))
    _rename_if_missing(defaults, "default_model", "model")
    _rename_if_missing(defaults, "default_reasoning_effort", "reasoning_effort")
    defaults.pop("reasoning_families", None)
    if defaults:
        providers["defaults"] = defaults

    catalog_by_alias = _migrate_provider_models(providers)
    routing = _as_dict(canonical.get("routing"))
    _migrate_model_cards(routing, catalog_by_alias)

    canonical["providers"] = providers
    canonical["routing"] = routing


def _rename_if_missing(target: dict[str, Any], old: str, new: str) -> None:
    if new not in target and old in target:
        target[new] = target.pop(old)
        return
    target.pop(old, None)


def _migrate_provider_models(providers: dict[str, Any]) -> dict[str, str]:
    catalog_by_alias: dict[str, str] = {}
    provider_models = providers.get("models")
    if not isinstance(provider_models, list):
        return catalog_by_alias

    for model in provider_models:
        if not isinstance(model, dict):
            continue
        alias = str(model.get("name") or "").strip()
        catalog = str(model.get("catalog") or alias).strip()
        if alias:
            catalog_by_alias[alias] = catalog
        family = model.pop("reasoning_family", None)
        if family and "reasoning" not in model:
            model["reasoning"] = {"family": family}
        _migrate_backend_refs(model)
    return catalog_by_alias


def _migrate_backend_refs(model: dict[str, Any]) -> None:
    backend_refs = model.get("backend_refs")
    if not isinstance(backend_refs, list):
        return
    for backend in backend_refs:
        if not isinstance(backend, dict):
            continue
        legacy_type = backend.pop("type", None)
        if not backend.get("provider") and legacy_type:
            backend["provider"] = legacy_type
        if not backend.get("provider"):
            backend["provider"] = "vllm"


def _migrate_model_cards(
    routing: dict[str, Any], catalog_by_alias: dict[str, str]
) -> None:
    migrated_cards: list[dict[str, Any]] = []
    for card in _clone_list(routing.get("modelCards")):
        if not isinstance(card, dict):
            continue
        alias = str(card.get("name") or "").strip()
        if alias in catalog_by_alias:
            card["name"] = catalog_by_alias[alias]
        _migrate_quality_score(card)
        if set(card) != {"name"}:
            migrated_cards.append(card)

    if migrated_cards:
        routing["modelCards"] = migrated_cards
    else:
        routing.pop("modelCards", None)


def _migrate_quality_score(card: dict[str, Any]) -> None:
    quality = card.pop("quality_score", None)
    if not isinstance(quality, (int, float)):
        return
    evaluations = _clone_list(card.get("evaluations"))
    evaluations.append(
        {
            "benchmark": "vllm-sr/operator-rating@1.0.0",
            "metrics": {"score": float(quality)},
        }
    )
    card["evaluations"] = evaluations


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _clone_list(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []
