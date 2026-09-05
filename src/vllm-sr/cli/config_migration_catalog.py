"""Migration helpers for the catalog-backed v0.3 model contract."""

from copy import deepcopy
from typing import Any


def migrate_v03_catalog_contract(
    canonical: dict[str, Any],
    *,
    router_owns_transport: bool | None = None,
) -> None:
    """Migrate legacy model metadata into the compact catalog-backed surface."""

    if router_owns_transport is None:
        router_owns_transport = bool(canonical.get("listeners"))

    providers = _as_dict(canonical.get("providers"))
    defaults = _as_dict(providers.get("defaults"))
    _rename_if_missing(defaults, "default_model", "model")
    _rename_if_missing(defaults, "default_reasoning_effort", "reasoning_effort")
    reasoning_families = _as_dict(defaults.pop("reasoning_families", None))
    if defaults:
        providers["defaults"] = defaults
    else:
        providers.pop("defaults", None)

    catalog_by_alias = _migrate_provider_models(
        providers,
        reasoning_families,
        router_owns_transport=router_owns_transport,
    )
    routing = _as_dict(canonical.get("routing"))
    _migrate_model_cards(routing, catalog_by_alias)

    canonical["providers"] = providers
    canonical["routing"] = routing


def _rename_if_missing(target: dict[str, Any], old: str, new: str) -> None:
    if new not in target and old in target:
        target[new] = target.pop(old)
        return
    target.pop(old, None)


def _migrate_provider_models(
    providers: dict[str, Any],
    reasoning_families: dict[str, Any],
    *,
    router_owns_transport: bool,
) -> dict[str, str]:
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
            definition = reasoning_families.get(str(family))
            model["reasoning"] = (
                deepcopy(definition)
                if isinstance(definition, dict)
                else {"family": family}
            )
        _migrate_backend_refs(model, router_owns_transport=router_owns_transport)
    return catalog_by_alias


def _migrate_backend_refs(
    model: dict[str, Any], *, router_owns_transport: bool
) -> None:
    backend_refs = model.get("backend_refs")
    if not isinstance(backend_refs, list) or not backend_refs:
        # This is the one historical implicit endpoint that v0.3 exposed.  It
        # belongs in explicit migration only: api_format remains a wire-format
        # selector and steady-state materialization never infers a Provider ID
        # from it.  External-gateway metadata (listeners: []) stays untouched.
        if router_owns_transport and model.get("api_format") == "anthropic":
            model["backend_refs"] = [{"provider": "anthropic"}]
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
    migrated_by_name: dict[str, dict[str, Any]] = {}
    for card in _clone_list(routing.get("modelCards")):
        if not isinstance(card, dict):
            continue
        alias = str(card.get("name") or "").strip()
        if alias in catalog_by_alias:
            card["name"] = catalog_by_alias[alias]
        _migrate_quality_score(card)
        if set(card) == {"name"}:
            continue
        card_name = str(card.get("name") or "").strip()
        existing = migrated_by_name.get(card_name)
        if existing is not None:
            _merge_model_card(existing, card, card_name)
            continue
        migrated_cards.append(card)
        migrated_by_name[card_name] = card

    if migrated_cards:
        routing["modelCards"] = migrated_cards
    else:
        routing.pop("modelCards", None)


def _merge_model_card(
    target: dict[str, Any], source: dict[str, Any], card_name: str
) -> None:
    """Coalesce aliases that now share one canonical Model Card identity."""

    for key, value in source.items():
        if key == "name":
            continue
        if key not in target:
            target[key] = deepcopy(value)
            continue
        _merge_model_card_value(target, key, value, card_name, key)


def _merge_model_card_value(
    target: dict[str, Any],
    key: str,
    value: Any,
    card_name: str,
    field_path: str,
) -> None:
    current = target[key]
    if current == value:
        return
    if isinstance(current, dict) and isinstance(value, dict):
        for nested_key, nested_value in value.items():
            nested_path = f"{field_path}.{nested_key}"
            if nested_key not in current:
                current[nested_key] = deepcopy(nested_value)
            else:
                _merge_model_card_value(
                    current, nested_key, nested_value, card_name, nested_path
                )
        return
    raise ValueError(
        "routing.modelCards aliases collapse to "
        f"{card_name!r} with conflicting field {field_path!r}"
    )


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
