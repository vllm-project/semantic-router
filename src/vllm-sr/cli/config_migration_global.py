"""Compatibility helpers for migrating legacy global config blocks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_CANONICAL_GLOBAL_BLOCKS = {
    "router",
    "services",
    "stores",
    "integrations",
    "model_catalog",
}
_ROUTER_KEYS = {
    "strategy",
    "auto_model_name",
    "auto_model_names",
    "include_config_models_in_list",
    "clear_route_cache",
    "model_selection",
}
_STREAMED_BODY_KEYS = {
    "streamed_body_mode": "enabled",
    "max_streamed_body_bytes": "max_bytes",
    "streamed_body_timeout_sec": "timeout_sec",
}
_DIRECT_SERVICE_KEYS = {
    "response_api",
    "router_replay",
    "api",
    "observability",
    "authz",
    "ratelimit",
}
_DIRECT_STORE_KEYS = {
    "semantic_cache",
    "response_cache",
    "memory",
    "vector_store",
}
_DIRECT_INTEGRATION_KEYS = {"tools", "looper"}


def normalize_global_layout(global_config: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical global layout for a legacy global config block."""

    if not global_config:
        return {}

    normalized = deepcopy(global_config)
    legacy_runtime = _as_dict(normalized.pop("runtime", {}))
    if legacy_runtime:
        legacy_router = _as_dict(legacy_runtime.pop("router", {}))
        for key, value in legacy_router.items():
            place_global_block(normalized, key, value)
        for key, value in legacy_runtime.items():
            place_global_block(normalized, key, value)

    legacy_models = _as_dict(normalized.pop("models", {}))
    if legacy_models:
        embeddings = _as_dict(legacy_models.get("embeddings"))
        if embeddings and "semantic" in embeddings:
            place_global_block(normalized, "embedding_models", embeddings["semantic"])
        if "system" in legacy_models:
            place_global_block(normalized, "system_models", legacy_models["system"])
        if "external" in legacy_models:
            place_global_block(normalized, "external_models", legacy_models["external"])

    legacy_modules = _as_dict(normalized.pop("modules", {}))
    if legacy_modules:
        for key, value in legacy_modules.items():
            place_global_block(normalized, key, value)

    for key, value in list(global_config.items()):
        place_global_block(normalized, key, value)
    return normalized


def place_global_block(global_config: dict[str, Any], key: str, value: Any) -> None:
    """Place one legacy top-level value in its canonical global block."""

    if key == "auto_model_names" and isinstance(value, list):
        _ensure_dict(global_config, "router").setdefault(key, deepcopy(value))
        global_config.pop(key, None)
        return
    if value in (None, "", [], {}):
        return

    router = _ensure_dict(global_config, "router")
    services = _ensure_dict(global_config, "services")
    stores = _ensure_dict(global_config, "stores")
    integrations = _ensure_dict(global_config, "integrations")
    model_catalog = _ensure_dict(global_config, "model_catalog")
    embeddings = _ensure_dict(model_catalog, "embeddings")
    modules = _ensure_dict(model_catalog, "modules")
    classifier = _ensure_dict(modules, "classifier")
    hallucination = _ensure_dict(modules, "hallucination_mitigation")

    if key in _CANONICAL_GLOBAL_BLOCKS:
        return
    if _place_runtime_block(
        key, value, router, services, stores, integrations
    ) or _place_model_catalog_block(key, value, model_catalog, embeddings, modules):
        global_config.pop(key, None)
        return
    if key == "classifier":
        _place_classifier_block(classifier, value)
        global_config.pop(key, None)
        return
    if key == "hallucination_mitigation":
        _place_hallucination_block(hallucination, value)
        global_config.pop(key, None)


def _place_runtime_block(
    key: str,
    value: Any,
    router: dict[str, Any],
    services: dict[str, Any],
    stores: dict[str, Any],
    integrations: dict[str, Any],
) -> bool:
    if key in _ROUTER_KEYS:
        router.setdefault(key, deepcopy(value))
        return True
    streamed_body_key = _STREAMED_BODY_KEYS.get(key)
    if streamed_body_key:
        streamed_body = _ensure_dict(router, "streamed_body")
        streamed_body.setdefault(streamed_body_key, deepcopy(value))
        return True
    if key in _DIRECT_SERVICE_KEYS:
        services.setdefault(key, deepcopy(value))
        return True
    if key in _DIRECT_STORE_KEYS:
        stores.setdefault(key, deepcopy(value))
        return True
    if key in _DIRECT_INTEGRATION_KEYS:
        integrations.setdefault(key, deepcopy(value))
        return True
    return False


def _place_model_catalog_block(
    key: str,
    value: Any,
    model_catalog: dict[str, Any],
    embeddings: dict[str, Any],
    modules: dict[str, Any],
) -> bool:
    direct_targets = {
        "system_models": (model_catalog, "system"),
        "external_models": (model_catalog, "external"),
        "embedding_models": (embeddings, "semantic"),
        "prompt_compression": (modules, "prompt_compression"),
        "modality_detector": (modules, "modality_detector"),
    }
    direct_target = direct_targets.get(key)
    if direct_target is not None:
        target, canonical_key = direct_target
        target.setdefault(canonical_key, deepcopy(value))
        return True
    if key == "bert_model":
        _place_legacy_bert_model(embeddings, value)
        return True
    if key == "prompt_guard":
        prompt_guard = deepcopy(value) if isinstance(value, dict) else {}
        if "model_id" in prompt_guard and "model_ref" not in prompt_guard:
            prompt_guard["model_ref"] = "prompt_guard"
        modules.setdefault(key, prompt_guard)
        return True
    if key == "feedback_detector":
        feedback = deepcopy(value) if isinstance(value, dict) else {}
        if "model_id" in feedback and "model_ref" not in feedback:
            feedback["model_ref"] = "feedback_detector"
        modules.setdefault(key, feedback)
        return True
    return False


def _place_legacy_bert_model(embeddings: dict[str, Any], value: Any) -> None:
    semantic = _ensure_dict(embeddings, "semantic")
    legacy_bert = deepcopy(value) if isinstance(value, dict) else {}
    if "model_id" in legacy_bert and "bert_model_path" not in semantic:
        semantic["bert_model_path"] = deepcopy(legacy_bert["model_id"])
    if "use_cpu" in legacy_bert and "use_cpu" not in semantic:
        semantic["use_cpu"] = deepcopy(legacy_bert["use_cpu"])
    if "threshold" in legacy_bert:
        embedding_config = _ensure_dict(semantic, "embedding_config")
        embedding_config.setdefault(
            "min_score_threshold", deepcopy(legacy_bert["threshold"])
        )


def _place_classifier_block(classifier: dict[str, Any], value: Any) -> None:
    classifier_value = deepcopy(value) if isinstance(value, dict) else {}
    if "category_model" in classifier_value:
        domain = deepcopy(classifier_value.pop("category_model"))
        if (
            isinstance(domain, dict)
            and "model_id" in domain
            and "model_ref" not in domain
        ):
            domain["model_ref"] = "domain_classifier"
        classifier.setdefault("domain", domain)
    if "pii_model" in classifier_value:
        pii = deepcopy(classifier_value.pop("pii_model"))
        if isinstance(pii, dict) and "model_id" in pii and "model_ref" not in pii:
            pii["model_ref"] = "pii_classifier"
        classifier.setdefault("pii", pii)
    if "mcp_category_model" in classifier_value:
        classifier.setdefault(
            "mcp", deepcopy(classifier_value.pop("mcp_category_model"))
        )
    if "preference_model" in classifier_value:
        classifier.setdefault(
            "preference", deepcopy(classifier_value.pop("preference_model"))
        )


def _place_hallucination_block(hallucination: dict[str, Any], value: Any) -> None:
    hallucination_value = deepcopy(value) if isinstance(value, dict) else {}
    _move_model_ref(
        hallucination,
        hallucination_value,
        legacy_key="fact_check_model",
        canonical_key="fact_check",
        model_ref="fact_check_classifier",
    )
    _move_model_ref(
        hallucination,
        hallucination_value,
        legacy_key="hallucination_model",
        canonical_key="detector",
        model_ref="hallucination_detector",
    )
    _move_model_ref(
        hallucination,
        hallucination_value,
        legacy_key="nli_model",
        canonical_key="explainer",
        model_ref="hallucination_explainer",
    )
    if "enabled" in hallucination_value:
        hallucination.setdefault("enabled", deepcopy(hallucination_value["enabled"]))


def _move_model_ref(
    target: dict[str, Any],
    source: dict[str, Any],
    *,
    legacy_key: str,
    canonical_key: str,
    model_ref: str,
) -> None:
    if legacy_key not in source:
        return
    model = deepcopy(source.pop(legacy_key))
    if isinstance(model, dict) and "model_id" in model and "model_ref" not in model:
        model["model_ref"] = model_ref
    target.setdefault(canonical_key, model)


def _ensure_dict(target: dict[str, Any], key: str) -> dict[str, Any]:
    existing = target.get(key)
    if isinstance(existing, dict):
        return existing
    created: dict[str, Any] = {}
    target[key] = created
    return created


def _as_dict(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}
