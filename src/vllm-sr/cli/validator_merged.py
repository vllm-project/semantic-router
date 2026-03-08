"""Merged runtime-config validation helpers."""

from typing import Any

from cli.consts import EXTERNAL_API_MODEL_FORMATS
from cli.validator_common import ValidationError, iter_merged_condition_nodes

_REQUIRED_MERGED_FIELDS = (
    "vllm_endpoints",
    "model_config",
    "default_model",
    "decisions",
    "categories",
)


def validate_merged_config(merged_config: dict[str, Any]) -> list[ValidationError]:
    """Validate the merged router configuration."""
    errors: list[ValidationError] = []
    errors.extend(_missing_required_field_errors(merged_config))
    errors.extend(_endpoint_errors(merged_config))
    errors.extend(_category_errors(merged_config))
    errors.extend(_contrastive_preference_errors(merged_config))
    return errors


def _missing_required_field_errors(
    merged_config: dict[str, Any],
) -> list[ValidationError]:
    return [
        ValidationError(f"Missing required field: {field}", field=field)
        for field in _REQUIRED_MERGED_FIELDS
        if field not in merged_config
    ]


def _endpoint_errors(merged_config: dict[str, Any]) -> list[ValidationError]:
    if "vllm_endpoints" not in merged_config:
        return []

    errors: list[ValidationError] = []
    endpoints = merged_config["vllm_endpoints"]
    if not endpoints and not _all_models_use_external_api(merged_config):
        errors.append(
            ValidationError("No vLLM endpoints configured", field="vllm_endpoints")
        )

    endpoint_names: set[str] = set()
    for endpoint in endpoints:
        endpoint_name = endpoint["name"]
        if endpoint_name in endpoint_names:
            errors.append(
                ValidationError(
                    f"Duplicate endpoint name: {endpoint_name}",
                    field="vllm_endpoints",
                )
            )
            continue
        endpoint_names.add(endpoint_name)

    return errors


def _category_errors(merged_config: dict[str, Any]) -> list[ValidationError]:
    categories = merged_config.get("categories")
    if categories or not _has_domain_conditions(merged_config):
        return []

    return [
        ValidationError(
            "No categories configured or auto-generated",
            field="categories",
        )
    ]


def _contrastive_preference_errors(
    merged_config: dict[str, Any],
) -> list[ValidationError]:
    preference_cfg = _preference_model_config(merged_config)
    if not preference_cfg.get("embedding_model"):
        return []
    if _has_embedding_model_path(merged_config.get("embedding_models", {}) or {}):
        return []
    return [
        ValidationError(
            "preference_model.use_contrastive=true requires an embedding model path "
            "(qwen3_model_path, gemma_model_path, or mmbert_model_path)",
            field="embedding_models",
        )
    ]


def _all_models_use_external_api(merged_config: dict[str, Any]) -> bool:
    model_config = merged_config.get("model_config") or {}
    if not model_config:
        return False
    return all(
        model_cfg.get("api_format") in EXTERNAL_API_MODEL_FORMATS
        for model_cfg in model_config.values()
        if isinstance(model_cfg, dict)
    )


def _has_domain_conditions(merged_config: dict[str, Any]) -> bool:
    return any(
        condition.get("type") == "domain"
        for decision in merged_config.get("decisions", [])
        for condition in iter_merged_condition_nodes(
            decision.get("rules", {}).get("conditions", [])
        )
    )


def _preference_model_config(merged_config: dict[str, Any]) -> dict[str, Any]:
    classifier_cfg = merged_config.get("classifier") or {}
    if not classifier_cfg:
        return {}
    return classifier_cfg.get("preference_model", {}) or {}


def _has_embedding_model_path(embedding_cfg: dict[str, Any]) -> bool:
    return any(
        embedding_cfg.get(key)
        for key in (
            "qwen3_model_path",
            "gemma_model_path",
            "mmbert_model_path",
        )
    )
