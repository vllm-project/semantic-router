"""Validate PII token-window geometry without loading any model resources."""

from cli.model_runtime_defaults import iter_effective_routing_profiles
from cli.models import UserConfig
from cli.validation_error import ValidationError


def validate_pii_windows(
    config: UserConfig, deployments: dict
) -> list[ValidationError]:
    catalog = (config.global_ or {}).get("model_catalog") or {}
    pii = ((catalog.get("modules") or {}).get("classifier") or {}).get("pii") or {}
    window = pii.get("window")
    errors = []
    for name, routing in iter_effective_routing_profiles(config):
        prefix = "routing" if name == "default" else f"recipes.{name}.routing"
        binding = routing.model_bindings.get("pii_classifier")
        field = "global.model_catalog.modules.classifier.pii.window"
        if binding is not None:
            deployment = deployments.get(binding.deployment)
            if deployment is None:
                continue  # The shared reference validator reports this case.
            field = f"{prefix}.model_bindings.pii_classifier"
            message = _bound_window_error(window, deployment)
        else:
            message = _local_window_error(pii)
        if message:
            errors.append(ValidationError(message, field=field))
    return errors


def _bound_window_error(window, deployment):
    budget = deployment.get("input") or {}
    overflow = budget.get("overflow") or "reject"
    if window is None:
        if overflow == "window":
            return "PII window overflow requires explicit module window geometry"
        return None
    if deployment.get("provider") not in {"candle", "ort"}:
        return "PII window requires a local Candle or ORT deployment"
    if overflow != "window":
        return "PII window requires deployment input.overflow=window"
    limit = budget.get("max_tokens", 0)
    if not _integer(limit) or limit <= 0:
        return "PII window requires a positive deployment input.max_tokens"
    return _geometry_error(window, limit)


def _local_window_error(pii):
    if pii.get("window") is None:
        return None
    # Sparse canonical PII modules inherit the local mmBERT selector. A
    # supplied backend replaces that selector during canonical normalization.
    if pii.get("backend") is not None or pii.get("use_mmbert_32k", True) is not True:
        return "PII window requires the local mmbert32k model"
    limit = pii.get("max_sequence_length", 0)
    if not _integer(limit) or limit < 0:
        return "PII max_sequence_length must be a nonnegative integer"
    return _geometry_error(pii["window"], limit or 512)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _geometry_error(window, limit):
    if not isinstance(window, dict):
        return "PII window must be an object"
    size, overlap = window.get("size", 0), window.get("overlap", 0)
    if not _integer(size) or size <= 0 or size > limit:
        return "PII window.size must be positive and fit the effective input budget"
    if not _integer(overlap) or overlap < 0 or overlap >= size:
        return "PII window.overlap must be nonnegative and smaller than window.size"
    return None
