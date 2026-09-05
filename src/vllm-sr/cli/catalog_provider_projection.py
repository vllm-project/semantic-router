"""Catalog-driven provider projection for generated runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from cli.model_catalog import DEFAULT_CHANNEL, _load_catalog_document
from cli.model_catalog_types import ModelCatalogError
from cli.models import BackendRef, Model, UserConfig


class CatalogProviderProjectionError(ValueError):
    """A public provider binding cannot be projected from the built-in catalog."""


@dataclass(frozen=True)
class _CatalogProviderIndex:
    providers: dict[str, dict[str, Any]]
    models: dict[str, dict[str, Any]]


_API_FORMAT_TO_PROTOCOL = {
    "openai": "openai/chat-completions@1",
    "responses": "openai/responses@1",
    "anthropic": "anthropic/messages@1",
}
_PROTOCOL_TO_API_FORMAT = {
    protocol: api_format for api_format, protocol in _API_FORMAT_TO_PROTOCOL.items()
}
_PROVIDER_MODEL_ID_KIND_DEPLOYMENT_NAME = "deployment_name"


def resolve_builtin_provider_id(
    candidate: str,
    *,
    required_protocol: str,
    compatible_fallback: str = "openai-compatible",
) -> str:
    """Resolve an external provider name without conflating it with its API.

    Exact catalog Provider IDs preserve identity (for example ``vllm`` or
    ``openai``).  Unknown names using an OpenAI-compatible wire contract map to
    the generic compatibility provider, never to the OpenAI first-party API.
    """

    normalized = candidate.strip().lower()
    providers = _catalog_provider_index().providers
    provider_id = normalized if normalized in providers else compatible_fallback
    provider = providers.get(provider_id)
    if provider is None:
        raise CatalogProviderProjectionError(
            f"built-in compatible provider {provider_id!r} is unavailable"
        )
    protocols = _string_list(provider.get("protocols"), f"provider {provider_id!r}")
    operations = _string_list(
        provider.get("supported_operations"), f"provider {provider_id!r}"
    )
    if (
        required_protocol not in protocols
        or f"{required_protocol}#create" not in operations
    ):
        raise CatalogProviderProjectionError(
            f"Provider ID {provider_id!r} cannot create requests using "
            f"{required_protocol!r}"
        )
    return provider_id


def project_provider_models_for_envoy(user_config: UserConfig) -> tuple[Model, ...]:
    """Return immutable-input, catalog-materialized models for Envoy rendering.

    The public config remains sparse. Provider URLs, protocol choices, auth
    conventions, and default headers all come from the same generated catalog
    snapshot consumed by the Router and Dashboard.
    """

    catalog = _catalog_provider_index()
    projected_models: list[Model] = []
    for model_index, authored_model in enumerate(user_config.providers.models):
        projected = authored_model.model_copy(deep=True)
        card = _resolve_model_card(catalog, authored_model, model_index)
        _validate_envoy_physical_backend(projected, card, model_index)
        selected_protocol = ""
        for backend_index, backend in enumerate(projected.backend_refs):
            path = f"providers.models[{model_index}].backend_refs[{backend_index}]"
            provider = catalog.providers.get(backend.provider)
            if provider is None:
                raise CatalogProviderProjectionError(
                    f"{path}.provider {backend.provider!r} is not a built-in Provider ID"
                )
            protocol, model_id = _resolve_binding(
                authored_model,
                card,
                provider,
                selected_protocol=selected_protocol,
                path=path,
            )
            selected_protocol = protocol
            _apply_provider_defaults(backend, provider, path)
            if backend.weight == 0:
                backend.weight = 1
            if model_id:
                external_ids = dict(projected.external_model_ids or {})
                external_ids.setdefault(backend.provider, model_id)
                projected.external_model_ids = external_ids

        if selected_protocol and not projected.api_format:
            projected.api_format = _api_format_for_protocol(
                selected_protocol,
                f"providers.models[{model_index}]",
            )
        projected_models.append(projected)
    return tuple(projected_models)


def _validate_envoy_physical_backend(
    model: Model,
    card: dict[str, Any],
    model_index: int,
) -> None:
    """Require an explicit provider whenever the CLI is building Envoy transport."""

    if model.backend_refs or card.get("kind") == "virtual":
        return
    raise CatalogProviderProjectionError(
        f"providers.models[{model_index}] {model.name!r} is a physical model used "
        "by a router-owned listener and must define backend_refs with an explicit "
        "Provider ID; api_format selects only the upstream wire format"
    )


def _resolve_model_card(
    catalog: _CatalogProviderIndex,
    model: Model,
    model_index: int,
) -> dict[str, Any]:
    if not model.catalog:
        return {"id": model.name, "kind": "physical"}
    card = catalog.models.get(model.catalog)
    if card is None:
        raise CatalogProviderProjectionError(
            f"providers.models[{model_index}].catalog {model.catalog!r} "
            "is not a built-in model"
        )
    return card


def _resolve_binding(
    model: Model,
    card: dict[str, Any],
    provider: dict[str, Any],
    *,
    selected_protocol: str,
    path: str,
) -> tuple[str, str]:
    provider_id = _required_string(provider, "id", f"{path}.provider")
    card_id = _required_string(card, "id", f"{path}.catalog")
    model_id = _configured_model_id(model, provider_id)
    bindings = _catalog_model_bindings(provider, card_id, model_id, path)
    available_protocols = _binding_protocols(bindings, path)

    explicit_protocol = ""
    if model.api_format:
        explicit_protocol = _protocol_for_api_format(model.api_format, path)
    protocol = explicit_protocol or _infer_protocol(
        provider,
        card,
        model_id,
        available_protocols,
        selected_protocol=selected_protocol,
        path=path,
    )
    _validate_protocol(provider, protocol, selected_protocol, path)

    matching_binding = next(
        (
            binding
            for binding in bindings
            if protocol in _string_list(binding.get("protocols"), path)
        ),
        None,
    )
    if not model_id and matching_binding is not None:
        if _provider_model_id_kind(matching_binding, f"{path}.model"):
            raise CatalogProviderProjectionError(
                f"{path}: provider_model_id or external_model_ids[{provider_id!r}] "
                f"is required for provider {provider_id!r} because its catalog "
                "mapping uses an operator-defined deployment name"
            )
        model_id = _required_string(matching_binding, "id", f"{path}.model")

    if card.get("kind") == "physical" and not model_id:
        all_bindings = _catalog_model_bindings(provider, card_id, "", path)
        all_protocols = _binding_protocols(all_bindings, path)
        if all_protocols:
            raise CatalogProviderProjectionError(
                f"{path} has no catalog mapping for model {card_id!r} through "
                f"provider {provider_id!r} using protocol {protocol!r}; "
                f"available protocols: {', '.join(all_protocols)}"
            )
        raise CatalogProviderProjectionError(
            f"{path} cannot infer a native model ID because provider {provider_id!r} "
            f"has no catalog mapping for model {card_id!r}; set provider_model_id "
            "or external_model_ids explicitly"
        )
    return protocol, model_id


def _configured_model_id(model: Model, provider_id: str) -> str:
    if model.external_model_ids:
        provider_model_id = model.external_model_ids.get(provider_id, "").strip()
        if provider_model_id:
            return provider_model_id
    if model.provider_model_id:
        return model.provider_model_id.strip()
    if not model.catalog:
        return model.name
    return ""


def _catalog_model_bindings(
    provider: dict[str, Any],
    card_id: str,
    model_id: str,
    path: str,
) -> list[dict[str, Any]]:
    raw_bindings = provider.get("models") or []
    if not isinstance(raw_bindings, list):
        raise CatalogProviderProjectionError(
            f"{path}.provider catalog models must be a list"
        )
    result: list[dict[str, Any]] = []
    for index, binding in enumerate(raw_bindings):
        if not isinstance(binding, dict):
            raise CatalogProviderProjectionError(
                f"{path}.provider catalog contains an invalid model mapping"
            )
        if binding.get("catalog") != card_id:
            continue
        binding_path = f"{path}.provider catalog models[{index}]"
        if model_id and not _catalog_binding_accepts_model_id(
            binding, model_id, binding_path
        ):
            continue
        result.append(binding)
    return result


def _catalog_binding_accepts_model_id(
    binding: dict[str, Any], model_id: str, path: str
) -> bool:
    return (
        binding.get("id") == model_id
        or _provider_model_id_kind(binding, path)
        == _PROVIDER_MODEL_ID_KIND_DEPLOYMENT_NAME
    )


def _provider_model_id_kind(binding: dict[str, Any], path: str) -> str:
    restrictions = binding.get("restrictions")
    if restrictions is None:
        return ""
    if not isinstance(restrictions, dict):
        raise CatalogProviderProjectionError(f"{path}.restrictions must be an object")
    kind = restrictions.get("provider_model_id_kind")
    if kind is None:
        return ""
    if kind != _PROVIDER_MODEL_ID_KIND_DEPLOYMENT_NAME:
        raise CatalogProviderProjectionError(
            f"{path}.restrictions.provider_model_id_kind {kind!r} is unsupported"
        )
    return kind


def _binding_protocols(bindings: list[dict[str, Any]], path: str) -> tuple[str, ...]:
    protocols: list[str] = []
    for binding in bindings:
        for protocol in _string_list(binding.get("protocols"), path):
            if protocol not in protocols:
                protocols.append(protocol)
    return tuple(protocols)


def _infer_protocol(
    provider: dict[str, Any],
    card: dict[str, Any],
    model_id: str,
    protocols: tuple[str, ...],
    *,
    selected_protocol: str,
    path: str,
) -> str:
    provider_id = _required_string(provider, "id", f"{path}.provider")
    card_id = _required_string(card, "id", f"{path}.catalog")
    if not protocols:
        if not model_id and card.get("kind") == "physical":
            raise CatalogProviderProjectionError(
                f"{path}.protocol cannot be inferred: provider {provider_id!r} has "
                f"no catalog mapping for model {card_id!r}; set provider_model_id "
                "and api_format explicitly"
            )
        if selected_protocol:
            return selected_protocol
        return _required_string(
            provider, "default_protocol", f"{path}.provider catalog"
        )
    if len(protocols) == 1:
        return protocols[0]
    if selected_protocol:
        if selected_protocol in protocols:
            return selected_protocol
        raise CatalogProviderProjectionError(
            f"{path}.protocol cannot use {selected_protocol!r} selected by another "
            f"backend; provider {provider_id!r} exposes model {card_id!r} through "
            f"{', '.join(protocols)}"
        )
    default_protocol = _required_string(
        provider, "default_protocol", f"{path}.provider catalog"
    )
    if default_protocol in protocols:
        return default_protocol
    raise CatalogProviderProjectionError(
        f"{path}.protocol is ambiguous: provider {provider_id!r} exposes model "
        f"{card_id!r} through {', '.join(protocols)} and its default protocol is "
        "not one of them; set api_format explicitly"
    )


def _validate_protocol(
    provider: dict[str, Any],
    protocol: str,
    selected_protocol: str,
    path: str,
) -> None:
    provider_id = _required_string(provider, "id", f"{path}.provider")
    if protocol not in _string_list(provider.get("protocols"), path):
        raise CatalogProviderProjectionError(
            f"{path}.protocol {protocol!r} is not supported by provider {provider_id!r}"
        )
    operations = _string_list(provider.get("supported_operations"), path)
    if f"{protocol}#create" not in operations:
        raise CatalogProviderProjectionError(
            f"{path}.protocol {protocol!r} cannot create requests through provider "
            f"{provider_id!r}"
        )
    if selected_protocol and protocol != selected_protocol:
        raise CatalogProviderProjectionError(
            f"{path}.protocol {protocol!r} conflicts with {selected_protocol!r}; "
            "one model alias must use one wire protocol"
        )


def _apply_provider_defaults(
    backend: BackendRef,
    provider: dict[str, Any],
    path: str,
) -> None:
    provider_id = _required_string(provider, "id", f"{path}.provider")
    if not backend.endpoint and not backend.base_url:
        default_base_url = str(provider.get("default_base_url") or "").strip()
        if not default_base_url:
            raise CatalogProviderProjectionError(
                f"{path} requires endpoint or base_url because provider "
                f"{provider_id!r} has no default"
            )
        backend.base_url = default_base_url

    default_headers = _string_mapping(provider.get("default_headers"), path)
    extra_headers = {**default_headers, **(backend.extra_headers or {})}
    backend.extra_headers = extra_headers or None

    auth = provider.get("auth") or {}
    if not isinstance(auth, dict):
        raise CatalogProviderProjectionError(
            f"{path}.provider catalog auth must be an object"
        )
    if not backend.auth_header:
        backend.auth_header = str(auth.get("header") or "").strip() or None
    if backend.auth_prefix is None:
        backend.auth_prefix = str(auth.get("prefix") or "").strip()


def _protocol_for_api_format(api_format: str, path: str) -> str:
    protocol = _API_FORMAT_TO_PROTOCOL.get(api_format)
    if protocol is None:
        raise CatalogProviderProjectionError(
            f"{path}.api_format {api_format!r} is unsupported; use openai, "
            "responses, or anthropic"
        )
    return protocol


def _api_format_for_protocol(protocol: str, path: str) -> str:
    api_format = _PROTOCOL_TO_API_FORMAT.get(protocol)
    if api_format is None:
        raise CatalogProviderProjectionError(
            f"{path} resolved unsupported catalog protocol {protocol!r}"
        )
    return api_format


def _required_string(value: dict[str, Any], key: str, path: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise CatalogProviderProjectionError(f"{path}.{key} must be a string")
    return result.strip()


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CatalogProviderProjectionError(
            f"{path}.provider catalog protocols must be a string list"
        )
    return tuple(item.strip() for item in value)


def _string_mapping(value: Any, path: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise CatalogProviderProjectionError(
            f"{path}.provider catalog default_headers must be a string map"
        )
    return dict(value)


@lru_cache(maxsize=1)
def _catalog_provider_index() -> _CatalogProviderIndex:
    try:
        _, document = _load_catalog_document(DEFAULT_CHANNEL)
    except ModelCatalogError as error:
        raise CatalogProviderProjectionError(
            "the packaged built-in model catalog is unavailable"
        ) from error
    return _CatalogProviderIndex(
        providers=_index_catalog_entries(document.get("providers"), "provider"),
        models=_index_catalog_entries(document.get("models"), "model"),
    )


def _index_catalog_entries(value: Any, kind: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise CatalogProviderProjectionError(
            f"the packaged catalog {kind} inventory must be a list"
        )
    result: dict[str, dict[str, Any]] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise CatalogProviderProjectionError(
                f"the packaged catalog contains an invalid {kind} entry"
            )
        entry_id = _required_string(entry, "id", f"catalog {kind}")
        if entry_id in result:
            raise CatalogProviderProjectionError(
                f"the packaged catalog contains duplicate {kind} {entry_id!r}"
            )
        result[entry_id] = entry
    return result
