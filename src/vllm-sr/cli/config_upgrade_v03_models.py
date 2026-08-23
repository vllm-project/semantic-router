"""Translate v0.3 provider bindings and model cards into v0.4 Models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from cli.config_upgrade_v03_support import (
    MigrationContext,
    as_list,
    as_mapping,
    canonical_decimal,
    environment_reference,
    reject_unknown_fields,
)

_PROVIDER_FIELDS = frozenset(
    {
        "name",
        "reasoning_family",
        "provider_model_id",
        "backend_refs",
        "pricing",
        "reliability",
        "api_format",
        "external_model_ids",
    }
)
_CARD_FIELDS = frozenset(
    {
        "name",
        "param_size",
        "context_window_size",
        "description",
        "capabilities",
        "loras",
        "tags",
        "quality_score",
        "modality",
    }
)
_BACKEND_FIELDS = frozenset(
    {
        "name",
        "endpoint",
        "protocol",
        "weight",
        "type",
        "base_url",
        "provider",
        "auth_header",
        "auth_prefix",
        "extra_headers",
        "api_version",
        "chat_path",
        "api_key",
        "api_key_env",
    }
)
_PRICING_FIELDS = frozenset(
    {
        "currency",
        "prompt_per_1m",
        "cached_input_per_1m",
        "cache_write_per_1m",
        "completion_per_1m",
    }
)
_RELIABILITY_FIELDS = frozenset(
    {
        "lb_policy",
        "retry_count",
        "retry_on",
        "consecutive_5xx",
        "base_ejection_time",
        "max_ejection_percent",
        "health_check_path",
        "health_check_interval",
        "health_check_timeout",
    }
)
_REASONING_FAMILY_FIELDS = frozenset({"type", "parameter"})
_KNOWN_REASONING_PARAMETERS = {
    "chat_template_kwargs": "enable_thinking",
    "reasoning_effort": "reasoning.effort",
    "top_level_reasoning_effort": "reasoning_effort",
}
_API_FORMAT_PROVIDERS = {
    "anthropic": "anthropic-compatible",
    "anthropic-compatible": "anthropic-compatible",
    "openai": "openai-compatible",
    "openai-compatible": "openai-compatible",
}
_TRANSPORT_TYPES = {"chat", "completion", "completions", "text"}
_STANDARD_CHAT_PATHS = {
    "/chat/completions",
    "/v1/chat/completions",
    "/v1/messages",
}


@dataclass(frozen=True)
class ModelUpgradeResult:
    """Model resources plus global metadata produced by the translation."""

    models: list[dict[str, Any]]
    credentials: dict[str, dict[str, str]]
    default_model: str | None
    billing_currency: str | None


class _CredentialRegistry:
    def __init__(self, context: MigrationContext) -> None:
        self.context = context
        self.definitions: dict[str, dict[str, str]] = {}
        self.by_source: dict[tuple[str, str], str] = {}

    def bind(
        self,
        *,
        model_name: str,
        backend_name: str,
        secret_env: str,
        adapter: str,
    ) -> str:
        source = (adapter, secret_env)
        if source in self.by_source:
            return self.by_source[source]

        base = _slug(f"{model_name}-{backend_name}-credential")
        candidate = base
        suffix = 2
        while candidate in self.definitions:
            candidate = f"{base}-{suffix}"
            suffix += 1
        self.definitions[candidate] = {
            "credential_adapter_id": adapter,
            "secret_env": secret_env,
        }
        self.by_source[source] = candidate
        return candidate


def v03_default_model(providers_value: Any) -> str | None:
    """Read the v0.3 default model without interpreting any runtime behavior."""

    if not isinstance(providers_value, dict):
        return None
    defaults = providers_value.get("defaults")
    if not isinstance(defaults, dict):
        return None
    value = defaults.get("default_model")
    return value.strip() if isinstance(value, str) and value.strip() else None


def translate_v03_models(
    providers_value: Any,
    cards_value: Any,
    reasoning_efforts: dict[str, set[str]],
    context: MigrationContext,
) -> ModelUpgradeResult:
    """Translate v0.3 providers.models plus routing.modelCards."""

    providers = as_mapping(providers_value, "providers", context)
    reject_unknown_fields(providers, {"defaults", "models"}, "providers", context)
    defaults = as_mapping(providers.get("defaults"), "providers.defaults", context)
    reject_unknown_fields(
        defaults,
        {"default_model", "reasoning_families", "default_reasoning_effort"},
        "providers.defaults",
        context,
    )
    default_model = _optional_trimmed(defaults.get("default_model"))
    default_effort = _optional_trimmed(defaults.get("default_reasoning_effort"))
    reasoning_families = _reasoning_families(defaults, context)

    cards = _model_cards(cards_value, context)
    credentials = _CredentialRegistry(context)
    currencies: set[str] = set()
    models: list[dict[str, Any]] = []
    seen_models: set[str] = set()

    for index, raw_model in enumerate(
        as_list(providers.get("models"), "providers.models", context)
    ):
        path = f"providers.models[{index}]"
        model = as_mapping(raw_model, path, context)
        reject_unknown_fields(model, _PROVIDER_FIELDS, path, context)
        name = _required_name(model.get("name"), f"{path}.name", context)
        if not name:
            continue
        if name in seen_models:
            context.add(
                "duplicate_model",
                f"{path}.name",
                f"model {name!r} is declared more than once",
                "keep one provider model per logical name",
            )
            continue
        seen_models.add(name)

        card = _translate_card(
            name,
            cards.pop(name, None),
            model,
            reasoning_families,
            default_effort,
            reasoning_efforts.get(name, set()),
            context,
        )
        connections = _translate_connections(
            name,
            model,
            credentials,
            context,
            path,
        )
        runtime = _translate_reliability(model.get("reliability"), path, context)
        pricing, currency = _translate_pricing(model.get("pricing"), path, context)
        if currency:
            currencies.add(currency)

        translated: dict[str, Any] = {
            "name": name,
            "card": card,
            "connections": connections,
        }
        if runtime:
            translated["runtime"] = runtime
        if pricing:
            translated["pricing"] = pricing
        models.append(translated)

    for name, (_, path) in sorted(cards.items()):
        context.add(
            "model_card_without_connection",
            path,
            f"model card {name!r} has no matching providers.models entry",
            "add a provider model with at least one backend_ref or remove the orphan card",
        )

    billing_currency: str | None = None
    if len(currencies) > 1:
        context.add(
            "mixed_billing_currency",
            "providers.models[].pricing.currency",
            f"v0.3 models use multiple currencies: {', '.join(sorted(currencies))}",
            "convert prices to one ISO-4217 currency before migrating",
        )
    elif currencies:
        billing_currency = next(iter(currencies))

    return ModelUpgradeResult(
        models=models,
        credentials=credentials.definitions,
        default_model=default_model,
        billing_currency=billing_currency,
    )


def _model_cards(
    value: Any, context: MigrationContext
) -> dict[str, tuple[dict[str, Any], str]]:
    cards: dict[str, tuple[dict[str, Any], str]] = {}
    for index, raw_card in enumerate(as_list(value, "routing.modelCards", context)):
        path = f"routing.modelCards[{index}]"
        card = as_mapping(raw_card, path, context)
        reject_unknown_fields(card, _CARD_FIELDS, path, context)
        name = _required_name(card.get("name"), f"{path}.name", context)
        if not name:
            continue
        if name in cards:
            context.add(
                "duplicate_model_card",
                f"{path}.name",
                f"model card {name!r} is declared more than once",
                "keep one model card per logical Model",
            )
            continue
        cards[name] = (card, path)
    return cards


def _reasoning_families(
    defaults: dict[str, Any], context: MigrationContext
) -> dict[str, dict[str, str]]:
    raw_families = as_mapping(
        defaults.get("reasoning_families"),
        "providers.defaults.reasoning_families",
        context,
    )
    result: dict[str, dict[str, str]] = {}
    for family_name, raw_family in raw_families.items():
        path = f"providers.defaults.reasoning_families.{family_name}"
        family = as_mapping(raw_family, path, context)
        reject_unknown_fields(family, _REASONING_FAMILY_FIELDS, path, context)
        family_type = _required_name(family.get("type"), f"{path}.type", context)
        parameter = _required_name(
            family.get("parameter"), f"{path}.parameter", context
        )
        expected_parameter = _KNOWN_REASONING_PARAMETERS.get(family_type)
        if family_type and expected_parameter is None:
            context.add(
                "unsupported_reasoning_family",
                f"{path}.type",
                f"reasoning family type {family_type!r} has no v0.4 wire contract",
                "choose a documented v0.4 reasoning type",
            )
        elif parameter and parameter != expected_parameter:
            context.add(
                "unsupported_reasoning_parameter",
                f"{path}.parameter",
                f"parameter {parameter!r} is not the canonical parameter for {family_type!r}",
                f"use {expected_parameter!r} or remove reasoning controls",
            )
        if family_type and parameter:
            result[str(family_name)] = {
                "type": family_type,
                "parameter": parameter,
            }
    return result


def _translate_card(
    model_name: str,
    card_entry: tuple[dict[str, Any], str] | None,
    provider_model: dict[str, Any],
    families: dict[str, dict[str, str]],
    default_effort: str | None,
    referenced_efforts: set[str],
    context: MigrationContext,
) -> dict[str, Any]:
    raw_card, card_path = card_entry or ({"name": model_name}, "routing.modelCards")
    card: dict[str, Any] = {}
    for field_name in (
        "param_size",
        "context_window_size",
        "description",
        "capabilities",
        "quality_score",
        "modality",
        "tags",
    ):
        value = raw_card.get(field_name)
        if value not in (None, "", []):
            card[field_name] = value

    loras: list[str] = []
    for index, raw_lora in enumerate(
        as_list(raw_card.get("loras"), f"{card_path}.loras", context)
    ):
        path = f"{card_path}.loras[{index}]"
        lora = as_mapping(raw_lora, path, context)
        reject_unknown_fields(lora, {"name", "description"}, path, context)
        name = _required_name(lora.get("name"), f"{path}.name", context)
        if name:
            loras.append(name)
        if _optional_trimmed(lora.get("description")):
            context.add(
                "lossy_lora_description",
                f"{path}.description",
                "v0.4 Model cards store LoRA names but not LoRA descriptions",
                "move the description to external documentation or remove it before migrating",
            )
    if loras:
        card["loras"] = loras

    family_name = _optional_trimmed(provider_model.get("reasoning_family"))
    if family_name:
        family = families.get(family_name)
        if family is None:
            context.add(
                "unknown_reasoning_family",
                "providers.models[].reasoning_family",
                f"Model {model_name!r} references unknown family {family_name!r}",
                "declare the family in providers.defaults.reasoning_families",
            )
        else:
            efforts = set(referenced_efforts)
            if default_effort:
                efforts.add(default_effort)
            card["reasoning"] = {
                "type": family["type"],
                "efforts": sorted(efforts),
            }
    return card


def _translate_connections(
    model_name: str,
    provider_model: dict[str, Any],
    credentials: _CredentialRegistry,
    context: MigrationContext,
    model_path: str,
) -> list[dict[str, Any]]:
    backend_refs = as_list(
        provider_model.get("backend_refs"), f"{model_path}.backend_refs", context
    )
    if not backend_refs:
        context.add(
            "missing_model_connection",
            f"{model_path}.backend_refs",
            f"Model {model_name!r} has no backend connection",
            "add at least one backend_ref before migrating",
        )
        return []

    api_format = _optional_trimmed(provider_model.get("api_format"))
    normalized_format = api_format.lower() if api_format else None
    if normalized_format and normalized_format not in _API_FORMAT_PROVIDERS:
        context.add(
            "unsupported_api_format",
            f"{model_path}.api_format",
            f"API format {api_format!r} has no verified v0.4 protocol mapping",
            "use openai, openai-compatible, anthropic, or anthropic-compatible",
        )

    provider_model_id = _optional_trimmed(provider_model.get("provider_model_id"))
    external_ids = _string_mapping(
        provider_model.get("external_model_ids"),
        f"{model_path}.external_model_ids",
        context,
    )
    used_external_ids: set[str] = set()
    connections: list[dict[str, Any]] = []

    for index, raw_backend in enumerate(backend_refs):
        path = f"{model_path}.backend_refs[{index}]"
        backend = as_mapping(raw_backend, path, context)
        reject_unknown_fields(backend, _BACKEND_FIELDS, path, context)
        provider, provider_keys = _connection_provider(
            backend, normalized_format, path, context
        )
        endpoint = _connection_endpoint(backend, path, context)
        upstream_model = None
        for key in provider_keys:
            if key in external_ids:
                upstream_model = external_ids[key]
                used_external_ids.add(key)
                break
        upstream_model = upstream_model or provider_model_id or model_name

        connection: dict[str, Any] = {
            "provider": provider,
            "interface": (
                "messages"
                if provider in {"anthropic", "anthropic-compatible"}
                else "chat"
            ),
            "model": upstream_model,
        }
        if endpoint:
            connection["endpoint"] = endpoint
        weight = canonical_decimal(
            backend.get("weight", 1), f"{path}.weight", context, positive=True
        )
        if weight and weight != "1":
            connection["weight"] = weight

        secret_env = _backend_secret_env(backend, path, context)
        if secret_env:
            adapter = _credential_adapter(backend, provider, path, context)
            backend_name = _optional_trimmed(backend.get("name")) or str(index + 1)
            connection["credential"] = credentials.bind(
                model_name=model_name,
                backend_name=backend_name,
                secret_env=secret_env,
                adapter=adapter,
            )

        _reject_backend_features(backend, path, context)
        connections.append(connection)

    for key in sorted(set(external_ids) - used_external_ids):
        context.add(
            "unmapped_external_model_id",
            f"{model_path}.external_model_ids.{key}",
            "the provider-specific model ID is not selected by any backend_ref",
            "align the key with a backend name/provider/type or remove the unused mapping",
        )
    return connections


def _connection_provider(
    backend: dict[str, Any],
    api_format: str | None,
    path: str,
    context: MigrationContext,
) -> tuple[str, list[str]]:
    explicit_provider = _optional_trimmed(backend.get("provider"))
    backend_type = _optional_trimmed(backend.get("type"))
    backend_name = _optional_trimmed(backend.get("name"))
    provider = explicit_provider
    if not provider and backend_type and backend_type.lower() not in _TRANSPORT_TYPES:
        provider = backend_type
    if not provider:
        provider = _API_FORMAT_PROVIDERS.get(api_format or "", "vllm")
    if not provider:
        context.add(
            "missing_provider",
            f"{path}.provider",
            "cannot determine the v0.4 Provider Integration",
            "set backend_ref.provider or a supported model api_format",
        )
        provider = "vllm"
    lookup_keys = [
        item
        for item in (
            explicit_provider,
            backend_type,
            backend_name,
            provider,
            api_format,
        )
        if item
    ]
    return provider, list(dict.fromkeys(lookup_keys))


def _connection_endpoint(
    backend: dict[str, Any], path: str, context: MigrationContext
) -> str | None:
    endpoint = _optional_trimmed(backend.get("endpoint"))
    base_url = _optional_trimmed(backend.get("base_url"))
    if endpoint and base_url and endpoint.rstrip("/") != base_url.rstrip("/"):
        context.add(
            "ambiguous_backend_origin",
            path,
            "endpoint and base_url identify different origins",
            "keep one authoritative backend URL",
        )
    value = base_url or endpoint
    if not value:
        return None

    protocol = _optional_trimmed(backend.get("protocol"))
    if value.lower().startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if protocol and parsed.scheme.lower() != protocol.lower():
            context.add(
                "protocol_mismatch",
                f"{path}.protocol",
                f"protocol {protocol!r} conflicts with URL scheme {parsed.scheme!r}",
                "make protocol and the backend URL scheme match",
            )
        return value.rstrip("/")
    scheme = (protocol or "http").lower()
    if scheme not in {"http", "https"}:
        context.add(
            "unsupported_protocol",
            f"{path}.protocol",
            f"protocol {scheme!r} cannot be represented by a v0.4 HTTP connection",
            "use http or https",
        )
        scheme = "http"
    return f"{scheme}://{value.rstrip('/')}"


def _backend_secret_env(
    backend: dict[str, Any], path: str, context: MigrationContext
) -> str | None:
    explicit_env = environment_reference(backend.get("api_key_env"))
    raw_env = backend.get("api_key_env")
    if raw_env not in (None, "") and explicit_env is None:
        context.add(
            "invalid_secret_reference",
            f"{path}.api_key_env",
            "api_key_env must contain an environment variable name",
            "use a name such as MODEL_API_KEY",
        )

    inline_value = backend.get("api_key")
    inline_env = environment_reference(inline_value)
    if inline_value not in (None, "") and inline_env is None:
        # The recursive secret scan also catches this. Keep connection translation
        # deterministic without recording a duplicate issue.
        return explicit_env
    if explicit_env and inline_env and explicit_env != inline_env:
        context.add(
            "conflicting_secret_reference",
            path,
            "api_key and api_key_env reference different environment variables",
            "keep one environment-backed API-key reference",
        )
    return explicit_env or inline_env


def _credential_adapter(
    backend: dict[str, Any], provider: str, path: str, context: MigrationContext
) -> str:
    header = (_optional_trimmed(backend.get("auth_header")) or "").lower()
    prefix = (_optional_trimmed(backend.get("auth_prefix")) or "").lower()
    if not header:
        return (
            "x-api-key"
            if provider in {"anthropic", "anthropic-compatible"}
            else "bearer"
        )
    if header == "authorization" and prefix in {"", "bearer"}:
        return "bearer"
    if header == "x-api-key" and not prefix:
        return "x-api-key"
    if header == "api-key" and not prefix:
        return "api-key"
    context.add(
        "unsupported_auth_shape",
        f"{path}.auth_header",
        f"header {header!r} with prefix {prefix!r} has no built-in credential adapter",
        "use Authorization/Bearer, X-Api-Key, or Api-Key",
    )
    return "bearer"


def _reject_backend_features(
    backend: dict[str, Any], path: str, context: MigrationContext
) -> None:
    if backend.get("extra_headers") not in (None, {}):
        context.add(
            "unsupported_backend_headers",
            f"{path}.extra_headers",
            "arbitrary backend headers are not part of a v0.4 Model connection",
            "move stable wire headers into a Provider Integration",
        )
    if backend.get("api_version") not in (None, ""):
        context.add(
            "unsupported_backend_api_version",
            f"{path}.api_version",
            "the backend API version cannot be copied into a Model connection",
            "encode it in the Provider Integration compiler",
        )
    chat_path = _optional_trimmed(backend.get("chat_path"))
    if chat_path and chat_path not in _STANDARD_CHAT_PATHS:
        context.add(
            "custom_backend_path",
            f"{path}.chat_path",
            f"custom chat path {chat_path!r} has no connection-level v0.4 field",
            "provide a Provider Integration whose compiler owns this path",
        )


def _translate_reliability(
    value: Any, model_path: str, context: MigrationContext
) -> dict[str, Any]:
    if value is None:
        return {}
    path = f"{model_path}.reliability"
    reliability = as_mapping(value, path, context)
    reject_unknown_fields(reliability, _RELIABILITY_FIELDS, path, context)
    for field_name in sorted(set(reliability) - {"retry_count"}):
        if reliability.get(field_name) not in (None, "", [], {}):
            context.add(
                "unsupported_reliability_policy",
                f"{path}.{field_name}",
                "this transport reliability setting has no Model-runtime equivalent in v0.4",
                "remove it or configure the equivalent deployment/Provider Integration policy",
            )
    retry_count = reliability.get("retry_count")
    if retry_count is None:
        return {}
    return {"max_retries": retry_count}


def _translate_pricing(
    value: Any, model_path: str, context: MigrationContext
) -> tuple[dict[str, str], str | None]:
    if value is None:
        return {}, None
    path = f"{model_path}.pricing"
    pricing = as_mapping(value, path, context)
    reject_unknown_fields(pricing, _PRICING_FIELDS, path, context)
    mapping = {
        "prompt_per_1m": "input_cost_per_million_tokens",
        "completion_per_1m": "output_cost_per_million_tokens",
        "cached_input_per_1m": "cache_read_cost_per_million_tokens",
        "cache_write_per_1m": "cache_write_cost_per_million_tokens",
    }
    result: dict[str, str] = {}
    for old_name, new_name in mapping.items():
        if pricing.get(old_name) is None:
            continue
        rendered = canonical_decimal(
            pricing[old_name], f"{path}.{old_name}", context, positive=False
        )
        if rendered is not None:
            result[new_name] = rendered
    if not result:
        return {}, None
    currency = _optional_trimmed(pricing.get("currency")) or "USD"
    if not re.fullmatch(r"[A-Z]{3}", currency):
        context.add(
            "invalid_billing_currency",
            f"{path}.currency",
            f"currency {currency!r} is not an uppercase ISO-4217 code",
            "use a three-letter code such as USD",
        )
    return result, currency


def _string_mapping(value: Any, path: str, context: MigrationContext) -> dict[str, str]:
    mapping = as_mapping(value, path, context)
    result: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = _optional_trimmed(raw_key)
        item = _optional_trimmed(raw_value)
        if not key or not item:
            context.add(
                "invalid_string_mapping",
                f"{path}.{raw_key}",
                "provider and model IDs must be non-empty strings",
                "use one non-empty string key and value",
            )
            continue
        result[key] = item
    return result


def _required_name(value: Any, path: str, context: MigrationContext) -> str:
    name = _optional_trimmed(value)
    if not name:
        context.add(
            "missing_name",
            path,
            "a non-empty trimmed name is required",
            "supply a stable human-readable name",
        )
        return ""
    return name


def _optional_trimmed(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-._")
    return slug[:127] or "migrated-credential"
