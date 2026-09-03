"""Pure validation and parsing helpers for built-in model catalogs."""

from __future__ import annotations

import re
from typing import Any

from cli.model_catalog_types import (
    CatalogCompatibility,
    CatalogComponentVersions,
    ModelCatalogError,
)

CATALOG_SCHEMA = "vllm-sr/model-catalog/v1"
DEFAULT_CHANNEL = "latest"
SUPPORTED_CONFIG_SCHEMAS = frozenset({"v0.3"})
SUPPORTED_CATALOG_FEATURES = frozenset(
    {"entrypoints", "isolated_recipes", "decision_algorithms"}
)
SUPPORTED_MODEL_KINDS = frozenset({"virtual"})
SUPPORTED_PROTOCOLS = frozenset(
    {"openai_chat", "openai_responses", "anthropic_messages"}
)
_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_CATALOG_VERSION = re.compile(r"^v\d+\.\d+$")
_CATALOG_RESOURCE_VERSION = re.compile(r"^(?:latest|v\d+\.\d+)$")
_SLUG = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_PUBLIC_MODEL_ID = re.compile(r"^vllm-sr/[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_MODEL_REFERENCE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:+-]*(?:/[A-Za-z0-9][A-Za-z0-9._:+-]*)*$"
)
_MAX_SLUG_LENGTH = 128
_MAX_MODEL_REFERENCE_LENGTH = 256

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "catalog_version",
        "channel",
        "release",
        "compatibility",
        "defaults",
        "assets",
        "models",
    }
)
_DEFAULT_KEYS = frozenset({"model", "enabled"})
_COMPATIBILITY_KEYS = frozenset({"cli", "router", "config_schema", "required_features"})
_VERSION_CONSTRAINT_KEYS = frozenset({"min", "max_exclusive"})
_ASSET_KEYS = frozenset({"id", "bundle", "sha256"})
_MODEL_KEYS = frozenset(
    {
        "id",
        "display_name",
        "description",
        "kind",
        "family",
        "generation",
        "policy_version",
        "asset",
        "entrypoint",
        "recipe",
        "protocols",
        "traits",
        "roles",
        "verification",
        "compatibility",
    }
)
_ROLE_KEYS = frozenset(
    {"name", "required", "minimum_candidates", "traits", "recommended_pool"}
)
_VERIFICATION_KEYS = frozenset({"authority", "asset_sha256"})

# Provider-model references inside one isolated recipe. Paths deliberately omit
# list indexes so the recursive walker remains stable as decisions and roles are
# added. Keep this allowlist aligned with the canonical Router schema instead of
# treating arbitrary string values as model references.
_RECIPE_MODEL_REF_OBJECT_LIST_PATHS = frozenset(
    {
        ("routing", "decisions", "modelRefs"),
        ("routing", "decisions", "candidateIterations", "models"),
    }
)
_RECIPE_MODEL_REF_SCALAR_PATHS = frozenset(
    {
        ("routing", "signals", "classifiers", "model"),
        ("routing", "decisions", "algorithm", "prompt", "model"),
        ("routing", "decisions", "algorithm", "remom", "synthesis_model"),
        ("routing", "decisions", "algorithm", "fusion", "model"),
        (
            "routing",
            "decisions",
            "algorithm",
            "fusion",
            "analysis_overrides",
            "model",
        ),
        (
            "routing",
            "decisions",
            "algorithm",
            "workflows",
            "planner",
            "model",
        ),
        (
            "routing",
            "decisions",
            "algorithm",
            "workflows",
            "final",
            "model",
        ),
        ("routing", "decisions", "plugins", "configuration", "model"),
        (
            "routing",
            "decisions",
            "plugins",
            "configuration",
            "backend_config",
            "model",
        ),
        ("routing", "decisions", "plugins", "configuration", "ar_model"),
        (
            "routing",
            "decisions",
            "plugins",
            "configuration",
            "diffusion_model",
        ),
    }
)
_RECIPE_MODEL_REF_STRING_LIST_PATHS = frozenset(
    {
        ("routing", "decisions", "algorithm", "fusion", "analysis_models"),
        (
            "routing",
            "decisions",
            "algorithm",
            "workflows",
            "roles",
            "models",
        ),
    }
)

_SECRET_KEY_TERMS = frozenset(
    {
        "authorization",
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
    }
)
_SECRET_LITERAL_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)https?://[^\s/:@]+:[^\s/@]+@"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bxox(?:b|p|a|r|s)-[A-Za-z0-9-]{16,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
        r"\s*[:=]\s*[^\s,;]{8,}"
    ),
)


def _compatibility(
    value: Any, component_versions: CatalogComponentVersions
) -> CatalogCompatibility:
    mapping = value if isinstance(value, dict) else {}

    for component in ("cli", "router"):
        constraint = mapping.get(component)
        if constraint is None:
            continue
        if not isinstance(constraint, dict):
            raise ModelCatalogError(
                f"catalog compatibility {component} must be a mapping"
            )
        current_text = getattr(component_versions, component)
        current = _version_tuple(current_text)
        if current is None:
            return CatalogCompatibility(
                False, f"installed {component} version is unknown"
            )
        minimum = str(constraint.get("min") or "").strip()
        maximum = str(constraint.get("max_exclusive") or "").strip()
        if minimum and current < _required_version_tuple(minimum):
            return CatalogCompatibility(False, f"requires {component} >= {minimum}")
        if maximum and current >= _required_version_tuple(maximum):
            return CatalogCompatibility(False, f"requires {component} < {maximum}")

    schema = str(mapping.get("config_schema") or "").strip()
    if schema and schema not in SUPPORTED_CONFIG_SCHEMAS:
        return CatalogCompatibility(False, f"requires config schema {schema}")
    features = mapping.get("required_features") or []
    if not isinstance(features, list) or any(
        not isinstance(feature, str) or not feature.strip() for feature in features
    ):
        raise ModelCatalogError(
            "catalog compatibility required_features must be a string list"
        )
    missing_features = sorted(set(features) - SUPPORTED_CATALOG_FEATURES)
    if missing_features:
        return CatalogCompatibility(
            False, "missing features: " + ", ".join(missing_features)
        )
    return CatalogCompatibility(True, "compatible")


def _merge_compatibility(base: Any, override: Any) -> dict[str, Any]:
    base_mapping = dict(base) if isinstance(base, dict) else {}
    if override is None:
        return base_mapping
    if not isinstance(override, dict):
        raise ModelCatalogError("catalog model compatibility must be a mapping")
    merged = dict(base_mapping)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _validate_catalog_model_role_pool(
    model_id: str,
    recipe_name: str,
    roles: tuple[dict[str, Any], ...],
    asset_document: dict[str, Any],
) -> None:
    recipes = asset_document.get("recipes")
    if not isinstance(recipes, list):
        raise ModelCatalogError("built-in model asset recipes are invalid")
    matching_recipes = [
        recipe
        for recipe in recipes
        if isinstance(recipe, dict) and recipe.get("name") == recipe_name
    ]
    if len(matching_recipes) != 1:
        raise ModelCatalogError(
            f"catalog model {model_id} references a missing packaged recipe"
        )

    providers = asset_document.get("providers")
    if providers is None:
        provider_models: list[Any] = []
    elif isinstance(providers, dict) and isinstance(providers.get("models"), list):
        provider_models = providers["models"]
    else:
        raise ModelCatalogError("built-in model asset providers are invalid")
    provider_names = {
        item.get("name")
        for item in provider_models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    actual_references = _collect_recipe_provider_model_references(
        matching_recipes[0], provider_names
    )
    recommended_references = {
        reference for role in roles for reference in role["recommended_pool"]
    }
    missing = sorted(actual_references - recommended_references)
    if missing:
        raise ModelCatalogError(
            f"catalog model {model_id} roles omit recipe model references: "
            + ", ".join(missing)
        )


def _collect_recipe_provider_model_references(
    recipe: dict[str, Any], provider_names: set[str]
) -> set[str]:
    references: set[str] = set()
    _walk_recipe_provider_model_references(
        recipe, (), provider_names=provider_names, references=references
    )
    return references


def _walk_recipe_provider_model_references(
    value: Any,
    path: tuple[str, ...],
    *,
    provider_names: set[str],
    references: set[str],
) -> None:
    if path in _RECIPE_MODEL_REF_OBJECT_LIST_PATHS:
        _collect_object_list_model_references(value, provider_names, references)
        return
    if path in _RECIPE_MODEL_REF_SCALAR_PATHS:
        _add_provider_model_reference(value, provider_names, references)
        return
    if path in _RECIPE_MODEL_REF_STRING_LIST_PATHS:
        _collect_string_list_model_references(value, provider_names, references)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                _walk_recipe_provider_model_references(
                    item,
                    (*path, key),
                    provider_names=provider_names,
                    references=references,
                )
        return
    if isinstance(value, list):
        for item in value:
            _walk_recipe_provider_model_references(
                item,
                path,
                provider_names=provider_names,
                references=references,
            )


def _collect_object_list_model_references(
    value: Any, provider_names: set[str], references: set[str]
) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, dict):
            _add_provider_model_reference(item.get("model"), provider_names, references)


def _collect_string_list_model_references(
    value: Any, provider_names: set[str], references: set[str]
) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        _add_provider_model_reference(item, provider_names, references)


def _add_provider_model_reference(
    reference: Any, provider_names: set[str], references: set[str]
) -> None:
    if isinstance(reference, str) and reference in provider_names:
        references.add(reference)


def _parse_roles(value: Any, model_id: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ModelCatalogError(f"catalog model {model_id} has no backend roles")
    roles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ModelCatalogError(f"catalog model {model_id} role is invalid")
        _reject_unknown_keys(item, _ROLE_KEYS, f"model {model_id} role")
        name = _required_slug(item, "name")
        if name in seen or not isinstance(item.get("required"), bool):
            raise ModelCatalogError(f"catalog model {model_id} role is invalid")
        seen.add(name)
        minimum = _required_positive_int(item, "minimum_candidates")
        traits = _unique_slugs(item.get("traits"), f"{model_id}.{name}.traits")
        recommended = _unique_model_references(
            item.get("recommended_pool"), f"{model_id}.{name}.recommended_pool"
        )
        if minimum > len(recommended):
            raise ModelCatalogError(
                f"catalog model {model_id} role {name} minimum_candidates "
                "exceeds its recommended pool"
            )
        roles.append(
            {
                "name": name,
                "required": item["required"],
                "minimum_candidates": minimum,
                "traits": list(traits),
                "recommended_pool": list(recommended),
            }
        )
    if not any(role["required"] for role in roles):
        raise ModelCatalogError(
            f"catalog model {model_id} must declare at least one required role"
        )
    return tuple(roles)


def _validate_catalog_binding(
    resource_version: str,
    catalog_version: str,
    channel: str,
    release: str,
) -> None:
    if resource_version == DEFAULT_CHANNEL:
        if (
            channel != "latest"
            or catalog_version != "latest"
            or release != "unreleased"
        ):
            raise ModelCatalogError(
                "latest catalog must use latest channel/version and unreleased binding"
            )
        return
    if (
        channel != "release"
        or resource_version != catalog_version
        or release != catalog_version
        or not _CATALOG_VERSION.fullmatch(catalog_version)
    ):
        raise ModelCatalogError(
            "release catalog resource, channel, release, and catalog_version must match"
        )


def _validate_compatibility_mapping(
    value: Any, label: str, *, require_complete: bool = False
) -> None:
    if not isinstance(value, dict):
        raise ModelCatalogError(f"catalog {label} must be a mapping")
    _reject_unknown_keys(value, _COMPATIBILITY_KEYS, label)
    missing = _COMPATIBILITY_KEYS - set(value)
    if require_complete and missing:
        raise ModelCatalogError(
            f"catalog {label} is missing fields: {', '.join(sorted(missing))}"
        )
    for component in ("cli", "router"):
        _validate_version_constraint(value.get(component), label, component)
    _validate_config_schema(value.get("config_schema"), label)
    features = value.get("required_features")
    if features is not None:
        _unique_slugs(features, f"{label}.required_features")


def _validate_version_constraint(value: Any, label: str, component: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ModelCatalogError(f"catalog {label}.{component} must be a mapping")
    constraint_label = f"{label}.{component}"
    _reject_unknown_keys(value, _VERSION_CONSTRAINT_KEYS, constraint_label)
    minimum = value.get("min")
    maximum = value.get("max_exclusive")
    if minimum is not None:
        _require_semver_value(minimum, f"{constraint_label}.min")
    if maximum is not None:
        _require_semver_value(maximum, f"{constraint_label}.max_exclusive")
    if (
        minimum is not None
        and maximum is not None
        and _required_version_tuple(minimum) >= _required_version_tuple(maximum)
    ):
        raise ModelCatalogError(f"catalog {constraint_label} version range is empty")


def _validate_config_schema(schema: Any, label: str) -> None:
    if schema is not None and (
        not isinstance(schema, str) or not _CATALOG_VERSION.fullmatch(schema)
    ):
        raise ModelCatalogError(
            f"catalog {label}.config_schema must use vMAJOR.MINOR syntax"
        )


def _validate_manifest_security(value: Any, path: str = "catalog") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ModelCatalogError(
                    f"catalog manifest field name at {path} must be a string"
                )
            child_path = f"{path}.{key}"
            if _is_secret_like_key(key):
                raise ModelCatalogError(
                    f"catalog manifest contains secret-like field: {child_path}"
                )
            _validate_manifest_security(child, child_path)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_manifest_security(child, f"{path}[{index}]")
        return
    if isinstance(value, str) and any(
        pattern.search(value) for pattern in _SECRET_LITERAL_PATTERNS
    ):
        raise ModelCatalogError(
            f"catalog manifest contains credential-like literal at {path}"
        )


def _is_secret_like_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    terms = set(normalized.split("_"))
    if terms & _SECRET_KEY_TERMS:
        return True
    return normalized in {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "private_key",
        "client_secret",
        "token",
    } or normalized.endswith(("_token", "_private_key"))


def _reject_unknown_keys(
    value: dict[Any, Any], allowed: frozenset[str], label: str
) -> None:
    non_strings = [key for key in value if not isinstance(key, str)]
    if non_strings:
        raise ModelCatalogError(f"catalog {label} field names must be strings")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ModelCatalogError(
            f"catalog {label} contains unknown fields: {', '.join(unknown)}"
        )


def _required_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ModelCatalogError(f"catalog field {key} must be a mapping")
    return item


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ModelCatalogError(f"catalog field {key} must be a non-empty string")
    return item.strip()


def _required_catalog_identity(value: dict[str, Any], key: str, *, special: str) -> str:
    item = _required_string(value, key)
    if item != special and not _CATALOG_VERSION.fullmatch(item):
        raise ModelCatalogError(
            f"catalog field {key} must use {special!r} or vMAJOR.MINOR syntax"
        )
    return item


def _required_semver(value: dict[str, Any], key: str) -> str:
    item = _required_string(value, key)
    _require_semver_value(item, key)
    return item


def _require_semver_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or _version_tuple(value) is None:
        raise ModelCatalogError(
            f"catalog field {label} must use semantic version syntax"
        )
    return value


def _required_slug(value: dict[str, Any], key: str) -> str:
    item = _required_string(value, key)
    if len(item) > _MAX_SLUG_LENGTH or not _SLUG.fullmatch(item):
        raise ModelCatalogError(f"catalog field {key} must be a lowercase slug")
    return item


def _required_public_model_id(value: dict[str, Any], key: str) -> str:
    item = _required_string(value, key)
    if len(item) > _MAX_MODEL_REFERENCE_LENGTH or not _PUBLIC_MODEL_ID.fullmatch(item):
        raise ModelCatalogError(
            f"catalog field {key} must be a vllm-sr public model ID"
        )
    return item


def _required_enum(value: dict[str, Any], key: str, allowed: frozenset[str]) -> str:
    item = _required_string(value, key)
    if item not in allowed:
        raise ModelCatalogError(f"catalog field {key} has unsupported value: {item}")
    return item


def _required_positive_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item < 1:
        raise ModelCatalogError(f"catalog field {key} must be a positive integer")
    return item


def _unique_strings(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ModelCatalogError(
            f"catalog field {label} must be a non-empty string list"
        )
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise ModelCatalogError(f"catalog field {label} contains duplicates")
    return normalized


def _unique_model_ids(value: Any, label: str) -> tuple[str, ...]:
    items = _unique_strings(value, label)
    if any(
        len(item) > _MAX_MODEL_REFERENCE_LENGTH or not _PUBLIC_MODEL_ID.fullmatch(item)
        for item in items
    ):
        raise ModelCatalogError(
            f"catalog field {label} must contain vllm-sr public model IDs"
        )
    return items


def _unique_slugs(value: Any, label: str) -> tuple[str, ...]:
    items = _unique_strings(value, label)
    if any(len(item) > _MAX_SLUG_LENGTH or not _SLUG.fullmatch(item) for item in items):
        raise ModelCatalogError(f"catalog field {label} must contain lowercase slugs")
    return items


def _unique_model_references(value: Any, label: str) -> tuple[str, ...]:
    items = _unique_strings(value, label)
    if any(
        len(item) > _MAX_MODEL_REFERENCE_LENGTH or not _MODEL_REFERENCE.fullmatch(item)
        for item in items
    ):
        raise ModelCatalogError(
            f"catalog field {label} must contain safe model references"
        )
    return items


def _unique_enum_strings(
    value: Any, label: str, allowed: frozenset[str]
) -> tuple[str, ...]:
    items = _unique_strings(value, label)
    unsupported = sorted(set(items) - allowed)
    if unsupported:
        raise ModelCatalogError(
            f"catalog field {label} contains unsupported values: "
            + ", ".join(unsupported)
        )
    return items


SemVerPreRelease = tuple[tuple[int, int | str], ...]
SemVerKey = tuple[int, int, int, int, SemVerPreRelease]


def _version_tuple(value: str) -> SemVerKey | None:
    match = _SEMVER.fullmatch(value)
    if match is None:
        return None
    major, minor, patch, prerelease = match.groups()
    identifiers: list[tuple[int, int | str]] = []
    if prerelease is not None:
        for identifier in prerelease.split("."):
            if identifier.isdigit():
                if len(identifier) > 1 and identifier.startswith("0"):
                    return None
                identifiers.append((0, int(identifier)))
            else:
                identifiers.append((1, identifier))
    # A release has higher precedence than every prerelease with the same core.
    release_rank = 1 if prerelease is None else 0
    return (
        int(major),
        int(minor),
        int(patch),
        release_rank,
        tuple(identifiers),
    )


def _required_version_tuple(value: str) -> SemVerKey:
    parsed = _version_tuple(value)
    if parsed is None:
        raise ModelCatalogError(f"catalog compatibility version is invalid: {value}")
    return parsed
