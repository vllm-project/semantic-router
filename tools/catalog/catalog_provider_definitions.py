"""Validation for repository-authored provider definitions."""

from __future__ import annotations

import re
from typing import Any

from catalog_common import ASCII_DELETE, ASCII_SPACE, SLUG, CatalogBuildError
from catalog_common import mapping as _mapping
from catalog_common import nonempty_string as _nonempty_string
from catalog_common import reject_unknown as _reject_unknown
from catalog_common import sequence as _sequence
from catalog_common import validate_https_url as _validate_https_url

PROVIDER_FIELDS = {
    "id",
    "display_name",
    "description",
    "category",
    "support_tier",
    "default_base_url",
    "protocols",
    "default_protocol",
    "supported_operations",
    "path_overrides",
    "default_headers",
    "reasoning_transport",
    "api_version_query",
    "auth",
    "presentation",
    "conformance",
    "models",
}


def validate_providers(
    items: list[dict[str, Any]], protocol_definitions: list[dict[str, Any]]
) -> None:
    """Validate provider identities and their protocol-level runtime contract."""

    protocol_ids = {protocol["id"] for protocol in protocol_definitions}
    for index, item in enumerate(items):
        path = f"providers[{index}]"
        protocols = _validate_provider_identity(item, path, protocol_ids)
        auth = _validate_provider_auth(item, path)
        validate_provider_presentation(item, path, allow_featured=True)
        _validate_provider_conformance(item, path)
        _validate_provider_operations(item, path, protocols, protocol_definitions)
        _validate_provider_headers(item, path, auth)


def _validate_provider_identity(
    item: dict[str, Any], path: str, protocol_ids: set[str]
) -> list[str]:
    _reject_unknown(item, PROVIDER_FIELDS, path)
    identity = _nonempty_string(item.get("id"), f"{path}.id")
    if not SLUG.fullmatch(identity):
        raise CatalogBuildError(f"{path}.id must be a lowercase slug")
    if item.get("category") not in {"start_here", "model_api", "private_runtime"}:
        raise CatalogBuildError(f"{path}.category is unsupported")
    if item.get("support_tier") not in {"native", "compatible", "runtime"}:
        raise CatalogBuildError(f"{path}.support_tier is unsupported")
    protocols = _sequence(item.get("protocols"), f"{path}.protocols")
    if not protocols or any(protocol not in protocol_ids for protocol in protocols):
        raise CatalogBuildError(f"{path}.protocols references an unknown protocol")
    if item.get("default_protocol") not in protocols:
        raise CatalogBuildError(f"{path}.default_protocol must be listed in protocols")
    if item.get("reasoning_transport", "chat_template_kwargs") not in {
        "chat_template_kwargs",
        "top_level_effort",
        "top_level_boolean",
        "top_level_effort_template_switch",
        "top_level_effort_boolean_switch",
        "reasoning_object",
        "thinking_object",
        "thinking_object_effort",
        "output_config_effort",
        "deepseek_thinking",
    }:
        raise CatalogBuildError(f"{path}.reasoning_transport is unsupported")
    default_base_url = item.get("default_base_url")
    if default_base_url is not None:
        _validate_https_url(
            default_base_url,
            f"{path}.default_base_url",
            allow_fragment=False,
        )
    return protocols


def _validate_provider_auth(item: dict[str, Any], path: str) -> dict[str, Any]:
    auth = _mapping(item.get("auth"), f"{path}.auth")
    _reject_unknown(
        auth, {"strategy", "header", "prefix", "injected_header"}, f"{path}.auth"
    )
    if auth.get("strategy") not in {"none", "bearer", "api_key_header"}:
        raise CatalogBuildError(f"{path}.auth.strategy is unsupported")
    return auth


def validate_provider_presentation(
    item: dict[str, Any], path: str, *, allow_featured: bool = False
) -> None:
    """Validate shared provider/model presentation metadata."""

    presentation = _mapping(item.get("presentation"), f"{path}.presentation")
    allowed_fields = {"logo", "monogram", "monochrome"}
    if allow_featured:
        allowed_fields.add("featured")
    _reject_unknown(presentation, allowed_fields, f"{path}.presentation")
    if "featured" in presentation and not isinstance(presentation["featured"], bool):
        raise CatalogBuildError(f"{path}.presentation.featured must be a boolean")
    logo = _nonempty_string(presentation.get("logo"), f"{path}.presentation.logo")
    if not logo.startswith(("package:", "public:", "url:")) and logo != "monogram":
        raise CatalogBuildError(f"{path}.presentation.logo has an unsupported source")
    if logo.startswith("url:") and not logo.startswith("url:https://"):
        raise CatalogBuildError(
            f"{path}.presentation.logo external URLs must use HTTPS"
        )
    if logo.startswith("url:"):
        _validate_https_url(logo.removeprefix("url:"), f"{path}.presentation.logo")


def _validate_provider_conformance(item: dict[str, Any], path: str) -> None:
    conformance = _mapping(item.get("conformance"), f"{path}.conformance")
    _reject_unknown(conformance, {"status", "verified_at"}, f"{path}.conformance")
    status = conformance.get("status")
    if status not in {"unverified", "fixture_verified", "live_verified"}:
        raise CatalogBuildError(f"{path}.conformance.status is unsupported")
    if status != "unverified" and not conformance.get("verified_at"):
        raise CatalogBuildError(f"{path}.conformance.verified_at is required")


def _validate_provider_operations(
    item: dict[str, Any],
    path: str,
    protocols: list[str],
    protocol_definitions: list[dict[str, Any]],
) -> None:
    overrides = item.get("path_overrides", {})
    if not isinstance(overrides, dict):
        raise CatalogBuildError(f"{path}.path_overrides must be a mapping")
    valid_operations = {
        f"{protocol['id']}#{operation['id']}"
        for protocol in protocol_definitions
        if protocol["id"] in protocols
        for operation in protocol["operations"]
    }
    supported = _sequence(
        item.get("supported_operations"), f"{path}.supported_operations"
    )
    if not supported or len(supported) != len(set(supported)):
        raise CatalogBuildError(
            f"{path}.supported_operations references an unknown or duplicate operation"
        )
    if any(operation not in valid_operations for operation in supported):
        raise CatalogBuildError(
            f"{path}.supported_operations references an unknown or duplicate operation"
        )
    missing_create = [
        protocol for protocol in protocols if f"{protocol}#create" not in supported
    ]
    if missing_create:
        raise CatalogBuildError(
            f"{path}.supported_operations must include create for: "
            f"{', '.join(missing_create)}"
        )
    if any(operation not in supported for operation in overrides):
        raise CatalogBuildError(
            f"{path}.path_overrides references an unknown operation"
        )
    for operation, override in overrides.items():
        if (
            not isinstance(override, str)
            or not override.startswith("/")
            or any(
                ord(character) <= ASCII_SPACE or ord(character) == ASCII_DELETE
                for character in override
            )
        ):
            raise CatalogBuildError(
                f"{path}.path_overrides[{operation!r}] must be an absolute path "
                "without whitespace or controls"
            )


def _validate_provider_headers(
    item: dict[str, Any], path: str, auth: dict[str, Any]
) -> None:
    headers = item.get("default_headers", {})
    if not isinstance(headers, dict):
        raise CatalogBuildError(f"{path}.default_headers must be a mapping")
    forbidden = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        str(auth.get("header", "")).lower(),
    }
    seen: set[str] = set()
    for header, value in headers.items():
        _validate_provider_header(header, value, forbidden, path)
        normalized = str(header).lower()
        if normalized in seen:
            raise CatalogBuildError(
                f"{path}.default_headers contains a case-insensitive duplicate"
            )
        seen.add(normalized)


def _validate_provider_header(
    header: Any, value: Any, forbidden: set[str], path: str
) -> None:
    if not isinstance(header, str) or not re.fullmatch(
        r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", header
    ):
        raise CatalogBuildError(
            f"{path}.default_headers contains an invalid header name"
        )
    if header.lower() in forbidden:
        raise CatalogBuildError(
            f"{path}.default_headers cannot contain credential headers"
        )
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(
            (ord(character) < ASCII_SPACE and character != "\t")
            or ord(character) == ASCII_DELETE
            for character in value
        )
    ):
        raise CatalogBuildError(
            f"{path}.default_headers contains an invalid header value"
        )
