"""Safety contract for projecting backend replica pools into Envoy."""

from dataclasses import dataclass, fields
from typing import Any

from cli.models import BackendRef


@dataclass(frozen=True)
class BackendRouteSemantics:
    """Per-endpoint settings that one Envoy route cannot vary after LB."""

    provider: str
    transport: str
    discovery: str
    path: str
    tls_server_name: str
    credential: tuple[str, str]
    auth_header: str
    auth_prefix: str
    extra_headers: tuple[tuple[str, str], ...]
    api_version: str
    chat_path: str


_SEMANTIC_LABELS = {
    "provider": "provider",
    "transport": "URL scheme",
    "discovery": "DNS/IP discovery",
    "path": "base path",
    "tls_server_name": "TLS server name",
    "credential": "credential source",
    "auth_header": "auth header",
    "auth_prefix": "auth prefix",
    "extra_headers": "extra headers",
    "api_version": "API version",
    "chat_path": "chat path",
}


def backend_route_semantics(
    backend: BackendRef,
    endpoint: dict[str, Any],
) -> BackendRouteSemantics:
    """Capture settings that must remain constant across one Envoy cluster."""
    return BackendRouteSemantics(
        provider=backend.provider,
        transport=str(endpoint["protocol"]),
        discovery="dns" if endpoint["is_domain"] else "ip",
        path=str(endpoint["path"]),
        # TLS context is cluster-wide today. Distinct HTTPS names would use the
        # first endpoint's SNI even when Envoy selected another endpoint.
        tls_server_name=(str(endpoint["address"]) if endpoint["is_https"] else ""),
        credential=_credential_identity(backend),
        auth_header=backend.auth_header or "",
        auth_prefix=backend.auth_prefix or "",
        extra_headers=tuple(sorted((backend.extra_headers or {}).items())),
        api_version=backend.api_version or "",
        chat_path=backend.chat_path or "",
    )


def validate_homogeneous_backend_group(
    model_name: str,
    semantics: list[BackendRouteSemantics],
) -> None:
    """Fail closed when endpoint LB cannot preserve provider request semantics."""
    if not semantics[1:]:
        return
    baseline = semantics[0]
    for backend_index, current in enumerate(semantics[1:], start=1):
        differing = [
            _SEMANTIC_LABELS[field.name]
            for field in fields(BackendRouteSemantics)
            if getattr(baseline, field.name) != getattr(current, field.name)
        ]
        if not differing:
            continue
        raise ValueError(
            f"providers.models[{model_name!r}].backend_refs[{backend_index}] "
            "cannot share one Envoy cluster with backend_refs[0]: "
            f"{', '.join(differing)} differ. Split heterogeneous backends into "
            "separate model aliases so provider metadata follows the selected "
            "upstream."
        )


def _credential_identity(backend: BackendRef) -> tuple[str, str]:
    """Compare credential bindings without ever including their value in errors."""
    if backend.api_key:
        return ("inline", backend.api_key)
    if backend.api_key_env:
        return ("environment", backend.api_key_env)
    return ("", "")
