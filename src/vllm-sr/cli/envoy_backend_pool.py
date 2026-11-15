"""Safety contract for projecting backend replica pools into Envoy."""

import ipaddress
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlsplit

from cli.models import BackendRef, Model, UserConfig


@dataclass(frozen=True)
class EnvoyBackendTLS:
    """The TLS session of an https cluster. Unset fields keep the defaults
    the template renders: the endpoint's host and the system CA bundle."""

    server_name: str = ""
    ca_file: str = ""
    cert_file: str = ""
    key_file: str = ""


@dataclass(frozen=True)
class BackendRouteSemantics:
    """Per-endpoint settings that one Envoy route cannot vary after LB."""

    provider: str
    transport: str
    discovery: str
    path: str
    tls_server_name: str
    tls_files: tuple[str, str, str]
    credential: tuple[str, str]
    auth_header: str
    auth_prefix: str
    extra_headers: tuple[tuple[str, str], ...]
    api_version: str
    chat_path: str


@dataclass(frozen=True)
class EnvoyBackendGroup:
    """Materialized endpoints and aggregate transport facts for one alias."""

    endpoints: tuple[dict[str, Any], ...]
    has_https: bool
    uses_dns: bool
    tls: EnvoyBackendTLS | None = None


_SEMANTIC_LABELS = {
    "provider": "provider",
    "transport": "URL scheme",
    "discovery": "DNS/IP discovery",
    "path": "base path",
    "tls_server_name": "TLS server name",
    "tls_files": "TLS files",
    "credential": "credential source",
    "auth_header": "auth header",
    "auth_prefix": "auth prefix",
    "extra_headers": "headers",
    "api_version": "API version",
    "chat_path": "chat path",
}
_HTTP_DEFAULT_PORT = 80
_HTTPS_DEFAULT_PORT = 443
_MAX_ENDPOINT_PORT = 65535
_IPV6_VERSION = 6


def backend_route_semantics(
    backend: BackendRef,
    endpoint: dict[str, Any],
) -> BackendRouteSemantics:
    """Capture settings that must remain constant across one Envoy cluster."""
    tls = endpoint.get("tls") or EnvoyBackendTLS()
    return BackendRouteSemantics(
        provider=backend.provider,
        transport=str(endpoint["protocol"]),
        discovery="dns" if endpoint["is_domain"] else "ip",
        path=str(endpoint["path"]),
        # Envoy 1.34 shares one TLS context and session cache per cluster. Keep
        # that context bound to one server identity; distinct HTTPS hosts must
        # use separate aliases/clusters.
        tls_server_name=(
            (tls.server_name or str(endpoint["address"]))
            if endpoint["is_https"]
            else ""
        ),
        tls_files=(tls.ca_file, tls.cert_file, tls.key_file),
        credential=_credential_identity(backend),
        auth_header=(backend.auth.header if backend.auth else None) or "",
        auth_prefix=(backend.auth.prefix if backend.auth else None) or "",
        extra_headers=tuple(sorted((backend.headers or {}).items())),
        api_version=backend.api_version or "",
        chat_path=backend.chat_path or "",
    )


def default_backend_ca_file(user_config: UserConfig) -> str | None:
    """providers.defaults.tls.ca_file, the CA of https backends naming none."""
    tls = user_config.providers.defaults.tls
    return tls.ca_file if tls else None


def project_envoy_backend_group(
    model: Model, default_ca_file: str | None = None
) -> EnvoyBackendGroup:
    """Parse and validate the backend group exactly once for all CLI paths.

    default_ca_file is providers.defaults.tls.ca_file, the CA of every https
    backend whose own tls names none.
    """

    endpoints: list[dict[str, Any]] = []
    semantics: list[BackendRouteSemantics] = []
    has_https = False
    uses_dns = False
    endpoint_names: set[str] = set()
    for index, backend in enumerate(model.backend_refs):
        endpoint = _project_backend_endpoint(
            model.name, backend, index, default_ca_file
        )
        if endpoint is None:
            continue
        endpoint_name = str(endpoint["name"])
        if endpoint_name in endpoint_names:
            raise ValueError(
                f"providers.models[{model.name!r}].backend_refs[{index}].name "
                f"{endpoint_name!r} is duplicated"
            )
        endpoint_names.add(endpoint_name)
        endpoints.append(endpoint)
        semantics.append(backend_route_semantics(backend, endpoint))
        has_https = has_https or bool(endpoint["is_https"])
        uses_dns = uses_dns or bool(endpoint["is_domain"])

    validate_homogeneous_backend_group(model.name, semantics)
    return EnvoyBackendGroup(
        endpoints=tuple(endpoints),
        has_https=has_https,
        uses_dns=uses_dns,
        tls=next(
            (endpoint["tls"] for endpoint in endpoints if endpoint["is_https"]),
            None,
        ),
    )


def _project_backend_endpoint(
    model_name: str,
    backend: BackendRef,
    index: int,
    default_ca_file: str | None = None,
) -> dict[str, Any] | None:
    endpoint_str = backend.endpoint or ""
    if not endpoint_str:
        return None
    path = ""

    if "://" in endpoint_str:
        parsed = urlsplit(endpoint_str)
        host = parsed.hostname or parsed.netloc
        path = parsed.path.rstrip("/")
        protocol = (parsed.scheme or backend.protocol or "http").strip().lower()
        if parsed.query:
            raise ValueError(
                f"providers.models[{model_name!r}].backend_refs[{index}] "
                "endpoint query parameters are not supported by Envoy routing; "
                "use api_version for provider API-version queries"
            )
        try:
            explicit_port = parsed.port
        except ValueError as port_error:
            raise ValueError(
                f"providers.models[{model_name!r}].backend_refs[{index}] "
                f"has an invalid endpoint port: {port_error}"
            ) from port_error
        port = (
            explicit_port
            if explicit_port is not None
            else _default_port_for_protocol(protocol)
        )
        _validate_endpoint_port(
            port,
            model_name=model_name,
            backend_index=index,
        )
    else:
        protocol = (backend.protocol or "http").strip().lower() or "http"
        endpoint_str, path = _split_endpoint_path(endpoint_str)
        host, port = _split_endpoint_host_port(
            endpoint_str,
            protocol,
            model_name=model_name,
            backend_index=index,
        )

    is_https = protocol == "https"
    is_domain = not is_ip_address(host)
    if is_https and not is_domain:
        raise ValueError(
            f"providers.models[{model_name!r}].backend_refs[{index}] "
            "HTTPS endpoint must use a DNS hostname so Envoy can "
            "verify its certificate identity"
        )
    if backend.tls is not None and not is_https:
        raise ValueError(
            f"providers.models[{model_name!r}].backend_refs[{index}].tls "
            "applies to an https:// endpoint"
        )

    return {
        "name": (backend.name or "").strip() or f"backend-{index + 1}",
        "address": host,
        "port": int(port),
        "host_authority": _host_authority(host, int(port), protocol),
        "path": path,
        "weight": backend.weight,
        "protocol": protocol,
        "is_https": is_https,
        "is_domain": is_domain,
        "extra_headers": dict(backend.headers or {}),
        "tls": _effective_tls(backend, default_ca_file) if is_https else None,
    }


def _effective_tls(backend: BackendRef, default_ca_file: str | None) -> EnvoyBackendTLS:
    """An https backend's own tls, with the default CA when it names none."""
    tls = backend.tls
    return EnvoyBackendTLS(
        server_name=(tls.server_name if tls else None) or "",
        ca_file=(tls.ca_file if tls else None) or default_ca_file or "",
        cert_file=(tls.cert_file if tls else None) or "",
        key_file=(tls.key_file if tls else None) or "",
    )


def _split_endpoint_path(endpoint: str) -> tuple[str, str]:
    if "/" not in endpoint:
        return endpoint, ""
    host, path = endpoint.split("/", 1)
    return host, "/" + path


def _split_endpoint_host_port(
    endpoint: str,
    protocol: str,
    *,
    model_name: str,
    backend_index: int,
) -> tuple[str, int]:
    default_port = _default_port_for_protocol(protocol)
    if endpoint.startswith("["):
        closing_bracket = endpoint.find("]")
        if closing_bracket <= 1:
            raise _invalid_endpoint_authority(model_name, backend_index, endpoint)
        host = endpoint[1:closing_bracket]
        suffix = endpoint[closing_bracket + 1 :]
        if not suffix:
            return host, default_port
        if not suffix.startswith(":") or not suffix[1:]:
            raise _invalid_endpoint_authority(model_name, backend_index, endpoint)
        raw_port = suffix[1:]
    elif endpoint.count(":") > 1:
        raise ValueError(
            f"providers.models[{model_name!r}].backend_refs[{backend_index}] "
            "IPv6 endpoint addresses must be enclosed in brackets"
        )
    elif ":" not in endpoint:
        if not endpoint:
            raise _invalid_endpoint_authority(model_name, backend_index, endpoint)
        return endpoint, default_port
    else:
        host, raw_port = endpoint.rsplit(":", 1)
        if not host or not raw_port:
            raise _invalid_endpoint_authority(model_name, backend_index, endpoint)
    try:
        port = int(raw_port)
    except ValueError as port_error:
        raise ValueError(
            f"providers.models[{model_name!r}].backend_refs[{backend_index}] "
            f"has an invalid endpoint port: {raw_port!r}"
        ) from port_error
    _validate_endpoint_port(
        port,
        model_name=model_name,
        backend_index=backend_index,
    )
    return host, port


def _default_port_for_protocol(protocol: str) -> int:
    return _HTTPS_DEFAULT_PORT if protocol == "https" else _HTTP_DEFAULT_PORT


def _validate_endpoint_port(
    port: int,
    *,
    model_name: str,
    backend_index: int,
) -> None:
    if 1 <= port <= _MAX_ENDPOINT_PORT:
        return
    raise ValueError(
        f"providers.models[{model_name!r}].backend_refs[{backend_index}] "
        f"has an invalid endpoint port: {port}; expected 1..65535"
    )


def _invalid_endpoint_authority(
    model_name: str,
    backend_index: int,
    endpoint: str,
) -> ValueError:
    return ValueError(
        f"providers.models[{model_name!r}].backend_refs[{backend_index}] "
        f"has an invalid endpoint authority: {endpoint!r}"
    )


def _host_authority(host: str, port: int, protocol: str) -> str:
    authority_host = f"[{host}]" if _is_ipv6_address(host) else host
    if (protocol == "http" and port == _HTTP_DEFAULT_PORT) or (
        protocol == "https" and port == _HTTPS_DEFAULT_PORT
    ):
        return authority_host
    return f"{authority_host}:{port}"


def _is_ipv6_address(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).version == _IPV6_VERSION
    except ValueError:
        return False


def is_ip_address(host: str) -> bool:
    """Return whether a backend or internal service host is an IP address."""

    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


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
    if backend.auth is not None and backend.auth.api_key_env:
        return ("environment", backend.auth.api_key_env)
    return ("", "")
