"""Project Router tracing defaults onto the local stack's collector."""

from __future__ import annotations

from cli.runtime_stack import RuntimeStackLayout

# This is the Router-owned default, not a general collector-host heuristic.
DEFAULT_LOCAL_JAEGER_ENDPOINT = "vllm-sr-jaeger:4317"


def apply_local_tracing_endpoint(
    config: dict[str, object], stack_layout: RuntimeStackLayout
) -> bool:
    """Scope only an omitted or built-in local OTLP endpoint.

    Explicit external endpoints, environment references, disabled tracing and
    other exporters remain authoritative. Call only while materializing a local
    runtime document; the operator's source and target-neutral config stay intact.
    """
    mappings = []
    current = config
    for key in ("global", "services", "observability", "tracing", "exporter"):
        child = current.get(key)
        if child is None:
            child = {}
        if not isinstance(child, dict):
            return False
        mappings.append((current, key, child))
        current = child

    tracing = mappings[-1][0]
    exporter = current
    if tracing.get("enabled") is False or exporter.get("type", "otlp") != "otlp":
        return False
    endpoint = exporter.get("endpoint")
    if endpoint not in (None, DEFAULT_LOCAL_JAEGER_ENDPOINT):
        return False

    # Go's WithEndpoint consumes host:port; the environment counterpart is a URL.
    scoped_endpoint = stack_layout.otlp_service_endpoint.removeprefix("http://")
    if endpoint == scoped_endpoint:
        return False
    for parent, key, child in mappings:
        if parent.get(key) is None:
            parent[key] = child
    exporter["endpoint"] = scoped_endpoint
    return True
