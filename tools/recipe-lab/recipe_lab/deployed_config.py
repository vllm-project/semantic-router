"""Bind an authored v0.5 recipe to the exact source served by a Docker stack."""

import json
import re
from pathlib import Path

import yaml

from .loop_evidence import check, digest


def _service_defaults(stack):
    # The audited subset of vllm-sr's local service projection. Additional or
    # changed runtime transformations need review before they can qualify a run.
    prefix = f"{stack}-vllm-sr" if stack else "vllm-sr"
    return {
        ("management_api", "bind_address"): "0.0.0.0",
        ("management_api", "port"): 8080,
        ("observability", "tracing", "enabled"): False,
        ("observability", "tracing", "exporter", "endpoint"): f"{prefix}-jaeger:4317",
        ("response_api", "enabled"): True,
        ("response_api", "store_backend"): "redis",
        ("response_api", "redis", "address"): f"{prefix}-redis:6379",
        ("response_api", "redis", "db"): 0,
        ("response_api", "redis", "password"): "${VLLM_SR_STACK_REDIS_PASSWORD}",
        ("router_replay", "store_backend"): "file",
        ("startup_status", "store_backend"): "file",
    }


_ABSENT = object()
_TRACING = ("global", "services", "observability", "tracing")


def _check_projection(source, deployed, defaults, path=(), local_tracing=False):
    if path == _TRACING and isinstance(deployed, dict):
        traced = {} if source is _ABSENT else source
        exporter = traced.get("exporter", {}) if isinstance(traced, dict) else None
        local_tracing = (
            isinstance(exporter, dict)
            and traced.get("enabled") is not False
            and exporter.get("type", "otlp") == "otlp"
            and exporter.get("endpoint")
            in (
                None,
                "vllm-sr-jaeger:4317",
                defaults[("observability", "tracing", "exporter", "endpoint")],
            )
        )
        if local_tracing and deployed.get("enabled") is False:
            traced = dict(traced)
            if "exporter" not in traced and deployed.get("exporter") == {}:
                traced["exporter"] = {}
            if traced.get("enabled") is True:
                traced["enabled"] = False
            source = traced
    if isinstance(deployed, dict) and (isinstance(source, dict) or source is _ABSENT):
        original = {} if source is _ABSENT else source
        if source is _ABSENT:
            check(bool(deployed), "unexpected empty mapping in deployed configuration")
        check(
            original.keys() <= deployed.keys(), "the deployment removed authored fields"
        )
        for key, value in deployed.items():
            _check_projection(
                original.get(key, _ABSENT), value, defaults, (*path, key), local_tracing
            )
        return
    if isinstance(source, list) and isinstance(deployed, list):
        check(len(source) == len(deployed), "the deployment changed an authored list")
        for index, (left, right) in enumerate(zip(source, deployed, strict=False)):
            _check_projection(left, right, defaults, (*path, index), local_tracing)
        return
    expected = source
    if source is _ABSENT and path[:2] == ("global", "services"):
        expected = defaults.get(path[2:], _ABSENT)
        if path[: len(_TRACING)] == _TRACING and not local_tracing:
            expected = _ABSENT
    if (
        local_tracing
        and path == (*_TRACING, "exporter", "endpoint")
        and source == "vllm-sr-jaeger:4317"
        and deployed != source
    ):
        expected = defaults[path[2:]]
    check(
        type(expected) is type(deployed) and expected == deployed,
        "unverified deployment change at " + ".".join(map(str, path)),
    )


def config_identity(authored_path, deployed_path=None):
    """Return authored bytes, authored document and verified deployed source SHA-256."""
    raw = Path(authored_path).read_bytes()
    doc = yaml.safe_load(raw)
    check(
        isinstance(doc, dict) and doc.get("version") == "v0.5",
        "expected v0.5 source YAML",
    )
    if deployed_path is None:
        return raw, doc, digest(raw)
    active = Path(deployed_path)
    name = re.fullmatch(r"runtime-config(?:\.([A-Za-z0-9_.-]+))?\.yaml", active.name)
    check(name is not None, "expected a CLI-owned runtime-config[.STACK].yaml file")
    deployed_raw = active.read_bytes()
    deployed_sha = digest(deployed_raw)
    provenance = json.loads(active.with_suffix(".provenance.json").read_bytes())
    check(
        provenance
        == {
            "schema_version": "vllm-sr/runtime-config-provenance/v1",
            "source_digest": "sha256:" + digest(raw),
            "last_materialized_active_digest": "sha256:" + deployed_sha,
        },
        "CLI provenance does not bind the authored and deployed configuration hashes",
    )
    _check_projection(
        doc, yaml.safe_load(deployed_raw), _service_defaults(name.group(1))
    )
    return raw, doc, deployed_sha
