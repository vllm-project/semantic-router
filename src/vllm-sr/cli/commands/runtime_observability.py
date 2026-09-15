"""Project Router tracing defaults onto the local stack's collector."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import yaml

from cli.commands.runtime_paths import (
    read_private_state_bytes,
    recover_runtime_config_projection,
    runtime_config_projection_receipt,
    write_private_state_bytes,
    write_runtime_config_projection,
)
from cli.runtime_stack import RuntimeStackLayout

# This is the Router-owned default, not a general collector-host heuristic.
DEFAULT_LOCAL_JAEGER_ENDPOINT = "vllm-sr-jaeger:4317"


def apply_local_tracing_endpoint(
    config: dict[str, object],
    stack_layout: RuntimeStackLayout,
    *,
    enable_observability: bool = True,
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
    scoped_endpoint = stack_layout.otlp_service_endpoint.removeprefix("http://")
    if endpoint not in (None, DEFAULT_LOCAL_JAEGER_ENDPOINT, scoped_endpoint):
        return False

    # Go's WithEndpoint consumes host:port; the environment counterpart is a URL.
    if enable_observability and endpoint == scoped_endpoint:
        return False
    for parent, key, child in mappings:
        if parent.get(key) is None:
            parent[key] = child
    if enable_observability:
        exporter["endpoint"] = scoped_endpoint
    else:
        # Minimal stacks do not start the local Jaeger collector. Preserve
        # external exporters, but do not export to a service we will not run.
        tracing["enabled"] = False
    return True


def _tracing_block(config: dict[str, object]) -> dict[str, object] | None:
    current = config
    for key in ("global", "services", "observability", "tracing"):
        current = current.get(key)
        if not isinstance(current, dict):
            return None
    return current


def validate_package_tracing_mode(
    runtime_path: Path, stack_layout: RuntimeStackLayout, *, minimal: bool
) -> None:
    """Keep the immutable Recipe's realized config and pointer untouched."""
    if not minimal:
        return
    config = yaml.safe_load(runtime_path.read_bytes()) or {}
    if apply_local_tracing_endpoint(config, stack_layout, enable_observability=False):
        raise ValueError(
            "The active Recipe package uses the local tracing collector, which "
            "--minimal does not start. Use the Recipe deployment workflow to "
            "disable tracing or configure an external collector before serving "
            "this package with --minimal."
        )


def _load_tracing_projection(state_path: Path) -> dict[str, object] | None:
    state_data = read_private_state_bytes(state_path)
    state = json.loads(state_data) if state_data else None
    if state is not None and (
        not isinstance(state, dict)
        or not {"original_enabled", "projected_tracing"} <= set(state)
        or set(state) - {"original_enabled", "projected_tracing", "write"}
        or not isinstance(state["original_enabled"], dict)
        or set(state["original_enabled"]) - {"enabled"}
        or not isinstance(state["projected_tracing"], dict)
        or ("write" in state and not isinstance(state["write"], dict))
    ):
        raise ValueError(f"Invalid local tracing projection state: {state_path}")
    return state


def recover_runtime_tracing_projection(runtime_path: Path) -> None:
    """Recover source provenance before deciding whether source edits apply."""
    state = _load_tracing_projection(runtime_path.with_suffix(".tracing.json"))
    if state is not None and "write" in state:
        recover_runtime_config_projection(runtime_path, state["write"])


def reconcile_runtime_tracing(
    runtime_path: Path,
    stack_layout: RuntimeStackLayout,
    *,
    enable_observability: bool,
    before_replace: Callable[[bytes], None],
    reset_projection: bool = False,
) -> None:
    """Project the chosen active document while holding its runtime config lock.

    Remember only our temporary disable and restore it only while the tracing
    block still matches. A user's collector/exporter or enabled change takes
    ownership back. Unrelated Dashboard edits remain intact and stay divergent
    from source provenance. The receipt precedes disabling and outlives restoring
    the config, so an interrupted write cannot lose an applied override's origin.
    """
    state_path = runtime_path.with_suffix(".tracing.json")
    state = _load_tracing_projection(state_path)
    had_state = state is not None

    # A newly selected source owns even an explicit false equal to our override.
    if reset_projection:
        state = None

    original_data = runtime_path.read_bytes()
    config = yaml.safe_load(original_data) or {}
    tracing = _tracing_block(config)
    matches = state is not None and tracing == state["projected_tracing"]
    restored = False
    if matches and enable_observability:
        tracing.pop("enabled", None)
        tracing.update(state["original_enabled"])
        restored = True
    elif not matches:
        state = None

    original_tracing = deepcopy(_tracing_block(config) or {})
    changed = apply_local_tracing_endpoint(
        config, stack_layout, enable_observability=enable_observability
    )
    if changed and not enable_observability:
        state = {
            "original_enabled": {
                key: value
                for key, value in original_tracing.items()
                if key == "enabled"
            },
            "projected_tracing": deepcopy(_tracing_block(config)),
        }
    if changed or restored:
        projected_data = yaml.dump(
            config, default_flow_style=False, sort_keys=False
        ).encode()
        before_replace(projected_data)
        if state is None:
            state = {
                "original_enabled": {
                    key: value
                    for key, value in original_tracing.items()
                    if key == "enabled"
                },
                "projected_tracing": deepcopy(_tracing_block(config)),
            }
        state["write"] = runtime_config_projection_receipt(runtime_path, projected_data)
        write_private_state_bytes(state_path, json.dumps(state).encode())
        write_runtime_config_projection(runtime_path, projected_data, state["write"])
        had_state = True
    if had_state and (enable_observability or state is None):
        write_private_state_bytes(state_path, b"null\n")
