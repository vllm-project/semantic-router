"""Named-stack tracing stays local without rewriting authored configuration."""

import hashlib
import json
from copy import deepcopy

import pytest
import yaml
from cli.commands.runtime_observability import apply_local_tracing_endpoint
from cli.commands.runtime_paths import (
    _runtime_config_provenance_path,
    materialize_runtime_config,
)
from cli.commands.runtime_support import (
    build_effective_config_bytes,
    build_effective_config_document,
    realize_runtime_config,
)
from cli.consts import DEFAULT_STACK_NAME
from cli.runtime_stack import resolve_runtime_stack


def tracing_block(config):
    return config["global"]["services"]["observability"]["tracing"]


def source_config(tmp_path, tracing=None):
    source = tmp_path / "config.yaml"
    config = {"version": "v0.3", "listeners": [{"name": "http", "port": 8899}]}
    if tracing is not None:
        config["global"] = {"services": {"observability": {"tracing": tracing}}}
    source.write_text(yaml.safe_dump(config))
    return source


@pytest.mark.parametrize("endpoint", [None, "vllm-sr-jaeger:4317"])
@pytest.mark.parametrize("stack_name", [DEFAULT_STACK_NAME, "named-trial"])
def test_runtime_realization_scopes_default_collector_without_source_changes(
    tmp_path, monkeypatch, endpoint, stack_name
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", stack_name)
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "700")
    tracing = None if endpoint is None else {"exporter": {"endpoint": endpoint}}
    source = source_config(tmp_path, tracing)
    original = source.read_bytes()
    runtime = tmp_path / "active.yaml"

    realize_runtime_config(source, runtime, package_activation=True)

    expected_host = (
        "vllm-sr-jaeger"
        if stack_name == DEFAULT_STACK_NAME
        else "named-trial-vllm-sr-jaeger"
    )
    assert (
        tracing_block(yaml.safe_load(runtime.read_text()))["exporter"]["endpoint"]
        == f"{expected_host}:4317"
    )
    assert source.read_bytes() == original
    assert runtime.read_bytes() == build_effective_config_bytes(
        source, None, False, None, package_activation=True
    )


@pytest.mark.parametrize(
    "tracing",
    [
        {"exporter": {"endpoint": "collector.example:4317", "insecure": False}},
        {"exporter": {"endpoint": "localhost:4317"}},
        {"exporter": {"endpoint": "other-stack-vllm-sr-jaeger:4317"}},
        {"exporter": {"endpoint": "${MY_COLLECTOR}"}},
        {"enabled": False},
        {"exporter": {"type": "stdout"}},
    ],
)
def test_custom_or_disabled_tracing_is_preserved(tmp_path, monkeypatch, tracing):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "named-trial")
    source = source_config(tmp_path, tracing)
    original = source.read_bytes()
    runtime = yaml.safe_load(build_effective_config_bytes(source, None, False, None))
    assert tracing_block(runtime) == tracing
    assert source.read_bytes() == original


def test_target_neutral_config_has_no_local_tracing_projection(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "named-trial")
    source = source_config(tmp_path)
    config = build_effective_config_document(
        source, None, False, None, materialize_local_runtime=False
    )
    assert "global" not in config


def test_restart_preserves_active_custom_collector_and_source_hash(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "named-trial")
    source = source_config(tmp_path)
    authored = source.read_bytes()
    candidate = build_effective_config_bytes(source, None, False, None)
    active = materialize_runtime_config(source, candidate)
    provenance = json.loads(_runtime_config_provenance_path(active).read_text())
    assert (
        provenance["source_digest"] == "sha256:" + hashlib.sha256(authored).hexdigest()
    )
    assert provenance["last_materialized_active_digest"] == (
        "sha256:" + hashlib.sha256(candidate).hexdigest()
    )
    config = yaml.safe_load(active.read_text())
    tracing_block(config)["exporter"]["endpoint"] = "dashboard-selected.example:4317"
    active.write_text(yaml.safe_dump(config))
    edited = active.read_bytes()

    materialize_runtime_config(source, candidate)

    assert active.read_bytes() == edited
    assert source.read_bytes() == authored
    materialize_runtime_config(source, candidate, replace_active=True)
    assert active.read_bytes() == candidate
    assert source.read_bytes() == authored


def test_restart_refreshes_an_unedited_previous_runtime_default(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "named-trial")
    source = source_config(tmp_path)
    authored = source.read_bytes()
    candidate = build_effective_config_bytes(source, None, False, None)
    previous = yaml.safe_load(candidate)
    tracing_block(previous)["exporter"]["endpoint"] = "vllm-sr-jaeger:4317"
    active = materialize_runtime_config(source, yaml.safe_dump(previous).encode())

    materialize_runtime_config(source, candidate)

    assert active.read_bytes() == candidate
    assert source.read_bytes() == authored


def test_projection_is_idempotent_and_does_not_repair_invalid_shapes():
    stack = resolve_runtime_stack(stack_name="named-trial")
    config = {}
    assert apply_local_tracing_endpoint(config, stack)
    snapshot = deepcopy(config)
    assert not apply_local_tracing_endpoint(config, stack)
    assert config == snapshot
    invalid = {"global": {"services": {"observability": "invalid"}}}
    snapshot = deepcopy(invalid)
    assert not apply_local_tracing_endpoint(invalid, stack)
    assert invalid == snapshot
