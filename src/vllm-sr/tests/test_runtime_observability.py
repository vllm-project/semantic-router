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
from cli.commands.runtime_serve_config import _prepare_docker_runtime_config
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
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http", "address": "127.0.0.1", "port": 8899}],
    }
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
@pytest.mark.parametrize("enable_observability", [True, False])
def test_custom_or_disabled_tracing_is_preserved(
    tmp_path, monkeypatch, tracing, enable_observability
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "named-trial")
    source = source_config(tmp_path, tracing)
    original = source.read_bytes()
    runtime = yaml.safe_load(build_effective_config_bytes(source, None, False, None))
    apply_local_tracing_endpoint(
        runtime, resolve_runtime_stack(), enable_observability=enable_observability
    )
    assert tracing_block(runtime) == tracing
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "endpoint", [None, "vllm-sr-jaeger:4317", "named-trial-vllm-sr-jaeger:4317"]
)
def test_minimal_serve_disables_only_the_unprovisioned_collector(
    tmp_path, monkeypatch, endpoint, prepare_serve
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "named-trial")
    tracing = (
        None
        if endpoint is None
        else {"enabled": True, "exporter": {"endpoint": endpoint}}
    )
    source = source_config(tmp_path, tracing)
    original = source.read_bytes()
    runtime, _, lock = _prepare_docker_runtime_config(
        source, None, False, None, (), False, minimal=True
    )
    try:
        prepared = yaml.safe_load(runtime.read_text())
        assert tracing_block(prepared)["enabled"] is False
        assert source.read_bytes() == original
        snapshot = deepcopy(prepared)
        assert not apply_local_tracing_endpoint(
            prepared, resolve_runtime_stack(), enable_observability=False
        )
        assert prepared == snapshot
    finally:
        lock.close()


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


@pytest.fixture
def prepare_serve(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "named-trial")
    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(
        "cli.commands.runtime_serve_config.stop_runtime_before_config_replacement",
        lambda stack: None,
    )

    def prepare(source, *, minimal):
        runtime, _, lock = _prepare_docker_runtime_config(
            source, None, False, None, (), False, minimal=minimal
        )
        lock.close()
        return runtime

    return prepare


@pytest.mark.parametrize("edited", [False, True])
@pytest.mark.parametrize("first_minimal", [False, True])
def test_tracing_mode_switch_preserves_edits_and_provenance(
    tmp_path, prepare_serve, edited, first_minimal
):
    source = source_config(tmp_path)
    authored = source.read_bytes()
    active = prepare_serve(source, minimal=first_minimal)
    if edited:
        config = yaml.safe_load(active.read_text())
        config["global"]["router"] = {"clear_route_cache": False}
        active.write_text(yaml.safe_dump(config))
    for minimal in (True, True, False, False, True, False):
        prepare_serve(source, minimal=minimal)
        config = yaml.safe_load(active.read_text())
        assert (tracing_block(config).get("enabled") is False) == minimal
        assert tracing_block(config)["exporter"]["endpoint"] == (
            "named-trial-vllm-sr-jaeger:4317"
        )
        receipt = json.loads(_runtime_config_provenance_path(active).read_text())
        active_digest = "sha256:" + hashlib.sha256(active.read_bytes()).hexdigest()
        assert (receipt["last_materialized_active_digest"] != active_digest) == edited
        if edited:
            assert config["global"]["router"]["clear_route_cache"] is False
        assert source.read_bytes() == authored


@pytest.mark.parametrize("enabled", [None, True, False])
def test_package_active_tracing_switches_without_replacing_package(
    tmp_path, monkeypatch, prepare_serve, enabled
):
    tracing = {"enabled": enabled} if enabled is not None else None
    source = source_config(tmp_path, tracing)
    active = prepare_serve(source, minimal=False)
    config = yaml.safe_load(active.read_text())
    config["global"]["router"] = {"clear_route_cache": False}
    active.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "cli.commands.runtime_serve_config.active_recipe_package_for_stack",
        lambda **kwargs: {"name": "active-package"},
    )
    monkeypatch.setattr(
        "cli.commands.runtime_serve_config.active_recipe_package_config_path",
        lambda **kwargs: source,
    )
    for minimal in (True, True, False, False):
        prepare_serve(source, minimal=minimal)
        config = yaml.safe_load(active.read_text())
        assert config["global"]["router"]["clear_route_cache"] is False
        actual_enabled = tracing_block(config).get("enabled")
        assert actual_enabled is (False if minimal else enabled)


@pytest.mark.parametrize(
    "edit",
    [
        {"exporter": {"endpoint": "external.example:4317"}},
        {"exporter": {"endpoint": "${MY_COLLECTOR}"}},
        {"exporter": {"type": "stdout"}},
        {"enabled": True, "exporter": {"endpoint": "external.example:4317"}},
    ],
)
def test_user_tracing_edits_cancel_minimal_restore(tmp_path, prepare_serve, edit):
    source = source_config(tmp_path)
    active = prepare_serve(source, minimal=True)
    config = yaml.safe_load(active.read_text())
    tracing_block(config).update(edit)
    expected = deepcopy(tracing_block(config))
    active.write_text(yaml.safe_dump(config))
    for minimal in (True, False, True):
        prepare_serve(source, minimal=minimal)
        assert tracing_block(yaml.safe_load(active.read_text())) == expected


def test_source_updates_still_replace_an_unedited_minimal_config(
    tmp_path, prepare_serve
):
    source = source_config(tmp_path)
    active = prepare_serve(source, minimal=True)
    config = yaml.safe_load(source.read_text())
    config["global"] = {"router": {"clear_route_cache": False}}
    source.write_text(yaml.safe_dump(config))
    prepare_serve(source, minimal=True)
    assert yaml.safe_load(active.read_text())["global"]["router"] == {
        "clear_route_cache": False
    }
    prepare_serve(source, minimal=False)
    assert tracing_block(yaml.safe_load(active.read_text())).get("enabled") is not False


def test_new_source_disable_takes_ownership_from_minimal_projection(
    tmp_path, prepare_serve
):
    source = source_config(tmp_path)
    active = prepare_serve(source, minimal=True)
    config = yaml.safe_load(source.read_text())
    config["global"] = {
        "services": {
            "observability": {
                "tracing": deepcopy(tracing_block(yaml.safe_load(active.read_text())))
            }
        }
    }
    source.write_text(yaml.safe_dump(config))
    prepare_serve(source, minimal=False)
    assert tracing_block(yaml.safe_load(active.read_text()))["enabled"] is False
