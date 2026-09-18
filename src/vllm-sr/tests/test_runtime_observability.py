"""Named-stack tracing stays local without rewriting authored configuration."""

import hashlib
import json
from copy import deepcopy

import pytest
import yaml
from cli import core, runtime_lifecycle
from cli.commands import runtime_paths
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
from cli.recipe_activation_recovery import active_recipe_package_for_stack
from cli.recipe_package import recipe_digest
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


def install_active_package(tmp_path, source, active, *, name="test"):
    files = {
        "metadata.yaml": f"schema_version: vllm-sr/recipe-metadata/v1\nid: {name}\n".encode(),
        "config.yaml": source.read_bytes(),
        "probes.yaml": b"schema_version: vllm-sr/recipe-probes/v1\nname: test\n",
        "recipe.dsl": b'recipe "test" {}\n',
        "README.md": b"# Test Recipe\n",
    }
    digest = recipe_digest(files)
    store = tmp_path / ".vllm-sr" / "recipe-store" / "named-trial"
    object_dir = store / "objects" / "sha256" / digest.removeprefix("sha256:")
    object_dir.mkdir(parents=True)
    for filename, data in files.items():
        (object_dir / filename).write_bytes(data)
    pointer = {
        "schema_version": "vllm-sr/recipe-active/v1",
        "recipe_digest": digest,
        "config_digest": "sha256:" + hashlib.sha256(files["config.yaml"]).hexdigest(),
        "realized_config_digest": "sha256:"
        + hashlib.sha256(active.read_bytes()).hexdigest(),
        "activated_at": "2026-09-15T19:00:00Z",
    }
    (store / "active.json").write_text(json.dumps(pointer))
    assert active_recipe_package_for_stack(
        state_root_dir=tmp_path, stack_name="named-trial"
    )


def runtime_snapshot(tmp_path):
    return {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in (tmp_path / ".vllm-sr").rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("enabled", [None, True])
def test_package_local_collector_rejects_minimal_without_mutation(
    tmp_path, monkeypatch, prepare_serve, enabled
):
    source = source_config(tmp_path, {"enabled": enabled} if enabled else None)
    active = prepare_serve(source, minimal=False)
    install_active_package(tmp_path, source, active)
    before = runtime_snapshot(tmp_path)
    stops = []
    monkeypatch.setattr(
        "cli.commands.runtime_serve_config.stop_runtime_before_config_replacement",
        stops.append,
    )
    for minimal in (True, False, True, False):
        if minimal:
            with pytest.raises(ValueError, match="Recipe deployment workflow"):
                prepare_serve(source, minimal=True)
        else:
            prepare_serve(source, minimal=False)
        assert runtime_snapshot(tmp_path) == before
        assert active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="named-trial"
        )
    assert not stops


@pytest.mark.parametrize(
    "tracing",
    [
        {"enabled": False},
        {"enabled": True, "exporter": {"endpoint": "external.example:4317"}},
        {"exporter": {"type": "stdout"}},
    ],
)
def test_package_explicit_tracing_survives_repeated_serve(
    tmp_path, prepare_serve, tracing
):
    source = source_config(tmp_path, tracing)
    active = prepare_serve(source, minimal=False)
    install_active_package(tmp_path, source, active)
    before = runtime_snapshot(tmp_path)
    for minimal in (True, True, False, False):
        prepare_serve(source, minimal=minimal)
        assert runtime_snapshot(tmp_path) == before
        assert active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="named-trial"
        )


def test_new_package_disable_does_not_restore_previous_projection(
    tmp_path, prepare_serve
):
    source = source_config(tmp_path)
    active = prepare_serve(source, minimal=True)
    disabled = deepcopy(tracing_block(yaml.safe_load(active.read_text())))
    source_config(tmp_path, disabled)
    install_active_package(tmp_path, source, active, name="replacement")
    before = runtime_snapshot(tmp_path)
    for minimal in (False, True, False):
        prepare_serve(source, minimal=minimal)
        assert tracing_block(yaml.safe_load(active.read_text())) == disabled
        assert runtime_snapshot(tmp_path) == before
        assert active_recipe_package_for_stack(
            state_root_dir=tmp_path, stack_name="named-trial"
        )


@pytest.mark.parametrize("user_edited", [False, True])
def test_interrupted_projection_recovers_only_cli_owned_provenance(
    tmp_path, monkeypatch, prepare_serve, user_edited
):
    source = source_config(tmp_path)
    active = prepare_serve(source, minimal=False)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(
            runtime_paths,
            "write_private_state_bytes",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("interrupted provenance write")
            ),
        )
        with pytest.raises(OSError, match="interrupted provenance write"):
            prepare_serve(source, minimal=True)
    if user_edited:
        edited = yaml.safe_load(active.read_text())
        edited["global"]["router"] = {"clear_route_cache": True}
        active.write_text(yaml.safe_dump(edited))
    updated = yaml.safe_load(source.read_text())
    updated["global"] = {"router": {"clear_route_cache": False}}
    source.write_text(yaml.safe_dump(updated))
    prepare_serve(source, minimal=True)
    config = yaml.safe_load(active.read_text())
    assert config["global"]["router"]["clear_route_cache"] is user_edited
    receipt = json.loads(_runtime_config_provenance_path(active).read_text())
    current_digest = "sha256:" + hashlib.sha256(active.read_bytes()).hexdigest()
    assert (receipt["last_materialized_active_digest"] != current_digest) == user_edited
    assert tracing_block(config)["enabled"] is False


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


@pytest.fixture
def support_services(tmp_path, monkeypatch):
    """Exercise real mode orchestration against a stateful container boundary."""
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "mode-switch")
    stack = resolve_runtime_stack()
    states = {"external-collector": "running", "other-vllm-sr-jaeger": "running"}
    stopped = []
    storage_starts = []

    def stop(name):
        stopped.append(name)
        states[name] = "exited"
        return True

    def start(name):
        states[name] = "running"
        return 0, "", ""

    def provision(*args, **kwargs):
        storage_starts.append(True)
        return {"redis"}

    def unexpected_remove(*args, **kwargs):
        pytest.fail("Minimal mode must preserve containers and data")

    monkeypatch.setattr(
        core, "container_status_strict", lambda name: states.get(name, "not found")
    )
    monkeypatch.setattr(core, "container_stop_container", stop)
    monkeypatch.setattr(core, "container_remove_container", unexpected_remove)
    monkeypatch.setattr(core, "container_remove_network", unexpected_remove)
    monkeypatch.setattr(core, "provision_storage_backends", provision)
    for service in ("jaeger", "prometheus", "grafana"):
        name = getattr(stack, f"{service}_container_name")
        monkeypatch.setattr(
            runtime_lifecycle,
            f"container_start_{service}",
            lambda *args, name=name, **kwargs: start(name),
        )

    def serve(*, minimal):
        env = {}
        result = core._start_support_services(
            {}, stack.network_name, str(tmp_path), env, stack, not minimal
        )
        assert result == ({"redis"}, stack.network_name)
        assert env[core.MANAGED_STORAGE_BACKENDS_ENV] == "redis"

    return stack, states, stopped, storage_starts, serve


def test_full_minimal_full_stops_only_selected_collectors_and_preserves_data(
    support_services,
):
    stack, states, stopped, _, serve = support_services
    owned = {
        stack.jaeger_container_name,
        stack.prometheus_container_name,
        stack.grafana_container_name,
    }
    serve(minimal=True)
    assert not stopped
    serve(minimal=False)
    assert all(states[name] == "running" for name in owned)
    serve(minimal=True)
    assert set(stopped) == owned
    assert all(states[name] == "exited" for name in owned)
    serve(minimal=True)
    assert len(stopped) == 3
    serve(minimal=False)
    assert all(states[name] == "running" for name in owned)
    assert states["external-collector"] == "running"
    assert states["other-vllm-sr-jaeger"] == "running"


@pytest.mark.parametrize("state", ["paused", "restarting"])
def test_minimal_stops_collectors_in_transient_active_states(support_services, state):
    stack, states, stopped, _, serve = support_services
    states[stack.jaeger_container_name] = state
    serve(minimal=True)
    assert stopped == [stack.jaeger_container_name]
    assert states[stack.jaeger_container_name] == "exited"


@pytest.mark.parametrize(
    "failure", ["stop_failed", "still_running", "inspect_failed", "removing"]
)
def test_minimal_surfaces_unconfirmed_collector_shutdown(
    support_services, monkeypatch, failure
):
    stack, states, _, storage_starts, serve = support_services
    states[stack.jaeger_container_name] = "running"
    if failure == "inspect_failed":

        def inspect_failed(name):
            raise RuntimeError("managed container status inspection failed")

        monkeypatch.setattr(core, "container_status_strict", inspect_failed)
    elif failure == "removing":
        states[stack.jaeger_container_name] = "removing"
    else:
        monkeypatch.setattr(
            core, "container_stop_container", lambda name: failure == "still_running"
        )
    with pytest.raises(RuntimeError, match="container"):
        serve(minimal=True)
    assert not storage_starts
