"""`vllm-sr config apply` against the local stack (#4698, #4699)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from cli import (
    container_backend,
    core,
    local_stack_config,
    runtime_lifecycle,
    runtime_service_status,
)
from cli.commands import config_management, runtime_serve_config
from cli.commands.runtime_paths import pending_restart_path
from cli.local_stack_config import LocalStack, save_pending_restart
from cli.main import main
from cli.pending_activation import ORIGIN_CLI, read_pending_activation, record_file
from cli.router_management_client import RouterManagementError, RouterResponse
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner

MOVED_PORT = 8890
WARMUP_SECONDS = 120

SOURCE = f"""\
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: {MOVED_PORT}
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: host.docker.internal:8000
recipes:
  - name: default
"""


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalStack:
    (tmp_path / "config.yaml").write_text(SOURCE, encoding="utf-8")
    state_dir = tmp_path / ".vllm-sr"
    state_dir.mkdir()
    stack = LocalStack(
        layout=resolve_runtime_stack(),
        state_dir=state_dir,
        algorithm=None,
    )
    _publish_management_endpoint(monkeypatch, stack, 8080, 8080)
    return stack


def _publish_management_endpoint(
    monkeypatch, stack, container_port, host_port, host_ip="127.0.0.1"
):
    def management_port(container_name):
        assert container_name == stack.layout.router_container_name
        return container_port

    def ports(container_name):
        assert container_name == stack.layout.router_container_name
        return {
            f"{container_port}/tcp": [{"HostIp": host_ip, "HostPort": str(host_port)}],
            "8899/tcp": [{"HostIp": "0.0.0.0", "HostPort": "18899"}],
        }

    monkeypatch.setattr(local_stack_config, "_router_management_port", management_port)
    monkeypatch.setattr(local_stack_config, "_container_ports", ports)


def _set_management_port(config_path: Path, port: int) -> None:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document.setdefault("global", {}).setdefault("services", {})["management_api"] = {
        "bind_address": "0.0.0.0",
        "port": port,
    }
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")


def _staged_document(staged: Path) -> dict:
    """The config a refused apply staged for the next serve."""

    return yaml.safe_load(json.loads(staged.read_text(encoding="utf-8"))["config"])


class FakeClient:
    """Records what the CLI sends; answers with what the test scripted."""

    def __init__(self, *, plan=None, apply=None):
        self.sent: list[str] = []
        self._plan = plan or (lambda: {"changed": True, "current_etag": '"v1"'})
        self._apply = apply or (lambda: {"version": 2})

    def plan_config(self, yaml_text, mode):
        self.sent.append(yaml_text)
        return RouterResponse(payload=self._plan())

    def apply_config(self, yaml_text, mode, etag):
        self.sent.append(yaml_text)
        return RouterResponse(payload=self._apply())


def _apply(monkeypatch, stack, client, config_path: Path, *extra: str):
    return _config_command(monkeypatch, stack, client, config_path, "apply", *extra)


def _config_command(monkeypatch, stack, client, config_path, command, *extra):
    monkeypatch.setattr(config_management, "local_stack", lambda: stack)
    monkeypatch.setattr(config_management, "_client", lambda *_args: client)
    return CliRunner().invoke(
        main, ["config", command, "--config", str(config_path), *extra]
    )


# #4698: what apply sends is what serve materializes, so the Router keeps the
# stack's management listener and a later serve finds its own document.
def test_apply_sends_the_stack_wiring_serve_adds(monkeypatch, stack, tmp_path):
    client = FakeClient()
    result = _apply(monkeypatch, stack, client, tmp_path / "config.yaml")

    assert result.exit_code == 0, result.output
    sent = yaml.safe_load(client.sent[-1])
    assert sent["global"]["services"]["management_api"] == {
        "bind_address": "0.0.0.0",
        "port": 8080,
    }
    assert client.sent[0] == client.sent[1]


def test_an_explicit_endpoint_of_the_local_stack_still_gets_its_wiring(
    monkeypatch, stack, tmp_path
):
    client = FakeClient()
    result = _apply(
        monkeypatch,
        stack,
        client,
        tmp_path / "config.yaml",
        "--endpoint",
        "http://127.0.0.1:8080",
    )

    assert result.exit_code == 0, result.output
    assert "management_api" in yaml.safe_load(client.sent[-1])["global"]["services"]


@pytest.mark.parametrize("command", ["plan", "apply"])
@pytest.mark.parametrize(
    ("management_port", "endpoint_path"),
    [
        (8080, "/api/v1"),
        (8080, "/api/v1/"),
        (8081, ""),
        (8081, "/api/v1"),
        (8081, "/api/v1/"),
    ],
)
def test_custom_management_endpoint_keeps_local_runtime_wiring(
    monkeypatch, stack, tmp_path, command, management_port, endpoint_path
):
    source = tmp_path / "config.yaml"
    _set_management_port(source, management_port)
    _publish_management_endpoint(monkeypatch, stack, management_port, management_port)
    client = FakeClient()

    result = _config_command(
        monkeypatch,
        stack,
        client,
        source,
        command,
        "--endpoint",
        f"http://127.0.0.1:{management_port}{endpoint_path}",
    )

    assert result.exit_code == 0, result.output
    sent = yaml.safe_load(client.sent[-1])
    assert sent["global"]["services"]["management_api"]["port"] == management_port
    tracing = sent["global"]["services"]["observability"]["tracing"]
    assert tracing["exporter"]["endpoint"] == (
        stack.layout.otlp_service_endpoint.removeprefix("http://")
    )


@pytest.mark.parametrize(
    ("offset", "endpoint_path"), [(0, ""), (1000, ""), (1000, "/api/v1/")]
)
def test_custom_endpoint_stages_each_pending_change_at_the_running_port(
    monkeypatch, stack, tmp_path, offset, endpoint_path
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "custom-management")
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", str(offset))
    stack = replace(stack, layout=resolve_runtime_stack())
    source = tmp_path / "config.yaml"
    _set_management_port(source, 8082)
    _publish_management_endpoint(monkeypatch, stack, 8081, 8081 + offset)

    def refuse():
        raise RouterManagementError(
            "HTTP 409", status=409, code="RESTART_REQUIRED", detail="listeners changed"
        )

    client = FakeClient(plan=refuse)
    for _ in range(2):
        result = _apply(
            monkeypatch,
            stack,
            client,
            source,
            "--endpoint",
            f"http://127.0.0.1:{8081 + offset}{endpoint_path}",
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["status"] == "restart_required"
        assert "run `vllm-sr serve`" in result.stderr
        pending = read_pending_activation(stack.runtime_config())
        assert pending is not None and pending.origin == ORIGIN_CLI
        saved = _staged_document(pending_restart_path(stack.runtime_config()))
        assert saved["global"]["services"]["management_api"]["port"] == 8082
        assert not stack.runtime_config().exists()

    assert len(client.sent) == 2


def test_custom_endpoint_checks_source_ownership(monkeypatch, stack, tmp_path):
    elsewhere = tmp_path / "other" / "config.yaml"
    elsewhere.parent.mkdir()
    elsewhere.write_text(SOURCE, encoding="utf-8")
    _publish_management_endpoint(monkeypatch, stack, 8081, 8081)
    client = FakeClient()

    result = _apply(
        monkeypatch,
        stack,
        client,
        elsewhere,
        "--endpoint",
        "http://127.0.0.1:8081",
    )

    assert result.exit_code == 1
    assert "--replace-active-config" in result.output
    assert client.sent == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://router.example:8081",
        "http://127.0.0.1:8082",
        "http://127.0.0.1:18899",
        "https://127.0.0.1:8081",
        "http://127.0.0.1:8081/proxy",
        "http://127.0.0.1:8081/api/v1/proxy",
    ],
)
def test_other_endpoints_keep_the_source_document(
    monkeypatch, stack, tmp_path, endpoint
):
    _publish_management_endpoint(monkeypatch, stack, 8081, 8081)
    client = FakeClient()

    result = _apply(
        monkeypatch, stack, client, tmp_path / "config.yaml", "--endpoint", endpoint
    )

    assert result.exit_code == 0, result.output
    assert client.sent[-1] == SOURCE


def test_merge_mode_does_not_inspect_local_endpoint(monkeypatch, stack, tmp_path):
    def unexpected(_name):
        pytest.fail("merge mode must not inspect the local management port")

    monkeypatch.setattr(local_stack_config, "_router_management_port", unexpected)
    client = FakeClient()

    result = _apply(
        monkeypatch,
        stack,
        client,
        tmp_path / "config.yaml",
        "--mode",
        "merge",
        "--endpoint",
        "http://127.0.0.1:8081",
    )

    assert result.exit_code == 0, result.output
    assert client.sent[-1] == SOURCE


def test_unknown_management_publication_keeps_remote_restart_handling(
    monkeypatch, stack, tmp_path
):
    _publish_management_endpoint(monkeypatch, stack, 8081, 8081)
    monkeypatch.setattr(local_stack_config, "_container_ports", lambda _name: {})

    def refuse():
        raise RouterManagementError(
            "HTTP 409", status=409, code="RESTART_REQUIRED", detail="listeners changed"
        )

    result = _apply(
        monkeypatch,
        stack,
        FakeClient(plan=refuse),
        tmp_path / "config.yaml",
        "--endpoint",
        "http://127.0.0.1:8081",
    )

    assert result.exit_code == 1
    assert "helm upgrade" in result.output
    assert not stack.runtime_config().exists()


def test_apply_to_another_router_sends_the_file_as_written(
    monkeypatch, stack, tmp_path
):
    client = FakeClient()
    result = _apply(
        monkeypatch,
        stack,
        client,
        tmp_path / "config.yaml",
        "--endpoint",
        "http://router.example:8080",
    )

    assert result.exit_code == 0, result.output
    assert client.sent[-1] == SOURCE


def test_apply_of_a_file_outside_the_stack_says_how_to_serve_it(
    monkeypatch, stack, tmp_path
):
    elsewhere = tmp_path / "other" / "config.yaml"
    elsewhere.parent.mkdir()
    elsewhere.write_text(SOURCE, encoding="utf-8")

    result = _apply(monkeypatch, stack, FakeClient(), elsewhere)

    assert result.exit_code == 1
    assert "--replace-active-config" in result.output


# #4699: a change the Router can't take is saved for the next serve.
def test_restart_required_is_saved_for_the_next_serve(monkeypatch, stack, tmp_path):
    def refuse():
        raise RouterManagementError(
            "Router management API returned HTTP 409: RESTART_REQUIRED: listeners changed",
            status=409,
            code="RESTART_REQUIRED",
            detail="listeners changed; standalone mode binds its listeners at startup",
        )

    saved = []
    monkeypatch.setattr(
        config_management,
        "save_pending_restart",
        lambda *args, **_kwargs: saved.append(args) or tmp_path / "runtime-config.yaml",
    )
    result = _apply(
        monkeypatch, stack, FakeClient(plan=refuse), tmp_path / "config.yaml"
    )

    assert result.exit_code == 0, result.output
    assert "Restart required: run `vllm-sr serve` to apply." in result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "restart_required" and report["applied"] is False
    assert report["reason"].startswith("listeners changed")
    assert saved and saved[0][1] == tmp_path / "config.yaml"


def test_restart_required_without_a_local_stack_names_the_restart(
    monkeypatch, tmp_path
):
    (tmp_path / "config.yaml").write_text(SOURCE, encoding="utf-8")

    def refuse():
        raise RouterManagementError(
            "HTTP 409", status=409, code="RESTART_REQUIRED", detail="listeners changed"
        )

    result = _apply(
        monkeypatch, None, FakeClient(plan=refuse), tmp_path / "config.yaml"
    )

    assert result.exit_code == 1
    assert "Restart required: listeners changed" in result.output
    assert "Envoy" not in result.output


@pytest.mark.parametrize(
    ("new_version", "edit_during_save", "preserve_mtime"),
    [
        ("v0.5", False, False),
        ("v99.0", False, False),
        ("v0.5", True, False),
        ("v0.5", False, True),
    ],
)
def test_restart_required_rejects_a_source_change(
    monkeypatch, stack, tmp_path, new_version, edit_during_save, preserve_mtime
):
    source = tmp_path / "config.yaml"
    original_stat = source.stat()
    runtime = stack.runtime_config()
    previous_config = SOURCE.replace(f"port: {MOVED_PORT}", "port: 8899").encode()
    runtime.write_bytes(previous_config)

    def edit_source():
        source.write_text(
            SOURCE.replace(f"port: {MOVED_PORT}", "port: 8891").replace(
                "version: v0.5", f"version: {new_version}"
            ),
            encoding="utf-8",
        )
        if preserve_mtime:
            os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    def refuse():
        if edit_during_save:
            build = local_stack_config.build_effective_config_bytes

            def build_after_edit(*args, **kwargs):
                edit_source()
                return build(*args, **kwargs)

            monkeypatch.setattr(
                local_stack_config, "build_effective_config_bytes", build_after_edit
            )
        else:
            edit_source()
        raise RouterManagementError("HTTP 409", status=409, code="RESTART_REQUIRED")

    client = FakeClient(plan=refuse)
    result = _apply(monkeypatch, stack, client, source)

    assert len(client.sent) == 1
    planned = yaml.safe_load(client.sent[0])
    assert planned["listeners"][0]["port"] == 8890
    assert planned["version"] == "v0.5"
    assert result.exit_code == 1, result.output
    assert "Config source changed" in result.output
    assert runtime.read_bytes() == previous_config
    assert read_pending_activation(runtime) is None
    assert not runtime.with_suffix(".provenance.json").exists()
    assert not pending_restart_path(runtime).exists()


def test_a_timed_out_apply_says_the_change_may_still_activate(
    monkeypatch, stack, tmp_path
):
    def slow():
        raise RouterManagementError(
            "Router management API request timed out after 120s", timed_out=True
        )

    result = _apply(
        monkeypatch, stack, FakeClient(apply=slow), tmp_path / "config.yaml"
    )

    assert result.exit_code == 1
    assert "may still activate" in result.output
    assert "vllm-sr config versions" in result.output


def test_apply_waits_for_model_warmup_by_default():
    timeout = next(
        param
        for param in config_management.config_apply.params
        if param.name == "timeout"
    )
    assert (
        timeout.default == config_management.MUTATION_TIMEOUT_SECONDS == WARMUP_SECONDS
    )


def test_save_pending_restart_writes_what_serve_applies(stack, tmp_path):
    active = stack.runtime_config()
    served = SOURCE.replace(f"port: {MOVED_PORT}", "port: 8899").encode()
    active.write_bytes(served)

    staged = save_pending_restart(stack, tmp_path / "config.yaml", "listeners changed")

    # #7222: the Router reloads its runtime config whenever it changes, so the
    # change waits beside it and the Router keeps serving what it served.
    assert staged == active.with_name("runtime-config.pending-restart.json")
    assert active.read_bytes() == served
    assert not active.with_suffix(".provenance.json").exists()
    config = json.loads(staged.read_text(encoding="utf-8"))["config"]
    document = yaml.safe_load(config)
    assert document["listeners"][0]["port"] == MOVED_PORT
    assert document["global"]["services"]["management_api"]["bind_address"] == "0.0.0.0"
    assert staged.stat().st_mode & 0o777 == 0o600
    pending = read_pending_activation(active)
    assert pending is not None and pending.reason == "restart"
    assert pending.origin == ORIGIN_CLI
    assert pending.saved_by() == "with `vllm-sr config apply`"
    assert pending.detail == "listeners changed"
    assert pending.config_sha256 == hashlib.sha256(config.encode()).hexdigest()


def test_a_recipe_package_stack_does_not_save_a_restart(monkeypatch, stack, tmp_path):
    monkeypatch.setattr(
        local_stack_config, "active_recipe_package_for_stack", lambda **_: True
    )

    # serve runs the package's config, so a saved change would never apply.
    with pytest.raises(ValueError, match="Recipe package"):
        save_pending_restart(stack, tmp_path / "config.yaml", "listeners changed")
    assert not pending_restart_path(stack.runtime_config()).exists()
    assert read_pending_activation(stack.runtime_config()) is None


@pytest.mark.parametrize(
    ("outcome", "start_calls", "readiness_calls"),
    [("host-port", 0, 0), ("start", 1, 0), ("readiness", 1, 1), ("success", 1, 1)],
)
def test_serve_clears_staged_restart_only_after_readiness(
    monkeypatch, stack, tmp_path, caplog, outcome, start_calls, readiness_calls
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "test-pending-restart")
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "500")
    stack = replace(stack, layout=resolve_runtime_stack())
    source = tmp_path / "config.yaml"
    document = yaml.safe_load(SOURCE)
    document["listeners"][0].update(
        name="public", port=65100 if outcome == "host-port" else MOVED_PORT
    )
    document["global"] = {
        "services": {"observability": {"tracing": {"enabled": False}}}
    }
    source.write_text(yaml.safe_dump(document), encoding="utf-8")

    reason = "listeners[public].port changes"
    runtime_config = stack.runtime_config()
    active_bytes = SOURCE.replace(f"port: {MOVED_PORT}", "port: 8899").encode()
    runtime_config.write_bytes(active_bytes)
    staged = save_pending_restart(stack, source, reason)
    pending_file = record_file(runtime_config)
    pending_bytes = pending_file.read_bytes()
    staged_bytes = json.loads(staged.read_text(encoding="utf-8"))["config"].encode()

    start = Mock(
        return_value=(
            (1, "", "port already allocated") if outcome == "start" else (0, "", "")
        )
    )
    ready = Mock(side_effect=SystemExit(1) if outcome == "readiness" else None)
    monkeypatch.setattr(container_backend, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_backend, "acquire_runtime_lifecycle_lock", lambda **_: nullcontext()
    )
    monkeypatch.setattr(container_backend, "attach_controller", lambda *_: None)
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _: None)
    monkeypatch.setattr(
        core, "_prepare_runtime_network", lambda *a, **k: ("net", str(tmp_path))
    )
    monkeypatch.setattr(core, "_start_support_services", lambda *a, **k: (set(), "net"))
    monkeypatch.setattr(core, "container_start_vllm_sr", start)
    monkeypatch.setattr(core, "connect_runtime_container", lambda *_: None)
    monkeypatch.setattr(runtime_lifecycle, "wait_for_router_health", ready)
    monkeypatch.setattr(
        runtime_lifecycle, "container_status_strict", lambda *a, **k: "running"
    )
    monkeypatch.setattr(core, "container_status", lambda _: "running")
    monkeypatch.setattr(
        core,
        "inspect_container_mounts",
        lambda _: [{"Source": str(stack.state_dir), "Destination": "/app/.vllm-sr"}],
    )
    monkeypatch.setattr(core, "report_service_status", lambda *_: None)
    stopped = []
    monkeypatch.setattr(
        runtime_serve_config, "stop_runtime_before_config_replacement", stopped.append
    )

    runner = CliRunner()
    status_args = ["status", "router", "--target", "docker"]
    before = runner.invoke(main, status_args)
    assert before.exit_code == 0, before.output
    assert "Restart required" in before.output and reason in before.output

    serve_args = [
        "serve",
        "--target",
        "docker",
        "--gateway",
        "standalone",
        "--config",
        str(source),
    ]
    result = runner.invoke(main, serve_args)
    succeeded = outcome == "success"
    assert result.exit_code == (0 if succeeded else 1), result.output
    assert start.call_count == start_calls
    assert ready.call_count == readiness_calls
    assert len(stopped) == start_calls
    if outcome == "host-port":
        assert "invalid listener public host port 65600" in caplog.text
        assert runtime_config.read_bytes() == active_bytes
        assert staged.is_file()
    if start_calls:
        assert Path(start.call_args.kwargs["runtime_config_file"]) == runtime_config
        assert runtime_config.read_bytes() == staged_bytes

    after = runner.invoke(main, status_args)
    assert after.exit_code == 0, after.output
    if succeeded:
        assert not pending_file.exists()
        assert "Restart required" not in after.output
    else:
        assert (
            pending_file.is_file()
        ), "failed serve discarded the pending restart record"
        assert pending_file.read_bytes() == pending_bytes
        assert "Restart required" in after.output and reason in after.output

    if outcome in {"start", "readiness"}:
        start.return_value, ready.side_effect = (0, "", ""), None
        retry = runner.invoke(main, serve_args)
        assert retry.exit_code == 0, retry.output
        assert len(stopped) == 1
        assert runtime_config.read_bytes() == staged_bytes
        assert not pending_file.exists()
        assert "Restart required" not in runner.invoke(main, status_args).output


@pytest.mark.parametrize("running_port", [8080, 9095])
def test_status_keeps_the_bound_port_after_a_restart_required_save(
    monkeypatch, stack, tmp_path, running_port
):
    source = tmp_path / "config.yaml"
    stack.runtime_config().write_text(SOURCE, encoding="utf-8")
    candidate = yaml.safe_load(source.read_text(encoding="utf-8"))
    candidate["global"] = {
        "services": {"management_api": {"bind_address": "0.0.0.0", "port": 9096}}
    }
    source.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    record = tmp_path / "management-port"
    record.write_text(f"{running_port}\n", encoding="utf-8")
    probed = []

    def fake_exec(_container_name, command):
        if command == ["cat", runtime_service_status.ROUTER_MANAGEMENT_PORT_PATH]:
            return (0, record.read_text(encoding="utf-8"), "")
        if command == ["sh", "-c", 'cat "$VLLM_SR_RUNTIME_CONFIG_PATH"']:
            return (0, stack.runtime_config().read_text(encoding="utf-8"), "")
        probed.append(command[-1])
        return (
            (0 if command[-1] == f"http://localhost:{running_port}/health" else 7),
            "",
            "",
        )

    monkeypatch.setattr(runtime_service_status, "container_exec", fake_exec)
    staged = save_pending_restart(stack, source, "listeners changed")
    saved = _staged_document(staged)
    assert saved["global"]["services"]["management_api"]["port"] == 9096
    pending = read_pending_activation(stack.runtime_config())
    assert pending is not None and pending.reason == "restart"

    assert runtime_service_status._check_router_status(
        stack.layout.router_container_name
    )
    assert probed == [f"http://localhost:{running_port}/health"]


def test_local_stack_reads_the_router_container(monkeypatch, tmp_path):
    state_dir = tmp_path / ".vllm-sr"
    monkeypatch.setattr(
        local_stack_config,
        "inspect_container_mounts",
        lambda _name: [{"Destination": "/app/.vllm-sr", "Source": str(state_dir)}],
    )
    monkeypatch.setattr(
        local_stack_config,
        "_container_env",
        lambda _name: {
            "VLLM_SR_PLATFORM": "rocm",
            "VLLM_SR_ALGORITHM_OVERRIDE": "random",
        },
    )

    found = local_stack_config.local_stack()

    assert found is not None
    assert found.state_dir == state_dir
    assert found.algorithm == "random"
    assert found.owns(tmp_path / "config.yaml")
    assert not found.owns(tmp_path / "elsewhere" / "config.yaml")


@pytest.mark.parametrize(
    ("host", "host_ip", "expected"),
    [
        ("localhost", "127.0.0.1", True),
        ("localhost", "0.0.0.0", True),
        ("localhost", "::1", True),
        ("localhost", "::", True),
        ("127.0.0.1", "127.0.0.1", True),
        ("127.0.0.1", "0.0.0.0", True),
        ("::1", "::1", True),
        ("::1", "::", True),
        ("127.0.0.1", "::1", False),
        ("::1", "127.0.0.1", False),
        ("localhost", "192.0.2.42", False),
        ("localhost", "", False),
        ("localhost", None, False),
    ],
)
def test_management_publication_matches_the_loopback_address_family(
    monkeypatch, stack, host, host_ip, expected
):
    _publish_management_endpoint(monkeypatch, stack, 8081, 9081, host_ip)

    assert stack.publishes_management_endpoint(host, 9081) is expected


@pytest.mark.parametrize(
    "bindings",
    [
        None,
        {},
        [None],
        ["9081"],
        [{"HostIp": "127.0.0.1", "HostPort": "not-a-port"}],
        [{"HostIp": "127.0.0.1", "HostPort": 9081}],
        [{"HostIp": "127.0.0.1", "HostPort": None}],
    ],
)
def test_unknown_management_binding_cannot_claim_an_endpoint(
    monkeypatch, stack, bindings
):
    _publish_management_endpoint(monkeypatch, stack, 8081, 9081)
    monkeypatch.setattr(
        local_stack_config, "_container_ports", lambda _name: {"8081/tcp": bindings}
    )

    assert not stack.publishes_management_endpoint("127.0.0.1", 9081)


def test_invalid_running_management_port_cannot_claim_an_endpoint(monkeypatch, stack):
    def invalid(_name):
        raise ValueError("invalid Router management port record")

    monkeypatch.setattr(local_stack_config, "_router_management_port", invalid)

    assert not stack.publishes_management_endpoint("127.0.0.1", 8080)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("http://localhost", ("localhost", 80)),
        ("http://127.0.0.1:8081/", ("127.0.0.1", 8081)),
        ("http://[::1]:8081", ("::1", 8081)),
        ("http://localhost:8080/api/v1", ("localhost", 8080)),
        ("http://localhost:8080/api/v1/", ("localhost", 8080)),
        ("http://localhost:8081/api/v1", ("localhost", 8081)),
        ("http://localhost:8081/api/v1/", ("localhost", 8081)),
        ("http://router.example:8081", None),
        ("https://localhost:8081", None),
        ("http://localhost:8081/proxy", None),
        ("http://localhost:8081/api/v1/proxy", None),
        ("http://localhost:8081?proxy=other", None),
        ("http://localhost:8081#proxy", None),
        ("http://user@localhost:8081", None),
    ],
)
def test_only_direct_loopback_http_origins_are_local_candidates(endpoint, expected):
    assert config_management._local_management_endpoint(endpoint) == expected


def test_container_port_inspection_returns_running_router_bindings(monkeypatch):
    bindings = {"8081/tcp": [{"HostIp": "127.0.0.1", "HostPort": "9081"}]}
    inspected = []

    def inspect(command, **kwargs):
        inspected.append(command)
        assert kwargs["timeout"] > 0
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"running": True, "ports": bindings}), ""
        )

    monkeypatch.setattr(local_stack_config, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(local_stack_config.subprocess, "run", inspect)

    assert local_stack_config._container_ports("selected-router") == bindings
    assert inspected[0][:3] == ["docker", "container", "inspect"]
    assert inspected[0][-1] == "selected-router"


@pytest.mark.parametrize(
    "output",
    [
        "",
        "not-json",
        "null",
        "[]",
        "{}",
        '{"running": false, "ports": {"8081/tcp": []}}',
        '{"running": "true", "ports": {"8081/tcp": []}}',
        '{"running": true, "ports": null}',
        '{"running": true, "ports": []}',
    ],
)
def test_missing_or_stopped_container_publication_is_unknown(monkeypatch, output):
    monkeypatch.setattr(local_stack_config, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        local_stack_config.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, output, ""),
    )

    assert local_stack_config._container_ports("selected-router") == {}


@pytest.mark.parametrize(
    "error",
    [
        OSError("runtime unavailable"),
        subprocess.CalledProcessError(1, ["docker", "container", "inspect"]),
        subprocess.TimeoutExpired(["docker", "container", "inspect"], 15),
        SystemExit("no supported runtime"),
    ],
)
def test_container_port_inspection_failure_is_unknown(monkeypatch, error):
    def unavailable(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(local_stack_config, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(local_stack_config.subprocess, "run", unavailable)

    assert local_stack_config._container_ports("selected-router") == {}
