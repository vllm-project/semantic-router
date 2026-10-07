"""The docker target in standalone mode: the Router serves the listeners."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from cli import container_cli, container_start, core
from cli.gateway_mode import GATEWAY_ENV, GATEWAY_EXTPROC, stack_gateway

REPO_ROOT = Path(__file__).resolve().parents[3]
LISTENERS = [{"name": "http-8899", "address": "127.0.0.1", "port": 8899}]


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")


def _start(tmp_path, monkeypatch, gateway):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\nlisteners:\n  - name: http-8899\n    address: 127.0.0.1\n    port: 8899\n"
    )
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    resolved_images = []

    def images(**kwargs):
        resolved_images.append(kwargs)
        return {"router": "img", "envoy": "img", "dashboard": "img"}

    monkeypatch.setattr(container_start, "get_runtime_images", images)
    rendered = []
    monkeypatch.setattr(
        container_start,
        "_render_split_envoy_config",
        lambda *a, **k: rendered.append(a),
    )
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    commands = []

    def fake_run(cmd, capture_output, text, check, env=None):
        commands.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        LISTENERS,
        network_name="vllm-sr-network",
        minimal=False,
        gateway=gateway,
    )
    assert rc == 0
    by_name = {cmd[cmd.index("--name") + 1]: cmd for cmd in commands if "--name" in cmd}
    return by_name, rendered, resolved_images


def test_standalone_runs_no_envoy_and_the_router_publishes_the_listeners(
    tmp_path, monkeypatch
):
    containers, rendered, images = _start(tmp_path, monkeypatch, "standalone")

    assert "vllm-sr-envoy-container" not in containers
    assert rendered == [], "no Envoy config is rendered for a stack without Envoy"
    assert images[0]["include_envoy"] is False, "no Envoy image is pulled either"
    router = containers["vllm-sr-router-container"]
    assert (
        "127.0.0.1:8899:8899" in router
    ), "the configured address governs host publication"
    assert f"{GATEWAY_ENV}=standalone" in router
    dashboard = containers["vllm-sr-dashboard-container"]
    assert "TARGET_ENVOY_URL=http://vllm-sr-router-container:8899" in dashboard
    assert not any(value.startswith("TARGET_ENVOY_ADMIN_URL=") for value in dashboard)
    assert f"{GATEWAY_ENV}=standalone" in dashboard


def test_extproc_keeps_the_split_envoy_stack(tmp_path, monkeypatch):
    containers, rendered, images = _start(tmp_path, monkeypatch, "extproc")

    assert rendered, "extproc renders the Envoy config"
    assert images[0]["include_envoy"] is True
    assert "127.0.0.1:8899:8899" in containers["vllm-sr-envoy-container"]
    assert "127.0.0.1:8899:8899" not in containers["vllm-sr-router-container"]
    assert f"{GATEWAY_ENV}=extproc" in containers["vllm-sr-router-container"]
    assert "TARGET_ENVOY_URL=http://vllm-sr-envoy-container:8899" in (
        containers["vllm-sr-dashboard-container"]
    )


def test_router_entrypoint_serves_the_mode_it_is_given():
    script = (REPO_ROOT / "src/vllm-sr/start-router.sh").read_text()
    assert 'if [ "${VLLM_SR_GATEWAY:-}" = "standalone" ]; then' in script
    assert "GATEWAY_ARGS=(-gateway=standalone -listener-address=0.0.0.0)" in script
    assert '"${GATEWAY_ARGS[@]}"' in script


def test_a_stack_without_a_recorded_mode_runs_envoy(monkeypatch):
    monkeypatch.delenv(GATEWAY_ENV, raising=False)
    assert stack_gateway() == GATEWAY_EXTPROC


def test_standalone_mounts_each_tls_listeners_certificate(tmp_path):
    (tmp_path / "certs").mkdir()
    (tmp_path / "certs" / "tls.crt").write_text("cert")
    absolute_key = tmp_path / "key.pem"
    absolute_key.write_text("key")
    listeners = [
        {
            "name": "https",
            "port": 8443,
            "tls": {"cert_file": "certs/tls.crt", "key_file": str(absolute_key)},
        },
        {"name": "http", "port": 8899},
    ]
    assert container_start._listener_tls_mounts(listeners, str(tmp_path)) == [
        f"{tmp_path}/certs/tls.crt:/app/certs/tls.crt:ro,z",
        f"{absolute_key}:{absolute_key}:ro,z",
    ]
    listeners[0]["tls"]["cert_file"] = "certs/missing.crt"
    with pytest.raises(ValueError, match=r"tls\.cert_file"):
        container_start._listener_tls_mounts(listeners, str(tmp_path))


def test_status_and_logs_explain_a_stack_without_envoy(monkeypatch, capsys, caplog):
    states = {
        "vllm-sr-router-container": "running",
        "vllm-sr-dashboard-container": "running",
    }
    monkeypatch.setattr(
        core, "container_status", lambda name: states.get(name, "not found")
    )
    reported = []
    monkeypatch.setattr(
        core, "report_service_status", lambda service, layout: reported.append(service)
    )
    monkeypatch.setattr(core, "inspect_container_mounts", lambda _name: [])

    core.show_status("all")
    assert reported == ["router", "dashboard"]
    assert "vllm-sr logs <router|dashboard>" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        core.show_logs("envoy")
    assert "runs in standalone mode, with no Envoy container" in caplog.text
