"""The docker target in standalone mode: the Router serves the listeners."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from cli import container_cli, container_start, core
from cli.container_listener_tls import _listener_tls_mounts
from cli.gateway_mode import GATEWAY_ENV, GATEWAY_EXTPROC, stack_gateway

REPO_ROOT = Path(__file__).resolve().parents[3]
LISTENERS = [{"name": "http-8899", "address": "127.0.0.1", "port": 8899}]


@pytest.fixture(autouse=True)
def _split_runtime_topology(monkeypatch):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")
    monkeypatch.delenv("VLLM_SR_LISTENER_TLS_DIRS", raising=False)


def _start(tmp_path, monkeypatch, gateway, *, listeners=LISTENERS, tls_image_id=""):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"version": "v0.5", "listeners": listeners}), encoding="utf-8"
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

    def fake_run(cmd, capture_output, text, check, env=None, timeout=None):
        commands.append(cmd)
        if cmd[1:3] == ["image", "inspect"]:
            assert "io.vllm-sr.listener-tls-directory-mounts" in cmd[4]
            return SimpleNamespace(stdout=tls_image_id + "\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="container-id\n", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {},
        listeners,
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
    assert _listener_tls_mounts(listeners, str(tmp_path)) == [
        f"{tmp_path}/certs/tls.crt:/app/certs/tls.crt:ro,z",
        f"{absolute_key}:{absolute_key}:ro,z",
    ]
    listeners[0]["tls"]["cert_file"] = "certs/missing.crt"
    with pytest.raises(ValueError, match=r"tls\.cert_file"):
        _listener_tls_mounts(listeners, str(tmp_path))


@pytest.mark.parametrize("separate", [False, True])
def test_standalone_mounts_tls_parent_directories_for_atomic_renewal(
    tmp_path, monkeypatch, separate
):
    cert_dir = tmp_path / "certs"
    key_dir = tmp_path / "keys" if separate else cert_dir
    cert_dir.mkdir()
    key_dir.mkdir(exist_ok=True)
    cert = cert_dir / "tls.crt"
    key = key_dir / "tls.key"
    cert.write_text("old certificate")
    key.write_text("old key")
    listeners = [
        {**LISTENERS[0], "tls": {"cert_file": "certs/tls.crt", "key_file": str(key)}}
    ]
    allowed_dirs = ["certs/."]
    if separate:
        allowed_dirs.append(str(key_dir))
    monkeypatch.setenv("VLLM_SR_LISTENER_TLS_DIRS", os.pathsep.join(allowed_dirs))
    image_id = "sha256:" + "a" * 64
    containers, _, _ = _start(
        tmp_path, monkeypatch, "standalone", listeners=listeners, tls_image_id=image_id
    )
    command = containers["vllm-sr-router-container"]
    first = "/run/vllm-sr/listener-tls/0"
    second = "/run/vllm-sr/listener-tls/1" if separate else first
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "-v"]
    tls_mounts = [mount for mount in mounts if ":/run/vllm-sr/listener-tls/" in mount]
    expected = [f"{cert_dir}:{first}:ro,z"]
    if separate:
        expected.append(f"{key_dir}:{second}:ro,z")
    assert tls_mounts == expected
    assert not any(mount.startswith((f"{cert}:", f"{key}:")) for mount in mounts)
    assert image_id in command

    _, mapping = next(
        arg for arg in command if arg.startswith("VLLM_SR_LISTENER_TLS_PATHS=")
    ).split("=", 1)
    assert json.loads(mapping) == {
        "certs/tls.crt": f"{first}/tls.crt",
        str(key): f"{second}/tls.key",
    }
    dashboard = containers["vllm-sr-dashboard-container"]
    assert not any("VLLM_SR_LISTENER_TLS_PATHS" in arg for arg in dashboard)
    assert not any(mount in dashboard for mount in tls_mounts)


@pytest.mark.parametrize("opt_in", [None, "..", "."])
def test_standalone_keeps_tls_file_mounts_without_a_matching_capable_opt_in(
    tmp_path, monkeypatch, opt_in
):
    cert = tmp_path / "tls.crt"
    key = tmp_path / "tls.key"
    cert.write_text("cert")
    key.write_text("key")
    listeners = [
        {**LISTENERS[0], "tls": {"cert_file": cert.name, "key_file": key.name}}
    ]
    if opt_in is not None:
        monkeypatch.setenv("VLLM_SR_LISTENER_TLS_DIRS", opt_in)
    # A recent image needs an exact opt-in; a legacy image uses file mounts
    # even when the source config directory is allowlisted.
    image_id = "" if opt_in == "." else "sha256:" + "a" * 64
    containers, _, _ = _start(
        tmp_path, monkeypatch, "standalone", listeners=listeners, tls_image_id=image_id
    )
    command = containers["vllm-sr-router-container"]
    assert f"{cert}:/app/tls.crt:ro,z" in command
    assert f"{key}:/app/tls.key:ro,z" in command
    assert not any(":/run/vllm-sr/listener-tls/" in arg for arg in command)
    assert not any(arg.startswith("VLLM_SR_LISTENER_TLS_PATHS=") for arg in command)


def test_standalone_keeps_undeclared_tls_parent_on_a_file_mount(tmp_path, monkeypatch):
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    cert = cert_dir / "tls.crt"
    cert.write_text("cert")
    key = tmp_path / "tls.key"
    key.write_text("key")
    listeners = [
        {**LISTENERS[0], "tls": {"cert_file": "certs/tls.crt", "key_file": key.name}}
    ]
    monkeypatch.setenv("VLLM_SR_LISTENER_TLS_DIRS", "certs")
    containers, _, _ = _start(
        tmp_path,
        monkeypatch,
        "standalone",
        listeners=listeners,
        tls_image_id="sha256:" + "a" * 64,
    )
    command = containers["vllm-sr-router-container"]
    assert f"{cert_dir}:/run/vllm-sr/listener-tls/0:ro,z" in command
    assert f"{key}:/app/tls.key:ro,z" in command
    assert not any(
        arg.startswith(f"{tmp_path}:/run/vllm-sr/listener-tls/") for arg in command
    )
    mapping = next(
        arg.split("=", 1)[1]
        for arg in command
        if arg.startswith("VLLM_SR_LISTENER_TLS_PATHS=")
    )
    assert json.loads(mapping) == {
        "certs/tls.crt": "/run/vllm-sr/listener-tls/0/tls.crt"
    }


def test_standalone_keeps_legacy_mount_for_a_listener_tls_symlink(
    tmp_path, monkeypatch
):
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    actual_cert = tmp_path / "issued.crt"
    actual_cert.write_text("cert")
    cert = cert_dir / "tls.crt"
    cert.symlink_to(actual_cert)
    key = cert_dir / "tls.key"
    key.write_text("key")
    listeners = [
        {
            **LISTENERS[0],
            "tls": {"cert_file": "certs/tls.crt", "key_file": "certs/tls.key"},
        }
    ]
    monkeypatch.setenv("VLLM_SR_LISTENER_TLS_DIRS", "certs")
    containers, _, _ = _start(
        tmp_path,
        monkeypatch,
        "standalone",
        listeners=listeners,
        tls_image_id="sha256:" + "a" * 64,
    )
    command = containers["vllm-sr-router-container"]
    assert f"{cert}:/app/certs/tls.crt:ro,z" in command
    assert f"{cert_dir}:/run/vllm-sr/listener-tls/0:ro,z" in command
    assert not any(
        arg.startswith(f"{tmp_path}:/run/vllm-sr/listener-tls/") for arg in command
    )
    mapping = next(
        arg.split("=", 1)[1]
        for arg in command
        if arg.startswith("VLLM_SR_LISTENER_TLS_PATHS=")
    )
    assert json.loads(mapping) == {
        "certs/tls.key": "/run/vllm-sr/listener-tls/0/tls.key"
    }


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
