from types import SimpleNamespace

from cli import core, docker_cli, docker_start


def test_docker_start_vllm_sr_sets_openclaw_shared_network_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    socket_path = tmp_path / "docker.sock"
    socket_path.write_text("")
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(docker_start, "get_docker_image", lambda **kwargs: "test-image")
    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(
        docker_start.shutil,
        "which",
        lambda name: str(docker_bin) if name == "docker" else None,
    )
    monkeypatch.setenv("VLLM_SR_DOCKER_SOCKET", str(socket_path))

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name="vllm-sr-network",
        minimal=True,
    )

    assert rc == 0
    assert "cmd" in captured
    assert "-e" in captured["cmd"]
    assert "OPENCLAW_DEFAULT_NETWORK_MODE=vllm-sr-network" in captured["cmd"]


def test_docker_start_vllm_sr_mounts_dashboard_data_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(docker_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(docker_start, "get_docker_image", lambda **kwargs: "test-image")
    monkeypatch.setattr(docker_start.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_start.shutil, "which", lambda name: None)

    rc, _, _ = docker_cli.docker_start_vllm_sr(
        str(config_path),
        {},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        network_name=None,
        openclaw_network_name="vllm-sr-network",
        minimal=True,
    )

    dashboard_data_dir = tmp_path / ".vllm-sr" / "dashboard-data"

    assert rc == 0
    assert dashboard_data_dir.is_dir()
    assert f"{dashboard_data_dir}:/app/data:z" in captured["cmd"]


def test_start_vllm_sr_creates_and_connects_shared_network_without_observability(
    monkeypatch,
):
    calls = []

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return _fn

    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(
        core,
        "load_config",
        lambda path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(
        core,
        "docker_container_status",
        lambda name: "not found" if name == "vllm-sr-container" else "running",
    )
    monkeypatch.setattr(core, "docker_create_network", record("docker_create_network"))
    monkeypatch.setattr(core, "docker_start_vllm_sr", record("docker_start_vllm_sr"))
    monkeypatch.setattr(
        core, "docker_network_connect", record("docker_network_connect")
    )
    monkeypatch.setattr(core, "docker_logs_since", lambda *args, **kwargs: (0, "", ""))
    monkeypatch.setattr(core, "docker_exec", lambda *args, **kwargs: (0, "ok", ""))
    monkeypatch.setattr(core, "load_openclaw_registry", lambda *args, **kwargs: [])
    monkeypatch.setattr(core, "docker_logs", lambda *args, **kwargs: None)

    core.start_vllm_sr("/tmp/config.yaml", env_vars={}, enable_observability=False)

    create_calls = [c for c in calls if c[0] == "docker_create_network"]
    start_calls = [c for c in calls if c[0] == "docker_start_vllm_sr"]
    connect_calls = [c for c in calls if c[0] == "docker_network_connect"]

    assert create_calls[0][1] == ("vllm-sr-network",)
    assert start_calls[0][2]["network_name"] is None
    assert start_calls[0][2]["openclaw_network_name"] == "vllm-sr-network"
    assert connect_calls[0][1] == ("vllm-sr-network", "vllm-sr-container")
