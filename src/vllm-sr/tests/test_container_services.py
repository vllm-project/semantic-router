import subprocess

from cli import container_services


def test_container_status_uses_exact_inspect_name(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="running\n", stderr="")

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services.subprocess, "run", fake_run)

    assert container_services.container_status("vllm-sr-router-container") == "running"
    assert calls == [
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}}",
            "vllm-sr-router-container",
        ]
    ]


def test_container_status_reports_missing_exact_name(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "podman")
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="no such container"
        ),
    )

    assert container_services.container_status("missing") == "not found"


def test_container_create_network_does_not_match_namespaced_suffix(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1:3] == ["network", "ls"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="mom-amd-v1-vllm-sr-network\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="network-id\n", stderr="")

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services.subprocess, "run", fake_run)

    assert container_services.container_create_network("vllm-sr-network")[0] == 0
    assert calls[-1] == ["docker", "network", "create", "vllm-sr-network"]


def test_container_create_network_accepts_exact_existing_name(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="vllm-sr-network\n",
            stderr="",
        )

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services.subprocess, "run", fake_run)

    assert container_services.container_create_network("vllm-sr-network")[0] == 0
    assert len(calls) == 1
