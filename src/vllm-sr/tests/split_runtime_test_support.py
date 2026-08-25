"""Command-capture helpers shared by split runtime tests."""

from types import SimpleNamespace

from cli import container_start

CONFIG_BODY = (
    "version: v0.3\n"
    "listeners:\n"
    "  - name: http-8899\n"
    "    address: 0.0.0.0\n"
    "    port: 8899\n"
    "global:\n"
    "  services:\n"
    "    backend_egress:\n"
    "      policy_file: /app/config/backend-egress-policy.yaml\n"
)


def capture_run_commands(monkeypatch):
    captured = []

    def fake_run(cmd, capture_output, text, check, env=None):
        captured.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(container_start.subprocess, "run", fake_run)
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *args, **kwargs: None
    )
    return captured


def find_container_run_cmd(commands, container_name):
    for command in commands:
        if (
            "--name" in command
            and command[command.index("--name") + 1] == container_name
        ):
            return command
    raise AssertionError(
        f"container command for {container_name} not found: {commands!r}"
    )


def option_values(command, option):
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


def stub_valid_container_cli(monkeypatch, tmp_path):
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )
    return docker_bin
