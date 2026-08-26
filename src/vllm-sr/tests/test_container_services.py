import json
import subprocess

from cli import container_mounts, container_services
from cli.runtime_stack import resolve_runtime_stack


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
                stdout="isolated-v1-vllm-sr-network\n",
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


def test_container_network_subnets_supports_docker_and_podman_shapes(monkeypatch):
    payload = [
        {
            "IPAM": {"Config": [{"Subnet": "172.24.0.0/16"}]},
            "subnets": [
                {"subnet": "fd42:24::/64"},
                {"subnet": "172.24.0.0/16"},
            ],
        }
    ]
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    assert container_services.container_network_subnets(
        "vllm-sr-network", runtime="podman"
    ) == (0, ["172.24.0.0/16", "fd42:24::/64"], "")


def test_container_network_subnets_fails_closed_on_invalid_inspection(monkeypatch):
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="{}", stderr=""
        ),
    )

    return_code, subnets, error = container_services.container_network_subnets(
        "vllm-sr-network", runtime="docker"
    )
    assert return_code != 0
    assert subnets == []
    assert "one document" in error


def _storage_start_environment(monkeypatch, commands, *, status="not found"):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: status)
    monkeypatch.setattr(container_services, "_is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        container_services,
        "_run_service_start",
        lambda cmd, _label: commands.append(cmd) or (0, "", ""),
    )


def test_redis_reads_its_password_from_a_mounted_config_not_the_argv(
    monkeypatch, tmp_path
):
    commands = []
    _storage_start_environment(monkeypatch, commands)
    conf_file = tmp_path / "redis.conf"
    conf_file.write_text("requirepass never-in-argv\n")
    layout = resolve_runtime_stack()

    container_services.container_start_redis(
        "test-network",
        layout,
        redis_conf_file=str(conf_file),
        data_volume="vllm-sr-redis-data",
    )

    command = commands[0]
    conf_mount = f"{conf_file}:{container_services.CONTAINER_REDIS_CONF_PATH}:ro,z"
    assert conf_mount in command
    assert f"vllm-sr-redis-data:{container_services.REDIS_DATA_MOUNT_PATH}" in command
    assert command[-2:] == [
        "redis-server",
        container_services.CONTAINER_REDIS_CONF_PATH,
    ]
    assert command[command.index("--restart") + 1] == "unless-stopped"
    redis_healthcheck = command[command.index("--health-cmd") + 1]
    assert container_services.CONTAINER_REDIS_CONF_PATH in redis_healthcheck
    assert 'REDISCLI_AUTH="$password"' in redis_healthcheck
    # The container process is a host process, so argv is world readable.
    assert not any(argument.startswith("--requirepass") for argument in command)
    assert not any("never-in-argv" in argument for argument in command)


def test_postgres_reads_its_password_from_a_file_never_from_argv(monkeypatch, tmp_path):
    commands = []
    _storage_start_environment(monkeypatch, commands)
    password_file = tmp_path / "postgres-password"
    password_file.write_text("never-in-argv")
    layout = resolve_runtime_stack()

    container_services.container_start_postgres(
        "test-network",
        layout,
        postgres_password_file=str(password_file),
        data_volume="vllm-sr-postgres-data",
    )

    command = commands[0]
    assert (
        f"POSTGRES_PASSWORD_FILE={container_services.CONTAINER_POSTGRES_PASSWORD_PATH}"
        in command
    )
    assert (
        f"{password_file}:{container_services.CONTAINER_POSTGRES_PASSWORD_PATH}:ro,z"
        in command
    )
    assert (
        f"vllm-sr-postgres-data:{container_services.POSTGRES_DATA_MOUNT_PATH}"
        in command
    )
    assert not any(argument.startswith("POSTGRES_PASSWORD=") for argument in command)
    assert not any("never-in-argv" in argument for argument in command)
    assert not any("router-secret" in argument for argument in command)
    assert command[command.index("--restart") + 1] == "unless-stopped"
    assert command[command.index("--health-cmd") + 1] == (
        "pg_isready -q -U router -d vsr"
    )


def test_replacing_a_storage_container_reads_its_volume_before_removing_it(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "exited")
    monkeypatch.setattr(
        container_services,
        "adopted_volume_name",
        lambda name, destination: events.append(("inspect", destination)) or "old-vol",
    )
    monkeypatch.setattr(
        container_services,
        "container_stop_container",
        lambda name: events.append(("stop", name)) or True,
    )
    monkeypatch.setattr(
        container_services,
        "container_remove_container",
        lambda name: events.append(("remove", name)) or True,
    )

    adopted = container_services._replace_existing_container(
        "vllm-sr-postgres",
        adopt_volume_destination=container_services.POSTGRES_DATA_MOUNT_PATH,
    )

    assert adopted == "old-vol"
    # Removal keeps the volume but destroys the record of which volume it was.
    assert events == [
        ("inspect", container_services.POSTGRES_DATA_MOUNT_PATH),
        ("stop", "vllm-sr-postgres"),
        ("remove", "vllm-sr-postgres"),
    ]


def test_a_replaced_container_with_no_volume_mount_is_not_an_error(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "exited")
    monkeypatch.setattr(
        container_services, "adopted_volume_name", lambda _name, _destination: None
    )
    monkeypatch.setattr(container_services, "container_stop_container", lambda _n: True)
    monkeypatch.setattr(
        container_services, "container_remove_container", lambda _n: True
    )

    assert (
        container_services._replace_existing_container(
            "vllm-sr-redis",
            adopt_volume_destination=container_services.REDIS_DATA_MOUNT_PATH,
        )
        is None
    )


def test_an_adopted_volume_is_mounted_when_the_caller_records_no_name(
    monkeypatch, tmp_path
):
    commands = []
    _storage_start_environment(monkeypatch, commands, status="exited")
    monkeypatch.setattr(
        container_services, "adopted_volume_name", lambda _name, _dest: "anon-hex-id"
    )
    monkeypatch.setattr(container_services, "container_stop_container", lambda _n: True)
    monkeypatch.setattr(
        container_services, "container_remove_container", lambda _n: True
    )

    # No `data_volume`: the volume read off the replaced container is the only
    # thing that can carry the data forward here.
    container_services.container_start_redis(
        "test-network",
        resolve_runtime_stack(),
        redis_conf_file=str(tmp_path / "redis.conf"),
    )

    assert f"anon-hex-id:{container_services.REDIS_DATA_MOUNT_PATH}" in commands[0]


def test_mount_destinations_are_unknown_when_the_runtime_cannot_answer(monkeypatch):
    monkeypatch.setattr(container_mounts, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_mounts.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("daemon unreachable")),
    )

    assert container_services.container_mount_destinations("vllm-sr-redis") is None
