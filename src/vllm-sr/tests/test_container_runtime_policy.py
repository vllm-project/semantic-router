import pytest
from cli.container_runtime_policy import (
    DOCKER_RESTART_POLICY,
    DockerHealthcheck,
    append_docker_runtime_policy,
    dashboard_healthcheck,
    envoy_healthcheck,
    postgres_healthcheck,
    redis_healthcheck,
    router_healthcheck,
)


def _option_value(command, option):
    return command[command.index(option) + 1]


def test_docker_runtime_policy_adds_restart_and_healthcheck_options():
    command = ["docker", "run", "-d"]

    append_docker_runtime_policy(
        command,
        "docker",
        DockerHealthcheck(command="probe", retries=4, start_period="1m"),
    )

    assert _option_value(command, "--restart") == DOCKER_RESTART_POLICY
    assert _option_value(command, "--health-cmd") == "probe"
    assert _option_value(command, "--health-interval") == "10s"
    assert _option_value(command, "--health-timeout") == "5s"
    assert _option_value(command, "--health-retries") == "4"
    assert _option_value(command, "--health-start-period") == "1m"


def test_runtime_policy_does_not_change_podman_commands():
    command = ["podman", "run", "-d"]

    append_docker_runtime_policy(
        command,
        "podman",
        DockerHealthcheck(command="probe"),
    )

    assert command == ["podman", "run", "-d"]


@pytest.mark.parametrize("command", ["", "   "])
def test_healthcheck_command_must_not_be_empty(command):
    with pytest.raises(ValueError, match="must not be empty"):
        DockerHealthcheck(command=command)


def test_router_healthcheck_uses_configured_tls_listener():
    healthcheck = router_healthcheck(
        9443,
        tls_enabled=True,
    )

    assert " -k " in f" {healthcheck.command} "
    assert "https://127.0.0.1:9443/ready" in healthcheck.command
    assert healthcheck.start_period == "30m"


def test_component_healthchecks_target_owned_readiness_interfaces():
    assert "/ready" in envoy_healthcheck().command
    assert "127.0.0.1/9901" in envoy_healthcheck().command
    assert dashboard_healthcheck().command.endswith("http://127.0.0.1:8700/healthz")
    assert postgres_healthcheck("router", "vsr").command == (
        "pg_isready -q -U router -d vsr"
    )


def test_redis_healthcheck_reads_secret_at_probe_time():
    command = redis_healthcheck("/run/redis.conf").command

    assert "/run/redis.conf" in command
    assert 'REDISCLI_AUTH="$password"' in command
    assert "--requirepass" not in command
