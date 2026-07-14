"""Credential propagation tests for the remote embedding provider."""

import base64

from cli import container_start
from cli.commands.runtime_support import (
    EMBEDDING_API_KEY_ENV,
    append_passthrough_env_vars,
)
from cli.dashboard_auth_runtime import DISABLED_DASHBOARD_AUTH_PLAN
from cli.k8s_secret_plan import desired_encoded_secret_data


def test_embedding_api_key_passthrough_is_masked(monkeypatch, caplog) -> None:
    secret = "embedding-secret-value"
    monkeypatch.setenv(EMBEDDING_API_KEY_ENV, secret)

    env_vars: dict[str, str] = {}
    with caplog.at_level("INFO", logger="cli.commands.runtime_support"):
        append_passthrough_env_vars(env_vars)

    assert env_vars[EMBEDDING_API_KEY_ENV] == secret
    assert secret not in caplog.text
    assert f"{EMBEDDING_API_KEY_ENV}=***" in caplog.text


def test_embedding_api_key_is_exposed_only_to_local_router(monkeypatch) -> None:
    secret = "embedding-secret-value"
    monkeypatch.setenv(EMBEDDING_API_KEY_ENV, "stale-host-value")
    common_env = {EMBEDDING_API_KEY_ENV: secret}

    router_env = container_start._service_process_env(
        "router", common_env, DISABLED_DASHBOARD_AUTH_PLAN
    )
    dashboard_env = container_start._service_process_env(
        "dashboard", common_env, DISABLED_DASHBOARD_AUTH_PLAN
    )
    envoy_env = container_start._service_process_env(
        "envoy", common_env, DISABLED_DASHBOARD_AUTH_PLAN
    )

    assert router_env[EMBEDDING_API_KEY_ENV] == secret
    assert EMBEDDING_API_KEY_ENV not in dashboard_env
    assert EMBEDDING_API_KEY_ENV not in envoy_env


def test_embedding_api_key_is_redacted_from_container_command() -> None:
    secret = "embedding-secret-value"
    command = ["docker", "run", "-e", f"{EMBEDDING_API_KEY_ENV}={secret}"]

    redacted = container_start._redact_container_command(command)

    assert secret not in " ".join(redacted)
    assert redacted[-1] == f"{EMBEDDING_API_KEY_ENV}=***"


def test_embedding_api_key_is_stored_in_k8s_secret_plan() -> None:
    secret = "embedding-secret-value"

    encoded = desired_encoded_secret_data({EMBEDDING_API_KEY_ENV: secret}, None)

    assert base64.b64decode(encoded[EMBEDDING_API_KEY_ENV]).decode() == secret
