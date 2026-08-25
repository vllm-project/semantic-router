from __future__ import annotations

from types import SimpleNamespace

import pytest
from cli.bootstrap import LOCAL_REPLICA_ID_ENV
from cli.control_plane_deployment import (
    control_plane_store_references,
    local_control_plane_secret_mounts,
    plan_local_control_plane,
    resolve_local_router_bindings,
    runtime_capabilities,
)
from cli.management_migration import (
    MIGRATION_BINARY,
    build_management_migration_command,
    run_management_migration,
)
from cli.runtime_stack import resolve_runtime_stack
from cli.storage_secrets import (
    POSTGRES_PASSWORD_ENV,
    POSTGRES_PASSWORD_PLACEHOLDER,
    REDIS_PASSWORD_ENV,
    REDIS_PASSWORD_PLACEHOLDER,
    PostgresSecret,
    RedisSecret,
    StorageSecrets,
)


def _durable_config(*, postgres=None, redis=None):
    return {
        "version": "v0.3",
        "global": {
            "stores": {
                "management": {
                    "postgres": (
                        {"dsn_env": "ACCESS_DATABASE_URL"}
                        if postgres is None
                        else postgres
                    )
                },
                "runtime": {
                    "redis": (
                        {"url_env": "ACCESS_RUNTIME_URL"} if redis is None else redis
                    )
                },
            }
        },
    }


def _storage_secrets(stack_name="vllm-sr"):
    return StorageSecrets(
        stack=stack_name,
        postgres=PostgresSecret(
            user="router",
            database="vsr",
            password="postgres-secret",
            volume="postgres-data",
        ),
        redis=RedisSecret(password="redis-secret", volume="redis-data"),
    )


def test_file_config_is_dependency_free_and_does_not_mutate_environment():
    runtime_env = {"PUBLIC_SETTING": "on"}

    plan = plan_local_control_plane(
        {"version": "v0.3"}, runtime_env, resolve_runtime_stack()
    )

    assert plan.required_backends == frozenset()
    assert (
        resolve_local_router_bindings(plan, set(), None, resolve_runtime_stack()) == {}
    )
    assert runtime_env == {"PUBLIC_SETTING": "on"}
    assert runtime_capabilities({"version": "v0.3"}).file_routing is True


def test_local_service_credentials_are_resolved_from_committed_state():
    config = {
        "version": "v0.3",
        "global": {
            "services": {
                "response_api": {
                    "enabled": True,
                    "store_backend": "redis",
                    "redis": {"password": REDIS_PASSWORD_PLACEHOLDER},
                },
                "router_replay": {
                    "enabled": True,
                    "store_backend": "postgres",
                    "postgres": {"password": POSTGRES_PASSWORD_PLACEHOLDER},
                },
            }
        },
    }
    stack = resolve_runtime_stack(stack_name="service-secrets")
    plan = plan_local_control_plane(config, {}, stack)
    bindings = resolve_local_router_bindings(
        plan,
        {"postgres", "redis"},
        _storage_secrets(stack.stack_name),
        stack,
    )

    assert plan.required_backends == frozenset()
    assert bindings[POSTGRES_PASSWORD_ENV] == "postgres-secret"
    assert bindings[REDIS_PASSWORD_ENV] == "redis-secret"


def test_durable_docker_plans_missing_store_refs_then_binds_committed_state():
    runtime_env = {}
    stack = resolve_runtime_stack(stack_name="durable-test")

    plan = plan_local_control_plane(_durable_config(), runtime_env, stack)
    bindings = resolve_local_router_bindings(
        plan,
        {"postgres", "redis"},
        _storage_secrets(stack.stack_name),
        stack,
    )

    assert plan.required_backends == frozenset({"postgres", "redis"})
    assert bindings["ACCESS_DATABASE_URL"].startswith("postgresql://router:")
    assert f"@{stack.postgres_container_name}:" in bindings["ACCESS_DATABASE_URL"]
    assert bindings["ACCESS_RUNTIME_URL"].startswith("redis://:")
    assert f"@{stack.redis_container_name}:6379/0" in bindings["ACCESS_RUNTIME_URL"]
    assert bindings[LOCAL_REPLICA_ID_ENV] == "local-durable-test"
    assert runtime_env == {}


def test_external_store_env_refs_bypass_local_store_provisioning():
    runtime_env = {
        "ACCESS_DATABASE_URL": "postgresql://external/db",
        "ACCESS_RUNTIME_URL": "redis://external:6379/0",
    }

    plan = plan_local_control_plane(
        _durable_config(), runtime_env, resolve_runtime_stack()
    )

    assert plan.required_backends == frozenset()
    assert resolve_local_router_bindings(
        plan, set(), None, resolve_runtime_stack()
    ) == {LOCAL_REPLICA_ID_ENV: "local-vllm-sr"}


def test_local_binding_fails_closed_without_committed_storage_state():
    stack = resolve_runtime_stack(stack_name="missing-state")
    plan = plan_local_control_plane(_durable_config(), {}, stack)

    with pytest.raises(ValueError, match="committed credential state"):
        resolve_local_router_bindings(plan, {"postgres", "redis"}, None, stack)


def test_store_file_refs_are_exact_read_only_mounts(tmp_path):
    postgres = tmp_path / "postgres-dsn"
    redis = tmp_path / "redis-url"
    postgres.write_text("postgresql://external/db", encoding="utf-8")
    redis.write_text("redis://external:6379/0", encoding="utf-8")
    config = _durable_config(
        postgres={"dsn_file": str(postgres)},
        redis={"url_file": str(redis)},
    )

    assert local_control_plane_secret_mounts(config) == tuple(
        f"{item}:{item}:ro" for item in sorted((str(postgres), str(redis)))
    )


@pytest.mark.parametrize(
    "postgres",
    [
        {},
        {"dsn_env": "ACCESS_DATABASE_URL", "dsn_file": "/run/secrets/db"},
        {"dsn_env": "lowercase_name"},
        {"dsn_file": "relative/path"},
    ],
)
def test_store_references_reject_ambiguous_or_non_secret_inputs(postgres):
    with pytest.raises(ValueError):
        control_plane_store_references(_durable_config(postgres=postgres))


def test_capabilities_are_derived_from_services_and_stores():
    config = _durable_config()
    config["global"]["services"] = {
        "management_api": {"enabled": True},
        "access": {"enabled": True},
    }

    capabilities = runtime_capabilities(config)

    assert capabilities.durable_management is True
    assert capabilities.distributed_state is True
    assert capabilities.management_api is True
    assert capabilities.native_access is True
    assert capabilities.file_routing is False


def test_runtime_store_without_management_store_is_rejected():
    config = {
        "version": "v0.3",
        "global": {"stores": {"runtime": {"redis": {"url_env": "ACCESS_RUNTIME_URL"}}}},
    }

    with pytest.raises(ValueError, match=r"requires global\.stores\.management"):
        runtime_capabilities(config)


def test_migration_command_inherits_secret_by_name_without_placing_value_in_argv():
    secret = "postgresql://user:secret@db/control"
    command = build_management_migration_command(
        _durable_config(),
        env_vars={"ACCESS_DATABASE_URL": secret},
        network_name="router-network",
        router_image="router:test",
        container_runtime="docker",
    )

    assert command[command.index("--entrypoint") + 1] == MIGRATION_BINARY
    assert command[command.index("-e") + 1] == "ACCESS_DATABASE_URL"
    assert command[-4:] == ["--dsn-env", "ACCESS_DATABASE_URL", "--timeout", "20s"]
    assert all(secret not in item for item in command)


def test_router_is_not_started_when_explicit_migration_never_succeeds(monkeypatch):
    monkeypatch.setattr(
        "cli.management_migration.get_runtime_images",
        lambda **_kwargs: {"router": "router:test", "envoy": "envoy:test"},
    )
    monkeypatch.setattr(
        "cli.management_migration.get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr("cli.management_migration.MIGRATION_ATTEMPTS", 2)
    monkeypatch.setattr("cli.management_migration.time.sleep", lambda _delay: None)
    attempts = []
    monkeypatch.setattr(
        "cli.management_migration.subprocess.run",
        lambda command, **kwargs: (
            attempts.append((command, kwargs["env"]))
            or SimpleNamespace(returncode=1, stdout="", stderr="unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="Router was not started"):
        run_management_migration(
            _durable_config(),
            env_vars={
                "ACCESS_DATABASE_URL": "postgresql://external/db",
                "ACCESS_RUNTIME_URL": "redis://external:6379/0",
                "PROVIDER_API_KEY": "provider-secret",
                "DASHBOARD_JWT_SECRET": "dashboard-secret",
            },
            network_name="router-network",
            image=None,
            router_image="router:test",
            envoy_image="envoy:test",
            pull_policy="never",
        )

    assert len(attempts) == 2
    for _command, child_env in attempts:
        assert child_env["ACCESS_DATABASE_URL"] == "postgresql://external/db"
        assert "ACCESS_RUNTIME_URL" not in child_env
        assert "PROVIDER_API_KEY" not in child_env
        assert "DASHBOARD_JWT_SECRET" not in child_env
