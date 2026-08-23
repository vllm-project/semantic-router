from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.bootstrap import (
    LOCAL_REPLICA_ID_ENV,
)
from cli.control_plane_deployment import (
    control_plane_mode,
    local_managed_secret_mounts,
    managed_store_references,
    plan_local_control_plane,
    resolve_local_router_bindings,
)
from cli.control_plane_migration import (
    MIGRATION_BINARY,
    build_control_plane_migration_command,
    run_control_plane_migration,
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


def _managed_config(*, postgres=None, valkey=None):
    return {
        "version": "v0.4",
        "global": {
            "control_plane": {"mode": "managed"},
            "stores": {
                "access": {
                    "type": "postgres",
                    "postgres": (
                        {"dsn_env": "ACCESS_DATABASE_URL"}
                        if postgres is None
                        else postgres
                    ),
                },
                "access_runtime": {
                    "type": "redis",
                    "redis": (
                        {"url_env": "ACCESS_RUNTIME_URL"} if valkey is None else valkey
                    ),
                },
            },
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


def test_standalone_is_dependency_free_and_does_not_mutate_environment():
    runtime_env = {"PUBLIC_SETTING": "on"}

    plan = plan_local_control_plane(
        {"version": "v0.4"}, runtime_env, resolve_runtime_stack()
    )

    assert plan.required_backends == frozenset()
    assert (
        resolve_local_router_bindings(plan, set(), None, resolve_runtime_stack()) == {}
    )
    assert runtime_env == {"PUBLIC_SETTING": "on"}
    assert control_plane_mode({"version": "v0.4"}) == "standalone"


def test_standalone_local_store_credentials_are_resolved_from_committed_state():
    config = {
        "version": "v0.4",
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
    stack = resolve_runtime_stack(stack_name="standalone-secrets")
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


def test_managed_docker_plans_missing_store_refs_then_binds_committed_state():
    runtime_env = {}
    stack = resolve_runtime_stack(stack_name="managed-test")

    plan = plan_local_control_plane(_managed_config(), runtime_env, stack)
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
    assert bindings[LOCAL_REPLICA_ID_ENV] == "local-managed-test"
    assert runtime_env == {}


def test_managed_external_env_refs_bypass_local_store_provisioning():
    runtime_env = {
        "ACCESS_DATABASE_URL": "postgresql://external/db",
        "ACCESS_RUNTIME_URL": "redis://external:6379/0",
    }

    plan = plan_local_control_plane(
        _managed_config(), runtime_env, resolve_runtime_stack()
    )

    assert plan.required_backends == frozenset()
    assert resolve_local_router_bindings(
        plan, set(), None, resolve_runtime_stack()
    ) == {LOCAL_REPLICA_ID_ENV: "local-vllm-sr"}
    assert runtime_env == {
        "ACCESS_DATABASE_URL": "postgresql://external/db",
        "ACCESS_RUNTIME_URL": "redis://external:6379/0",
    }


def test_local_binding_fails_closed_without_committed_storage_state():
    stack = resolve_runtime_stack(stack_name="missing-state")
    plan = plan_local_control_plane(_managed_config(), {}, stack)

    with pytest.raises(ValueError, match="committed credential state"):
        resolve_local_router_bindings(
            plan,
            {"postgres", "redis"},
            None,
            stack,
        )


def test_managed_file_refs_are_exact_read_only_mounts(tmp_path):
    postgres = tmp_path / "postgres-dsn"
    valkey = tmp_path / "valkey-url"
    postgres.write_text("postgresql://external/db", encoding="utf-8")
    valkey.write_text("redis://external:6379/0", encoding="utf-8")
    config = _managed_config(
        postgres={"dsn_file": str(postgres)},
        valkey={"url_file": str(valkey)},
    )

    assert local_managed_secret_mounts(config) == tuple(
        f"{path}:{path}:ro" for path in sorted((str(postgres), str(valkey)))
    )


def test_managed_mounts_ignore_non_control_plane_file_fields(tmp_path):
    postgres = tmp_path / "postgres-dsn"
    valkey = tmp_path / "valkey-url"
    postgres.write_text("postgresql://external/db", encoding="utf-8")
    valkey.write_text("redis://external:6379/0", encoding="utf-8")
    config = _managed_config(
        postgres={"dsn_file": str(postgres)},
        valkey={"url_file": str(valkey)},
    )
    config["global"]["stores"]["response_cache"] = {
        "milvus": {"connection": {"cert_file": "relative/client.pem"}}
    }

    assert local_managed_secret_mounts(config) == tuple(
        f"{path}:{path}:ro" for path in sorted((str(postgres), str(valkey)))
    )


def test_local_bootstrap_authority_mount_tracks_directory_removal(tmp_path):
    postgres = tmp_path / "postgres-dsn"
    valkey = tmp_path / "valkey-url"
    bootstrap_dir = tmp_path / "bootstrap"
    bootstrap_dir.mkdir(mode=0o700)
    bootstrap = bootstrap_dir / "router-token"
    for path, value in (
        (postgres, "postgresql://external/db"),
        (valkey, "redis://external:6379/0"),
        (bootstrap, "one-time-token"),
    ):
        path.write_text(value, encoding="utf-8")
    config = _managed_config(
        postgres={"dsn_file": str(postgres)},
        valkey={"url_file": str(valkey)},
    )
    config["global"]["services"] = {
        "management_api": {"auth": {"bootstrap": {"token_file": str(bootstrap)}}}
    }

    mounts = local_managed_secret_mounts(config)
    assert f"{bootstrap_dir}:{bootstrap_dir}:ro" in mounts
    assert all(str(bootstrap) + ":" not in mount for mount in mounts)

    bootstrap.unlink()
    assert f"{bootstrap_dir}:{bootstrap_dir}:ro" in local_managed_secret_mounts(config)


@pytest.mark.parametrize(
    "postgres",
    [
        {},
        {"dsn_env": "ACCESS_DATABASE_URL", "dsn_file": "/run/secrets/db"},
        {"dsn_env": "lowercase_name"},
        {"dsn_file": "relative/path"},
    ],
)
def test_managed_store_references_reject_ambiguous_or_non_secret_inputs(postgres):
    with pytest.raises(ValueError):
        managed_store_references(_managed_config(postgres=postgres))


def test_migration_command_inherits_secret_by_name_without_placing_value_in_argv():
    secret = "postgresql://user:secret@db/control"
    command = build_control_plane_migration_command(
        _managed_config(),
        env_vars={"ACCESS_DATABASE_URL": secret},
        network_name="managed-network",
        router_image="router:test",
        container_runtime="docker",
    )

    assert command[command.index("--entrypoint") + 1] == MIGRATION_BINARY
    assert command[command.index("-e") + 1] == "ACCESS_DATABASE_URL"
    assert command[-4:] == ["--dsn-env", "ACCESS_DATABASE_URL", "--timeout", "20s"]
    assert all(secret not in item for item in command)


def test_router_is_not_started_when_explicit_migration_never_succeeds(monkeypatch):
    monkeypatch.setattr(
        "cli.control_plane_migration.get_runtime_images",
        lambda **_kwargs: {"router": "router:test", "envoy": "envoy:test"},
    )
    monkeypatch.setattr(
        "cli.control_plane_migration.get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr("cli.control_plane_migration.MIGRATION_ATTEMPTS", 2)
    monkeypatch.setattr("cli.control_plane_migration.time.sleep", lambda _delay: None)
    attempts = []
    monkeypatch.setattr(
        "cli.control_plane_migration.subprocess.run",
        lambda command, **kwargs: (
            attempts.append((command, kwargs["env"]))
            or SimpleNamespace(returncode=1, stdout="", stderr="unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="Router was not started"):
        run_control_plane_migration(
            _managed_config(),
            env_vars={
                "ACCESS_DATABASE_URL": "postgresql://external/db",
                "ACCESS_RUNTIME_URL": "redis://external:6379/0",
                "PROVIDER_API_KEY": "provider-secret",
                "DASHBOARD_JWT_SECRET": "dashboard-secret",
            },
            network_name="managed-network",
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


def test_all_router_images_ship_the_one_shot_migrator():
    repository = Path(__file__).resolve().parents[3]
    dockerfiles = (
        repository / "src/vllm-sr/Dockerfile",
        repository / "src/vllm-sr/Dockerfile.rocm",
        repository / "src/vllm-sr/Dockerfile.cuda",
        repository / "tools/docker/Dockerfile.extproc",
        repository / "tools/docker/Dockerfile.extproc-rocm",
    )
    for dockerfile in dockerfiles:
        contents = dockerfile.read_text(encoding="utf-8")
        assert "./cmd/access-migrate" in contents
        assert "/usr/local/bin/access-migrate" in contents


def test_all_router_images_ship_the_canonical_built_in_recipe_distribution():
    repository = Path(__file__).resolve().parents[3]
    canonical_source = "config/recipes/built-in/latest/mom-v1"
    dockerfiles = {
        repository / "src/vllm-sr/Dockerfile": "/app/recipes/built-in/latest/mom-v1",
        repository
        / "src/vllm-sr/Dockerfile.rocm": "/app/recipes/built-in/latest/mom-v1",
        repository
        / "src/vllm-sr/Dockerfile.cuda": "/app/recipes/built-in/latest/mom-v1",
        repository
        / "tools/docker/Dockerfile.extproc": "/app/config/recipes/built-in/latest/mom-v1",
        repository
        / "tools/docker/Dockerfile.extproc-rocm": "/app/config/recipes/built-in/latest/mom-v1",
    }
    for dockerfile, destination in dockerfiles.items():
        contents = dockerfile.read_text(encoding="utf-8")
        assert (
            f"COPY {canonical_source}/config.yaml {destination}/config.yaml" in contents
        )
        assert (
            f"COPY {canonical_source}/metadata.yaml {destination}/metadata.yaml"
            in contents
        )
        expected_base = "/app/config" if "/tools/docker/" in str(dockerfile) else "/app"
        assert f"ENV VLLM_SR_CONFIG_BASE_DIR={expected_base}" in contents

    dashboard = (repository / "dashboard/backend/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert canonical_source not in dashboard

    chart_templates = repository / "deploy/helm/semantic-router/templates"
    rendered_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in chart_templates.glob("*.yaml")
    )
    assert canonical_source not in rendered_sources
