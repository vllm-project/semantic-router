import logging
from pathlib import Path

import pytest
import yaml
from cli.commands.runtime_support import resolve_effective_config_path
from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import (
    inject_local_service_runtime_defaults,
    inject_local_store_runtime_defaults,
)
from cli.storage_secrets import (
    POSTGRES_PASSWORD_PLACEHOLDER,
    REDIS_PASSWORD_PLACEHOLDER,
)


@pytest.fixture
def write_local_looper_config(tmp_path: Path):
    def _write(endpoint: str | None = None) -> Path:
        looper = {} if endpoint is None else {"endpoint": endpoint}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.4",
                    "listeners": [
                        {
                            "name": "http-generic",
                            "address": "0.0.0.0",
                            "port": 9011,
                        },
                    ],
                    "global": {"integrations": {"looper": looper}},
                },
                sort_keys=False,
            )
        )
        return config_path

    return _write


def test_resolve_effective_config_path_injects_local_service_runtime_defaults(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.4",
                "listeners": [
                    {
                        "name": "http-8899",
                        "address": "0.0.0.0",
                        "port": 8899,
                    }
                ],
            },
            sort_keys=False,
        )
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        platform=None,
    )

    assert effective_path == tmp_path / ".vllm-sr" / "compiled-bootstrap.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    response_api = effective["global"]["services"]["response_api"]
    assert response_api["enabled"] is True
    assert response_api == {
        "enabled": True,
        "store_backend": "memory",
    }

    router_replay = effective["global"]["services"]["router_replay"]
    assert router_replay == {
        "enabled": False,
        "store_backend": "memory",
    }
    assert effective["global"]["services"]["management_api"] == {
        "bind_address": "0.0.0.0",
        "port": 8080,
    }


def test_resolve_effective_config_path_preserves_explicit_management_listener(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.4",
                "listeners": [{"name": "public", "address": "0.0.0.0", "port": 8899}],
                "global": {
                    "services": {
                        "management_api": {
                            "bind_address": "0.0.0.0",
                            "port": 9090,
                        }
                    }
                },
            },
            sort_keys=False,
        )
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        platform=None,
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert effective["global"]["services"]["management_api"] == {
        "bind_address": "0.0.0.0",
        "port": 9090,
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "http://localhost:8899/v1/chat/completions",
        "http://127.0.0.2:8899/v1/chat/completions",
        "http://[::1]:8899/v1/chat/completions",
    ],
)
def test_resolve_effective_config_path_rewrites_local_looper_endpoint(
    endpoint: str | None,
    write_local_looper_config,
    monkeypatch,
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "test-stack")
    config_path = write_local_looper_config(endpoint)

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        platform=None,
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert effective["global"]["integrations"]["looper"]["endpoint"] == (
        "http://test-stack-vllm-sr-envoy-container:9011/v1/chat/completions"
    )


def test_resolve_effective_config_path_preserves_external_looper_endpoint(
    write_local_looper_config,
    monkeypatch,
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", "test-stack")
    external_endpoint = "https://gateway.example.test/v1/chat/completions"
    config_path = write_local_looper_config(external_endpoint)

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        platform=None,
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["global"]["integrations"]["looper"]["endpoint"] == external_endpoint
    )


def test_target_neutral_config_skips_all_local_runtime_materialization(
    tmp_path: Path, monkeypatch
):
    source = {
        "version": "v0.4",
        "listeners": [{"name": "http", "port": 8899}],
        "global": {
            "services": {},
            "stores": {},
            "integrations": {"looper": {}},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")

    def _unexpected_kb_materialization(*_args, **_kwargs):
        raise AssertionError("Kubernetes config must not materialize local KB state")

    monkeypatch.setattr(
        "cli.commands.runtime_support._sync_runtime_kb_store",
        _unexpected_kb_materialization,
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        platform=None,
        materialize_local_runtime=False,
    )

    assert effective_path == config_path
    assert yaml.safe_load(effective_path.read_text()) == source
    assert not (tmp_path / ".vllm-sr").exists()


def test_generated_runtime_config_references_credentials_instead_of_carrying_them(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.4",
                "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
                "global": {
                    "services": {
                        "response_api": {
                            "enabled": True,
                            "store_backend": "redis",
                        },
                        "router_replay": {
                            "enabled": True,
                            "store_backend": "postgres",
                        },
                    }
                },
            },
            sort_keys=False,
        )
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        platform=None,
    )

    services = yaml.safe_load(effective_path.read_text())["global"]["services"]
    assert services["response_api"]["redis"]["password"] == REDIS_PASSWORD_PLACEHOLDER
    assert (
        services["router_replay"]["postgres"]["password"]
        == POSTGRES_PASSWORD_PLACEHOLDER
    )
    # The generated document names the credential; it never carries one.
    assert "router-secret" not in effective_path.read_text()


def test_local_backend_defaults_skip_an_externally_hosted_endpoint():
    external_redis = "redis.external.example:6379"
    external_postgres = "postgres.external.example"
    config = {
        "version": "v0.4",
        "global": {
            "services": {
                "response_api": {
                    "enabled": True,
                    "store_backend": "redis",
                    "redis": {"address": external_redis},
                },
                "router_replay": {
                    "enabled": True,
                    "store_backend": "postgres",
                    "postgres": {"host": external_postgres, "user": "someone"},
                },
            }
        },
    }

    inject_local_service_runtime_defaults(config, resolve_runtime_stack())

    services = config["global"]["services"]
    assert services["response_api"]["redis"] == {"address": external_redis}
    assert services["router_replay"]["postgres"] == {
        "host": external_postgres,
        "user": "someone",
    }


def test_vector_store_metadata_defaults_skip_an_externally_hosted_postgres():
    metadata = {"host": "postgres.external.example", "password": "operator-owned"}
    config = {
        "version": "v0.4",
        "global": {
            "stores": {
                "response_cache": {"enabled": False},
                "vector_store": {
                    "enabled": True,
                    "metadata_store": "postgres",
                    "metadata_postgres": metadata,
                },
            }
        },
    }

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    assert changed is False
    assert config["global"]["stores"]["vector_store"]["metadata_postgres"] == metadata


def test_managed_metadata_postgres_credential_is_replaced_with_the_placeholder():
    """The store path has its own injection site, so it needs its own proof.

    `_inject_vector_store_metadata_postgres_defaults` calls the credential
    overwrite separately from the service path, so a regression there would not
    show up in any of the `global.services` cases.
    """

    config = {
        "version": "v0.4",
        "global": {
            "stores": {
                "response_cache": {"enabled": False},
                "vector_store": {
                    "enabled": True,
                    "metadata_store": "postgres",
                    "metadata_postgres": {"host": "vllm-sr-postgres"},
                },
            }
        },
    }

    assert inject_local_store_runtime_defaults(config, resolve_runtime_stack()) is True

    metadata = config["global"]["stores"]["vector_store"]["metadata_postgres"]
    assert metadata["password"] == POSTGRES_PASSWORD_PLACEHOLDER


def test_managed_credential_field_is_overwritten_and_reported(caplog):
    stale_password = "router-secret"
    config = {
        "version": "v0.4",
        "global": {
            "services": {
                "response_api": {
                    "enabled": True,
                    "store_backend": "redis",
                    "redis": {"address": "redis:6379", "password": stale_password},
                }
            }
        },
    }

    with caplog.at_level(logging.WARNING):
        changed = inject_local_service_runtime_defaults(config, resolve_runtime_stack())

    redis_config = config["global"]["services"]["response_api"]["redis"]
    assert changed is True
    assert redis_config["password"] == REDIS_PASSWORD_PLACEHOLDER
    warnings = [
        record
        for record in caplog.records
        if "global.services.response_api.redis" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert stale_password not in caplog.text


def test_managed_credential_placeholder_is_reapplied_without_a_warning(caplog):
    config = {
        "version": "v0.4",
        "global": {
            "services": {
                "response_api": {
                    "enabled": True,
                    "store_backend": "redis",
                    "redis": {
                        "address": "redis:6379",
                        "db": 0,
                        "password": REDIS_PASSWORD_PLACEHOLDER,
                    },
                }
            }
        },
    }

    with caplog.at_level(logging.WARNING):
        inject_local_service_runtime_defaults(config, resolve_runtime_stack())

    redis_config = config["global"]["services"]["response_api"]["redis"]
    assert redis_config["password"] == REDIS_PASSWORD_PLACEHOLDER
    assert caplog.records == []
