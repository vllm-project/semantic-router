import os
import socket

from cli import docker_services
from cli.bootstrap import build_bootstrap_config
from cli.docker_services import _is_port_in_use, docker_start_milvus
from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import (
    inject_local_service_runtime_defaults,
    inject_local_store_runtime_defaults,
)
from cli.storage_backends import detect_required_backends, start_storage_backends


def test_detect_required_backends_uses_canonical_defaults():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    }

    assert detect_required_backends(config) == {"redis", "postgres", "milvus"}


def test_start_storage_backends_passes_state_root_to_milvus(monkeypatch, tmp_path):
    captured = {}

    def fake_start_milvus(network_name, stack_layout, *, state_root_dir=None):
        captured["network_name"] = network_name
        captured["stack_layout"] = stack_layout
        captured["state_root_dir"] = state_root_dir
        return 0, "", ""

    monkeypatch.setattr("cli.storage_backends.docker_start_milvus", fake_start_milvus)

    stack_layout = resolve_runtime_stack()
    started = start_storage_backends(
        {"milvus"},
        "test-network",
        stack_layout,
        state_root_dir=str(tmp_path),
    )

    assert started == {"milvus"}
    assert captured["network_name"] == "test-network"
    assert captured["stack_layout"] is stack_layout
    assert captured["state_root_dir"] == str(tmp_path)


def test_detect_required_backends_skips_external_semantic_cache_milvus():
    config = {
        "version": "v0.3",
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": True,
                    "backend_type": "milvus",
                    "milvus": {
                        "connection": {
                            "host": "external-milvus",
                            "port": 19530,
                        }
                    },
                }
            }
        },
    }

    required = detect_required_backends(config, resolve_runtime_stack())

    assert "milvus" not in required
    assert "redis" in required
    assert "postgres" in required


def test_docker_start_milvus_reuses_network_alias(monkeypatch, tmp_path):
    def fail_run(_cmd, _label):
        raise AssertionError("Milvus container should not start when alias is present")

    monkeypatch.setattr(docker_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_services, "docker_container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(
        docker_services,
        "_running_container_for_network_alias",
        lambda _runtime, _network, _alias: "milvus-semantic-cache",
    )
    monkeypatch.setattr(docker_services, "_is_port_in_use", lambda _port: True)
    monkeypatch.setattr(docker_services, "_run_service_start", fail_run)

    return_code, _stdout, _stderr = docker_start_milvus(
        "test-network",
        resolve_runtime_stack(),
        state_root_dir=str(tmp_path),
    )

    assert return_code == 0


def test_docker_start_milvus_uses_explicit_state_root(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr(docker_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_services, "docker_container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(docker_services, "_is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        docker_services,
        "_run_service_start",
        lambda cmd, _label: commands.append(cmd) or (0, "", ""),
    )

    docker_start_milvus(
        "test-network",
        resolve_runtime_stack(),
        state_root_dir=str(tmp_path),
    )

    expected_data_dir = os.path.abspath(tmp_path / ".vllm-sr" / "milvus-data")
    assert f"{expected_data_dir}:/var/lib/milvus:z" in commands[0]


def test_docker_start_milvus_fails_on_port_conflict_without_container(
    monkeypatch, tmp_path
):
    def fail_run(_cmd, _label):
        raise AssertionError("Milvus container should not start when the port is busy")

    monkeypatch.setattr(docker_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        docker_services, "docker_container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(docker_services, "_is_port_in_use", lambda _port: True)
    monkeypatch.setattr(docker_services, "_run_service_start", fail_run)

    return_code, _stdout, stderr = docker_start_milvus(
        "test-network",
        resolve_runtime_stack(),
        state_root_dir=str(tmp_path),
    )

    assert return_code == 1
    assert "Milvus port" in stderr
    assert "not a running reusable container" in stderr


def test_detect_required_backends_uses_defaults_for_setup_mode_bootstrap_config():
    assert detect_required_backends(build_bootstrap_config()) == {
        "redis",
        "postgres",
        "milvus",
    }


def test_detect_required_backends_respects_explicit_service_disable():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "services": {
                "router_replay": {
                    "enabled": False,
                }
            }
        },
    }

    assert detect_required_backends(config) == {"redis", "milvus"}


def test_detect_required_backends_excludes_milvus_when_memory_override():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": True,
                    "backend_type": "memory",
                }
            }
        },
    }

    required = detect_required_backends(config)
    assert "milvus" not in required
    assert "redis" in required
    assert "postgres" in required


def test_detect_required_backends_excludes_milvus_when_cache_disabled():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": False,
                }
            }
        },
    }

    required = detect_required_backends(config)
    assert "milvus" not in required


def test_detect_required_backends_includes_postgres_for_vector_store_metadata():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": False,
                },
                "vector_store": {
                    "enabled": True,
                    "metadata_store": "postgres",
                },
            }
        },
    }

    assert detect_required_backends(config) == {"redis", "postgres"}


def test_detect_required_backends_ignores_disabled_vector_store_metadata():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "services": {
                "router_replay": {
                    "enabled": False,
                },
                "response_api": {
                    "enabled": False,
                },
            },
            "stores": {
                "semantic_cache": {
                    "enabled": False,
                },
                "vector_store": {
                    "enabled": False,
                    "metadata_store": "postgres",
                },
            },
        },
    }

    assert detect_required_backends(config) == set()


def test_inject_local_service_runtime_defaults_populates_canonical_connections():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    }

    changed = inject_local_service_runtime_defaults(config, resolve_runtime_stack())

    assert changed is True
    services = config["global"]["services"]
    assert services["response_api"]["store_backend"] == "redis"
    assert services["response_api"]["redis"]["address"] == "vllm-sr-redis:6379"
    assert services["response_api"]["redis"]["db"] == 0
    assert services["router_replay"]["store_backend"] == "postgres"
    assert services["router_replay"]["postgres"]["host"] == "vllm-sr-postgres"
    assert services["router_replay"]["postgres"]["database"] == "vsr"
    assert services["router_replay"]["postgres"]["user"] == "router"


def test_inject_local_service_runtime_defaults_keeps_setup_bootstrap_minimal():
    config = build_bootstrap_config()

    changed = inject_local_service_runtime_defaults(config, resolve_runtime_stack())

    assert changed is False
    assert "global" not in config


def test_inject_local_service_runtime_defaults_backfills_blank_backend_fields():
    config = {
        "version": "v0.3",
        "global": {
            "services": {
                "response_api": {
                    "enabled": True,
                    "store_backend": "redis",
                    "redis": {"address": "", "db": 5},
                },
                "router_replay": {
                    "enabled": True,
                    "store_backend": "postgres",
                    "postgres": {"host": "", "user": "custom-user"},
                },
            }
        },
    }

    changed = inject_local_service_runtime_defaults(config, resolve_runtime_stack())

    assert changed is True
    services = config["global"]["services"]
    assert services["response_api"]["redis"]["address"] == "vllm-sr-redis:6379"
    assert services["response_api"]["redis"]["db"] == 5
    assert services["router_replay"]["postgres"]["host"] == "vllm-sr-postgres"
    assert services["router_replay"]["postgres"]["user"] == "custom-user"
    assert services["router_replay"]["postgres"]["password"] == "router-secret"


def test_inject_local_store_runtime_defaults_populates_milvus_connection():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    }

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    assert changed is True
    stores = config["global"]["stores"]
    cache = stores["semantic_cache"]
    assert cache["backend_type"] == "milvus"
    milvus = cache["milvus"]
    conn = milvus["connection"]
    assert conn["host"] == "vllm-sr-milvus"
    assert conn["port"] == 19530  # container gRPC port
    assert conn["database"] == "default"
    assert conn["timeout"] == 30
    coll = milvus["collection"]
    assert coll["name"] == "semantic_cache"
    assert coll["vector_field"]["dimension"] == 768
    search = milvus["search"]
    assert search["params"]["ef"] == 64
    assert search["topk"] == 10
    dev = milvus["development"]
    assert dev["auto_create_collection"] is True


def test_inject_local_store_runtime_defaults_populates_vector_store_metadata_postgres():
    config = {
        "version": "v0.3",
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": False,
                },
                "vector_store": {
                    "enabled": True,
                    "metadata_store": "postgres",
                },
            }
        },
    }

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    assert changed is True
    metadata = config["global"]["stores"]["vector_store"]["metadata_postgres"]
    assert metadata["host"] == "vllm-sr-postgres"
    assert metadata["port"] == 5432
    assert metadata["database"] == "vsr"
    assert metadata["user"] == "router"
    assert metadata["password"] == "router-secret"
    assert metadata["ssl_mode"] == "disable"


def test_inject_local_store_runtime_defaults_backfills_vector_store_metadata_postgres():
    config = {
        "version": "v0.3",
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": False,
                },
                "vector_store": {
                    "enabled": True,
                    "metadata_store": "postgres",
                    "metadata_postgres": {
                        "host": "",
                        "database": "custom",
                    },
                },
            }
        },
    }

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    assert changed is True
    metadata = config["global"]["stores"]["vector_store"]["metadata_postgres"]
    assert metadata["host"] == "vllm-sr-postgres"
    assert metadata["database"] == "custom"
    assert metadata["user"] == "router"


def test_inject_local_store_runtime_defaults_preserves_user_milvus_config():
    config = {
        "version": "v0.3",
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": True,
                    "backend_type": "milvus",
                    "milvus": {
                        "connection": {
                            "host": "custom-milvus-host",
                            "port": 19531,
                        },
                        "collection": {
                            "name": "my_custom_collection",
                        },
                    },
                }
            }
        },
    }

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    milvus = config["global"]["stores"]["semantic_cache"]["milvus"]
    conn = milvus["connection"]
    assert conn["host"] == "custom-milvus-host"
    assert conn["port"] == 19531
    assert conn["database"] == "default"
    assert conn["timeout"] == 30
    assert milvus["collection"]["name"] == "my_custom_collection"
    assert changed is True


def test_inject_local_store_runtime_defaults_skips_memory_backend():
    config = {
        "version": "v0.3",
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": True,
                    "backend_type": "memory",
                }
            }
        },
    }

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    assert changed is False
    assert "milvus" not in config["global"]["stores"]["semantic_cache"]


def test_inject_local_store_runtime_defaults_keeps_setup_bootstrap_minimal():
    config = build_bootstrap_config()

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    assert changed is False


def test_is_port_in_use_returns_false_for_unused_port():
    assert _is_port_in_use(59999) is False


def test_is_port_in_use_returns_true_for_bound_port():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert _is_port_in_use(port) is True
    finally:
        srv.close()
