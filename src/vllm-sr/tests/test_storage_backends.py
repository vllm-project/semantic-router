import socket

from cli.bootstrap import build_bootstrap_config
from cli.docker_services import _is_port_in_use
from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import (
    inject_local_service_runtime_defaults,
    inject_local_store_runtime_defaults,
)
from cli.storage_backends import detect_required_backends


def test_detect_required_backends_uses_canonical_defaults():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    }

    assert detect_required_backends(config) == {"redis", "postgres", "milvus"}


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
    assert conn["host"] == "host.docker.internal"
    assert conn["port"] == 19530  # DEFAULT_MILVUS_PORT
    assert conn["database"] == "default"
    assert conn["timeout"] == 30
    coll = milvus["collection"]
    assert coll["name"] == "semantic_cache"
    assert coll["vector_field"]["dimension"] == 384
    search = milvus["search"]
    assert search["params"]["ef"] == 64
    assert search["topk"] == 10
    dev = milvus["development"]
    assert dev["auto_create_collection"] is True


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
