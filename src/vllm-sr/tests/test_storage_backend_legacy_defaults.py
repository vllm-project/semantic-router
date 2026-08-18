"""Legacy local-store defaults and host-port probe tests."""

import socket

from cli.bootstrap import build_bootstrap_config
from cli.container_services import _is_port_in_use
from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import inject_local_store_runtime_defaults


def test_inject_local_store_runtime_defaults_migrates_legacy_cache_key():
    config = {
        "version": "v0.3",
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": True,
                    "backend_type": "milvus",
                }
            }
        },
    }

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    stores = config["global"]["stores"]
    assert changed is True
    assert "semantic_cache" not in stores
    assert stores["response_cache"]["milvus"]["connection"]["host"] == "vllm-sr-milvus"


def test_inject_local_store_runtime_defaults_skips_memory_backend():
    config = {
        "version": "v0.3",
        "global": {
            "stores": {
                "response_cache": {
                    "enabled": True,
                    "backend_type": "memory",
                }
            }
        },
    }

    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())

    assert changed is False
    assert "milvus" not in config["global"]["stores"]["response_cache"]


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
