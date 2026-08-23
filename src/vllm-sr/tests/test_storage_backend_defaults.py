"""Canonical local-store defaults and host-port probe tests."""

import socket

from cli.container_services import _is_port_in_use
from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import inject_local_store_runtime_defaults


def test_inject_local_store_runtime_defaults_skips_memory_backend():
    config = {
        "version": "v0.4",
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


def test_is_port_in_use_returns_false_for_unused_port():
    assert _is_port_in_use(59999) is False


def test_is_port_in_use_returns_true_for_bound_port():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert _is_port_in_use(port) is True
    finally:
        server.close()
