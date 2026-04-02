from cli.bootstrap import build_bootstrap_config
from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import inject_local_service_runtime_defaults
from cli.storage_backends import detect_required_backends


def test_detect_required_backends_uses_canonical_defaults():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    }

    assert detect_required_backends(config) == {"redis", "postgres"}


def test_detect_required_backends_skips_setup_mode_bootstrap_config():
    assert detect_required_backends(build_bootstrap_config()) == set()


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

    assert detect_required_backends(config) == {"redis"}


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
