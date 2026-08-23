import os

from cli import container_services
from cli.bootstrap import build_bootstrap_config
from cli.container_services import (
    container_start_milvus,
    container_start_postgres,
    container_start_redis,
)
from cli.runtime_stack import resolve_runtime_stack
from cli.storage_backends import detect_required_backends, start_storage_backends

# `redis_conf_file` has no default: there is no way to start this stack's Redis
# without a credential file. The reuse checks below answer before the argv is
# ever built, so the path only has to satisfy the required keyword.
UNUSED_REDIS_CONF = "/nonexistent/vllm-sr/redis.conf"


def test_detect_required_backends_uses_canonical_defaults():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
    }

    assert detect_required_backends(config) == {"redis"}
    assert detect_required_backends(config, resolve_runtime_stack()) == {"redis"}


def test_start_storage_backends_passes_state_root_to_milvus(monkeypatch, tmp_path):
    captured = {}

    def fake_start_milvus(network_name, stack_layout, *, state_root_dir=None):
        captured["network_name"] = network_name
        captured["stack_layout"] = stack_layout
        captured["state_root_dir"] = state_root_dir
        return 0, "", ""

    monkeypatch.setattr(
        "cli.storage_backends.container_start_milvus", fake_start_milvus
    )

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
                "response_cache": {
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
    assert "postgres" not in required


def test_detect_required_backends_with_stack_ignores_external_service_stores():
    config = {
        "version": "v0.3",
        "global": {
            "services": {
                "response_api": {
                    "enabled": True,
                    "store_backend": "redis",
                    "redis": {"address": "redis.external.example:6379"},
                },
                "router_replay": {
                    "enabled": True,
                    "store_backend": "postgres",
                    "postgres": {"host": "postgres.external.example"},
                },
            },
            "stores": {"response_cache": {"backend_type": "memory"}},
        },
    }

    assert detect_required_backends(config, resolve_runtime_stack()) == set()


def test_detect_required_backends_with_stack_selects_exact_managed_services():
    config = {
        "version": "v0.3",
        "global": {
            "services": {
                "response_api": {
                    "enabled": True,
                    "store_backend": "redis",
                    "redis": {"address": "vllm-sr-redis:6379"},
                },
                "router_replay": {
                    "enabled": True,
                    "store_backend": "postgres",
                    "postgres": {"host": "postgres"},
                },
            },
            "stores": {"response_cache": {"backend_type": "memory"}},
        },
    }

    assert detect_required_backends(config, resolve_runtime_stack()) == {
        "redis",
        "postgres",
    }


def test_container_start_milvus_reuses_network_alias(monkeypatch, tmp_path):
    def fail_run(_cmd, _label):
        raise AssertionError("Milvus container should not start when alias is present")

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services, "container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(
        container_services,
        "_running_container_for_network_alias",
        lambda _runtime, _network, _alias: "milvus-semantic-cache",
    )
    monkeypatch.setattr(
        container_services,
        "_storage_ports_are_loopback_only",
        lambda _name: (True, ""),
    )
    monkeypatch.setattr(container_services, "_is_port_in_use", lambda _port: True)
    monkeypatch.setattr(container_services, "_run_service_start", fail_run)

    return_code, _stdout, _stderr = container_start_milvus(
        "test-network",
        resolve_runtime_stack(),
        state_root_dir=str(tmp_path),
    )

    assert return_code == 0


def test_container_start_milvus_uses_explicit_state_root(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services, "container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(
        container_services,
        "_running_container_for_network_alias",
        lambda _runtime, _network, _alias: None,
    )
    monkeypatch.setattr(container_services, "_is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        container_services,
        "_run_service_start",
        lambda cmd, _label: commands.append(cmd) or (0, "", ""),
    )

    container_start_milvus(
        "test-network",
        resolve_runtime_stack(),
        state_root_dir=str(tmp_path),
    )

    expected_data_dir = os.path.abspath(tmp_path / ".vllm-sr" / "milvus-data")
    assert f"{expected_data_dir}:/var/lib/milvus:z" in commands[0]


def test_container_start_milvus_uses_explicit_host_hidden_state_mount(
    monkeypatch, tmp_path
):
    commands = []
    host_hidden = tmp_path / "host-state"
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services, "container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(
        container_services,
        "_running_container_for_network_alias",
        lambda _runtime, _network, _alias: None,
    )
    monkeypatch.setattr(container_services, "_is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        container_services,
        "_run_service_start",
        lambda cmd, _label: commands.append(cmd) or (0, "", ""),
    )

    container_start_milvus(
        "test-network",
        resolve_runtime_stack(),
        state_root_dir=str(tmp_path / "container-view"),
        host_hidden_state_dir=str(host_hidden),
    )

    assert f"{host_hidden / 'milvus-data'}:/var/lib/milvus:z" in commands[0]
    assert not any(
        "container-view" in argument and ":/var/lib/milvus" in argument
        for argument in commands[0]
    )


def test_managed_storage_ports_bind_only_to_host_loopback(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services, "container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(
        container_services,
        "_running_container_for_network_alias",
        lambda _runtime, _network, _alias: None,
    )
    monkeypatch.setattr(container_services, "_is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        container_services,
        "_run_service_start",
        lambda cmd, _label: commands.append(cmd) or (0, "", ""),
    )
    layout = resolve_runtime_stack()

    container_start_redis(
        "test-network", layout, redis_conf_file=str(tmp_path / "redis.conf")
    )
    container_start_postgres(
        "test-network",
        layout,
        postgres_password_file=str(tmp_path / "postgres-password"),
    )
    container_start_milvus("test-network", layout, state_root_dir=str(tmp_path))

    assert f"127.0.0.1:{layout.redis_port}:6379" in commands[0]
    assert f"127.0.0.1:{layout.postgres_port}:5432" in commands[1]
    assert f"127.0.0.1:{layout.milvus_port}:19530" in commands[2]


def test_running_storage_with_public_port_binding_is_not_reused(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "stdout": (
                    '{"network_mode":"default","publish_all_ports":false,'
                    '"configured":{"6379/tcp":[{"HostIp":"0.0.0.0",'
                    '"HostPort":"6379"}]},"actual":{"6379/tcp":'
                    '[{"HostIp":"0.0.0.0","HostPort":"6379"}]}}'
                ),
            },
        )(),
    )

    code, _, stderr = container_start_redis(
        "test-network", resolve_runtime_stack(), redis_conf_file=UNUSED_REDIS_CONF
    )

    assert code == 1
    assert "unsafe published storage ports" in stderr
    assert "127.0.0.1 bindings" in stderr


def test_running_storage_with_loopback_binding_remains_reusable(monkeypatch):
    calls = []
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "running")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "inspect" in cmd:
            return type(
                "Result",
                (),
                {
                    "stdout": (
                        '{"network_mode":"default","publish_all_ports":false,'
                        '"configured":{"6379/tcp":[{"HostIp":"127.0.0.1",'
                        '"HostPort":"6379"}]},"actual":{"6379/tcp":'
                        '[{"HostIp":"127.0.0.1","HostPort":"6379"}]}}'
                    ),
                },
            )()
        return type("Result", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(container_services.subprocess, "run", fake_run)

    code, _, stderr = container_start_redis(
        None, resolve_runtime_stack(), redis_conf_file=UNUSED_REDIS_CONF
    )

    assert code == 0
    assert stderr == ""
    assert len(calls) == 2
    assert calls[0][1] == "inspect"
    assert calls[1][1:3] == ["network", "connect"]


def test_running_storage_with_host_network_is_not_reused(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "stdout": (
                    '{"network_mode":"host","publish_all_ports":false,'
                    '"configured":{},"actual":{}}'
                ),
            },
        )(),
    )

    code, _, stderr = container_start_redis(
        "test-network", resolve_runtime_stack(), redis_conf_file=UNUSED_REDIS_CONF
    )

    assert code == 1
    assert "host network mode" in stderr


def test_running_storage_with_container_network_namespace_is_not_reused(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "stdout": (
                    '{"network_mode":"container:peer","publish_all_ports":false,'
                    '"configured":{},"actual":{}}'
                ),
            },
        )(),
    )

    code, _, stderr = container_start_redis(
        "test-network", resolve_runtime_stack(), redis_conf_file=UNUSED_REDIS_CONF
    )

    assert code == 1
    assert "container network mode" in stderr


def test_storage_isolation_accepts_complete_ipv4_loopback_range(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "stdout": (
                    '{"network_mode":"default","publish_all_ports":false,'
                    '"configured":{"6379/tcp":[{"HostIp":"127.12.34.56",'
                    '"HostPort":"6379"}]},"actual":{}}'
                ),
            },
        )(),
    )

    assert container_services._storage_ports_are_loopback_only("redis") == (True, "")


def test_storage_isolation_rejects_missing_inspection_fields(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": "{}"})(),
    )

    safe, detail = container_services._storage_ports_are_loopback_only("redis")

    assert not safe
    assert "inspection is invalid" in detail


def test_running_storage_with_publish_all_ports_is_not_reused(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "stdout": (
                    '{"network_mode":"default","publish_all_ports":true,'
                    '"configured":{},"actual":{}}'
                ),
            },
        )(),
    )

    code, _, stderr = container_start_redis(
        "test-network", resolve_runtime_stack(), redis_conf_file=UNUSED_REDIS_CONF
    )

    assert code == 1
    assert "PublishAllPorts" in stderr


def test_running_storage_with_unsafe_effective_binding_is_not_reused(monkeypatch):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(container_services, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        container_services.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "stdout": (
                    '{"network_mode":"default","publish_all_ports":false,'
                    '"configured":{"6379/tcp":[{"HostIp":"127.0.0.1",'
                    '"HostPort":"6379"}]},"actual":{"6379/tcp":'
                    '[{"HostIp":"0.0.0.0","HostPort":"6379"}]}}'
                ),
            },
        )(),
    )

    code, _, stderr = container_start_redis(
        "test-network", resolve_runtime_stack(), redis_conf_file=UNUSED_REDIS_CONF
    )

    assert code == 1
    assert (
        "inspection failed: actual published port 6379/tcp uses non-loopback "
        "host address '0.0.0.0'"
    ) in stderr


def test_container_start_milvus_fails_on_port_conflict_without_container(
    monkeypatch, tmp_path
):
    def fail_run(_cmd, _label):
        raise AssertionError("Milvus container should not start when the port is busy")

    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services, "container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(
        container_services,
        "_running_container_for_network_alias",
        lambda _runtime, _network, _alias: None,
    )
    monkeypatch.setattr(container_services, "_is_port_in_use", lambda _port: True)
    monkeypatch.setattr(container_services, "_run_service_start", fail_run)

    return_code, _stdout, stderr = container_start_milvus(
        "test-network",
        resolve_runtime_stack(),
        state_root_dir=str(tmp_path),
    )

    assert return_code == 1
    assert "Milvus port" in stderr
    assert "not a running reusable container" in stderr


def test_detect_required_backends_uses_defaults_for_setup_mode_bootstrap_config():
    assert detect_required_backends(build_bootstrap_config()) == {"redis"}


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


def test_detect_required_backends_excludes_milvus_when_memory_override():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "stores": {
                "response_cache": {
                    "enabled": True,
                    "backend_type": "memory",
                }
            }
        },
    }

    required = detect_required_backends(config)
    assert "milvus" not in required
    assert "redis" in required
    assert "postgres" not in required


def test_detect_required_backends_excludes_milvus_when_cache_disabled():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "stores": {
                "response_cache": {
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
                "response_cache": {
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
                "response_cache": {
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
