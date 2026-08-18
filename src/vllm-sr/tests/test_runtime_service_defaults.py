from pathlib import Path

import pytest
import yaml
from cli.bootstrap import build_bootstrap_config
from cli.commands.runtime_support import resolve_effective_config_path


@pytest.fixture
def write_local_looper_config(tmp_path: Path):
    def _write(endpoint: str | None = None) -> Path:
        looper = {} if endpoint is None else {"endpoint": endpoint}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "listeners": [
                        {
                            "name": "http-generic",
                            "address": "0.0.0.0",
                            "port": 9011,
                        }
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
                "version": "v0.3",
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
        algorithm=None,
        setup_mode=False,
        platform=None,
    )

    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    response_api = effective["global"]["services"]["response_api"]
    assert response_api["enabled"] is True
    assert response_api["store_backend"] == "redis"
    assert response_api["redis"]["address"] == "vllm-sr-redis:6379"
    assert response_api["redis"]["db"] == 0

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
                "version": "v0.3",
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
        algorithm=None,
        setup_mode=False,
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
        algorithm=None,
        setup_mode=False,
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
        algorithm=None,
        setup_mode=False,
        platform=None,
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["global"]["integrations"]["looper"]["endpoint"] == external_endpoint
    )


def test_resolve_effective_config_path_preserves_setup_mode_bootstrap_config(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(build_bootstrap_config(), sort_keys=False))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=True,
        platform=None,
    )

    assert effective_path == config_path


def test_target_neutral_config_skips_all_local_runtime_materialization(
    tmp_path: Path, monkeypatch
):
    source = {
        "version": "v0.3",
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
        algorithm=None,
        setup_mode=False,
        platform=None,
        materialize_local_runtime=False,
    )

    assert effective_path == config_path
    assert yaml.safe_load(effective_path.read_text()) == source
    assert not (tmp_path / ".vllm-sr").exists()
