from pathlib import Path

import pytest
import yaml
from cli.bootstrap import build_bootstrap_config
from cli.commands.runtime_support import (
    append_passthrough_env_vars,
    apply_runtime_mode_env_vars,
    configure_runtime_override_env_vars,
    resolve_effective_config_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_apply_runtime_mode_env_vars_sets_dashboard_readonly_when_requested():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=False,
        readonly=True,
        setup_mode=False,
        platform=None,
    )

    assert env_vars["DASHBOARD_READONLY"] == "true"


def test_apply_runtime_mode_env_vars_skips_dashboard_readonly_in_minimal_mode():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=True,
        readonly=True,
        setup_mode=False,
        platform=None,
    )

    assert env_vars["DISABLE_DASHBOARD"] == "true"
    assert "DASHBOARD_READONLY" not in env_vars


def test_apply_runtime_mode_env_vars_sets_router_log_level_when_requested():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=False,
        readonly=False,
        setup_mode=False,
        platform=None,
        log_level="DEBUG",
    )

    assert env_vars["SR_LOG_LEVEL"] == "debug"


def test_append_passthrough_env_vars_includes_router_logging_settings(monkeypatch):
    monkeypatch.setenv("SR_LOG_LEVEL", "debug")
    monkeypatch.setenv("SR_LOG_ENCODING", "console")

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars)

    assert env_vars["SR_LOG_LEVEL"] == "debug"
    assert env_vars["SR_LOG_ENCODING"] == "console"


def test_append_passthrough_env_vars_masks_looper_shared_secret(monkeypatch, caplog):
    shared_secret = "0123456789abcdef" * 4
    monkeypatch.setenv("VLLM_SR_LOOPER_SHARED_SECRET", shared_secret)

    env_vars: dict[str, str] = {}
    with caplog.at_level("INFO", logger="cli.commands.runtime_support"):
        append_passthrough_env_vars(env_vars)

    assert env_vars["VLLM_SR_LOOPER_SHARED_SECRET"] == shared_secret
    assert shared_secret not in caplog.text
    assert "VLLM_SR_LOOPER_SHARED_SECRET=***" in caplog.text


def test_append_passthrough_env_vars_collects_dashboard_auth_and_masks_secrets(
    monkeypatch, caplog
):
    values = {
        "DASHBOARD_JWT_SECRET": "jwt-secret-value",
        "DASHBOARD_JWT_EXPIRY_HOURS": "12",
        "DASHBOARD_ADMIN_EMAIL": "admin@example.com",
        "DASHBOARD_ADMIN_PASSWORD": "admin-password-value",
        "DASHBOARD_ADMIN_NAME": "Local Admin",
        "DASHBOARD_SECURITY_PROFILE": "production",
        "DASHBOARD_PASSWORD_BLOCKLIST_PATH": "/tmp/password-blocklist.txt",
        "DASHBOARD_PASSWORD_BLOCKLIST_SHA256": "a" * 64,
        "DASHBOARD_ALLOW_OPEN_BOOTSTRAP": "false",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    env_vars: dict[str, str] = {}
    with caplog.at_level("INFO", logger="cli.commands.runtime_support"):
        append_passthrough_env_vars(env_vars)

    assert {name: env_vars[name] for name in values} == values
    assert "jwt-secret-value" not in caplog.text
    assert "admin-password-value" not in caplog.text
    assert "DASHBOARD_JWT_SECRET=***" in caplog.text
    assert "DASHBOARD_ADMIN_PASSWORD=***" in caplog.text


def test_append_passthrough_env_vars_excludes_dashboard_auth_for_nonlocal_target(
    monkeypatch,
):
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "jwt-secret-value")
    monkeypatch.setenv("DASHBOARD_ADMIN_EMAIL", "admin@example.com")

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars, include_dashboard_auth=False)

    assert "DASHBOARD_JWT_SECRET" not in env_vars
    assert "DASHBOARD_ADMIN_EMAIL" not in env_vars


@pytest.mark.parametrize(
    "invalid_secret",
    ["", "too-short", "g" * 64],
    ids=["empty", "wrong-length", "non-hexadecimal"],
)
def test_append_passthrough_env_vars_rejects_invalid_looper_shared_secret(
    monkeypatch, invalid_secret
):
    monkeypatch.setenv("VLLM_SR_LOOPER_SHARED_SECRET", invalid_secret)

    with pytest.raises(ValueError) as exc_info:
        append_passthrough_env_vars({})

    assert "must be exactly 64 hexadecimal characters" in str(exc_info.value)
    if invalid_secret:
        assert invalid_secret not in str(exc_info.value)


def test_resolve_effective_config_path_enables_amd_gpu_by_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_AMD_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "global": {
                    "model_catalog": {
                        "embeddings": {
                            "semantic": {
                                "use_cpu": True,
                                "embedding_config": {"model_type": "mmbert"},
                            }
                        },
                        "modules": {
                            "prompt_guard": {"use_cpu": True},
                            "classifier": {
                                "domain": {"use_cpu": True},
                            },
                        },
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
        platform="amd",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"]
        is False
    )
    assert (
        effective["global"]["model_catalog"]["modules"]["prompt_guard"]["use_cpu"]
        is False
    )
    assert (
        effective["global"]["model_catalog"]["modules"]["classifier"]["domain"][
            "use_cpu"
        ]
        is False
    )


def test_resolve_effective_config_path_enables_nvidia_gpu_by_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_NVIDIA_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_NVIDIA_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "global": {
                    "model_catalog": {
                        "embeddings": {
                            "semantic": {
                                "use_cpu": True,
                                "embedding_config": {"model_type": "mmbert"},
                            }
                        },
                        "modules": {
                            "prompt_guard": {"use_cpu": True},
                            "classifier": {
                                "domain": {"use_cpu": True},
                            },
                        },
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
        platform="nvidia",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"]
        is False
    )
    assert (
        effective["global"]["model_catalog"]["modules"]["prompt_guard"]["use_cpu"]
        is False
    )
    assert (
        effective["global"]["model_catalog"]["modules"]["classifier"]["domain"][
            "use_cpu"
        ]
        is False
    )


def test_resolve_effective_config_path_preserves_nvidia_use_cpu_when_requested(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("VLLM_SR_NVIDIA_PRESERVE_CPU", "1")
    monkeypatch.delenv("VLLM_SR_NVIDIA_FORCE_GPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "global": {
                    "model_catalog": {
                        "embeddings": {
                            "semantic": {
                                "use_cpu": True,
                                "embedding_config": {"model_type": "mmbert"},
                            }
                        },
                        "modules": {
                            "prompt_guard": {"use_cpu": True},
                        },
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
        platform="nvidia",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"]
        is True
    )
    assert (
        effective["global"]["model_catalog"]["modules"]["prompt_guard"]["use_cpu"]
        is True
    )


def test_resolve_effective_config_path_preserves_amd_use_cpu_when_requested(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("VLLM_SR_AMD_PRESERVE_CPU", "1")
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "global": {
                    "model_catalog": {
                        "embeddings": {
                            "semantic": {
                                "use_cpu": True,
                                "embedding_config": {"model_type": "mmbert"},
                            }
                        },
                        "modules": {
                            "prompt_guard": {"use_cpu": True},
                            "classifier": {
                                "domain": {"use_cpu": True},
                            },
                        },
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
        platform="amd",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"]
        is True
    )
    assert (
        effective["global"]["model_catalog"]["modules"]["prompt_guard"]["use_cpu"]
        is True
    )
    assert (
        effective["global"]["model_catalog"]["modules"]["classifier"]["domain"][
            "use_cpu"
        ]
        is True
    )


def test_resolve_effective_config_path_combines_algorithm_and_platform_overrides(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_AMD_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "routing": {"decisions": [{"name": "default"}]},
                "global": {
                    "model_catalog": {
                        "embeddings": {
                            "semantic": {
                                "use_cpu": True,
                                "embedding_config": {"model_type": "mmbert"},
                            }
                        }
                    }
                },
            },
            sort_keys=False,
        )
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm="multi_factor",
        setup_mode=False,
        platform="amd",
    )

    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    assert effective["routing"]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    assert (
        effective["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"]
        is False
    )


def test_resolve_effective_config_path_injects_missing_amd_gpu_defaults_by_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_AMD_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {
                        "name": "http",
                        "address": "0.0.0.0",
                        "port": 8899,
                    }
                ],
                "providers": {
                    "defaults": {"default_model": "test-model"},
                    "models": [
                        {
                            "name": "test-model",
                            "provider_model_id": "test-model",
                            "backend_refs": [{"endpoint": "127.0.0.1:8000"}],
                        }
                    ],
                },
                "routing": {
                    "modelCards": [{"name": "test-model"}],
                    "decisions": [
                        {
                            "name": "default-route",
                            "priority": 1,
                            "rules": {"operator": "AND", "conditions": []},
                            "modelRefs": [{"model": "test-model"}],
                        }
                    ],
                },
                "global": {},
            },
            sort_keys=False,
        )
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform="amd",
    )

    effective = yaml.safe_load(effective_path.read_text())
    model_catalog = effective["global"]["model_catalog"]
    assert model_catalog["embeddings"]["semantic"]["use_cpu"] is False
    assert model_catalog["modules"]["prompt_guard"]["use_cpu"] is False
    assert model_catalog["modules"]["classifier"]["domain"]["use_cpu"] is False
    assert model_catalog["modules"]["classifier"]["pii"]["use_cpu"] is False
    assert model_catalog["modules"]["feedback_detector"]["use_cpu"] is False
    assert (
        model_catalog["modules"]["modality_detector"]["classifier"]["use_cpu"] is False
    )
    assert "bert" not in model_catalog["embeddings"]


def test_resolve_effective_config_path_keeps_bert_deprecated_with_amd_gpu_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_AMD_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    balance_recipe = REPO_ROOT / "deploy" / "recipes" / "balance.yaml"
    config_path.write_text(balance_recipe.read_text(encoding="utf-8"))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform="amd",
    )

    effective = yaml.safe_load(effective_path.read_text())
    model_catalog = effective.get("global", {}).get("model_catalog", {})
    embeddings = model_catalog.get("embeddings", {})
    assert "bert" not in embeddings
    assert embeddings["semantic"]["use_cpu"] is False


def test_configure_runtime_override_env_vars_sets_internal_runtime_path(tmp_path: Path):
    env_vars: dict[str, str] = {}
    source_config = tmp_path / "config.yaml"
    source_config.write_text("version: v0.3\n")
    effective_config = tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective_config.parent.mkdir(parents=True, exist_ok=True)
    effective_config.write_text("version: v0.3\n")

    configure_runtime_override_env_vars(env_vars, source_config, effective_config)

    assert env_vars["VLLM_SR_SOURCE_CONFIG_PATH"] == "/app/config.yaml"
    assert (
        env_vars["VLLM_SR_RUNTIME_CONFIG_PATH"] == "/app/.vllm-sr/runtime-config.yaml"
    )


def test_resolve_effective_config_path_uses_state_root_for_runtime_override(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_STACK_NAME", raising=False)
    state_root = tmp_path / "state"
    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
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

    assert effective_path == state_root / ".vllm-sr" / "runtime-config.yaml"
    assert effective_path.exists()
    assert not (config_dir / ".vllm-sr" / "runtime-config.yaml").exists()

    env_vars: dict[str, str] = {}
    configure_runtime_override_env_vars(env_vars, config_path, effective_path)
    assert (
        env_vars["VLLM_SR_RUNTIME_CONFIG_PATH"] == "/app/.vllm-sr/runtime-config.yaml"
    )


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
    assert router_replay["enabled"] is True
    assert router_replay["store_backend"] == "postgres"
    assert router_replay["postgres"]["host"] == "vllm-sr-postgres"
    assert router_replay["postgres"]["port"] == 5432
    assert router_replay["postgres"]["database"] == "vsr"
    assert router_replay["postgres"]["user"] == "router"
    assert router_replay["postgres"]["password"] == "router-secret"
    assert router_replay["postgres"]["ssl_mode"] == "disable"


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
