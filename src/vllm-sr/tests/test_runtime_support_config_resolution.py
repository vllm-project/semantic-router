"""Tests for effective runtime config paths and platform overrides."""

from pathlib import Path

import pytest
import yaml
from cli.commands.runtime_support import (
    configure_runtime_override_env_vars,
    resolve_effective_config_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _domain_on_cpu_config() -> dict:
    return {
        "version": "v0.5",
        "global": {
            "model_catalog": {"modules": {"classifier": {"domain": {"use_cpu": True}}}}
        },
    }


def _domain_use_cpu(effective: dict) -> bool:
    return effective["global"]["model_catalog"]["modules"]["classifier"]["domain"][
        "use_cpu"
    ]


def test_resolve_effective_config_path_adds_no_embedding_settings_on_amd(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_ROCM_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_ROCM_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_domain_on_cpu_config(), sort_keys=False))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform="rocm",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert _domain_use_cpu(effective) is False
    assert "embeddings" not in effective["global"]["model_catalog"]


def test_resolve_effective_config_path_enables_nvidia_gpu_by_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_CUDA_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_CUDA_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_domain_on_cpu_config(), sort_keys=False))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform="cuda",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert _domain_use_cpu(effective) is False


def test_resolve_effective_config_path_preserves_nvidia_use_cpu_when_requested(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("VLLM_SR_CUDA_PRESERVE_CPU", "1")
    monkeypatch.delenv("VLLM_SR_CUDA_FORCE_GPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_domain_on_cpu_config(), sort_keys=False))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform="cuda",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert _domain_use_cpu(effective) is True


def test_resolve_effective_config_path_preserves_amd_use_cpu_when_requested(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("VLLM_SR_ROCM_PRESERVE_CPU", "1")
    monkeypatch.delenv("VLLM_SR_ROCM_FORCE_GPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_domain_on_cpu_config(), sort_keys=False))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform="rocm",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert _domain_use_cpu(effective) is True


def test_resolve_effective_config_path_combines_algorithm_and_platform_overrides(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_ROCM_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_ROCM_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config = _domain_on_cpu_config()
    config["recipes"] = [{"name": "default", "decisions": [{"name": "default"}]}]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm="multi_factor",
        setup_mode=False,
        platform="rocm",
    )

    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["recipes"][0]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    )
    assert _domain_use_cpu(effective) is False


def test_resolve_effective_config_path_injects_no_amd_model_settings_by_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_ROCM_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_ROCM_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.5",
                "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
                "providers": {
                    "defaults": {"model": "test-model"},
                    "models": [
                        {
                            "name": "test-model",
                            "provider_model_id": "test-model",
                            "backend_refs": [
                                {"endpoint": "127.0.0.1:8000", "provider": "vllm"}
                            ],
                        }
                    ],
                },
                "recipes": [
                    {
                        "name": "default",
                        "decisions": [
                            {
                                "name": "default-route",
                                "priority": 1,
                                "rules": {"operator": "AND", "conditions": []},
                                "modelRefs": [{"model": "test-model"}],
                            }
                        ],
                    }
                ],
                "global": {},
            },
            sort_keys=False,
        )
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform="rocm",
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert effective.get("global", {}).get("model_catalog", {}) == {}


def test_resolve_effective_config_path_keeps_a_recipe_catalog_with_amd_gpu_default(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_ROCM_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_ROCM_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    balance_recipe = REPO_ROOT / "config" / "recipes" / "balance" / "config.yaml"
    config_path.write_text(balance_recipe.read_text(encoding="utf-8"))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
        platform="rocm",
    )

    effective = yaml.safe_load(effective_path.read_text())
    source_catalog = yaml.safe_load(balance_recipe.read_text())["global"][
        "model_catalog"
    ]
    model_catalog = effective["global"]["model_catalog"]
    assert model_catalog.get("embeddings") == source_catalog.get("embeddings")
    assert model_catalog["deployments"] == source_catalog["deployments"]


@pytest.mark.parametrize("platform", ["rocm", "cuda"])
def test_effective_config_keeps_the_embedding_deployment_as_authored(
    tmp_path: Path, monkeypatch, platform
):
    monkeypatch.setenv(f"VLLM_SR_{platform.upper()}_FORCE_GPU", "1")
    deployment = {
        "provider": "model_runtime",
        "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
        "device": "cpu",
        "input": {"max_tokens": 128, "overflow": "reject"},
    }
    source = {
        "version": "v0.5",
        "global": {
            "model_catalog": {
                "deployments": {"cpu-embedding": deployment},
                "embeddings": {"default": "cpu-embedding"},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    original = config_path.read_bytes()
    effective_path = resolve_effective_config_path(
        config_path=config_path, algorithm=None, setup_mode=False, platform=platform
    )
    effective = yaml.safe_load(effective_path.read_text())
    catalog = effective["global"]["model_catalog"]
    assert catalog["embeddings"] == {"default": "cpu-embedding"}
    assert "modules" not in catalog
    assert catalog["deployments"] == {"cpu-embedding": deployment}
    assert config_path.read_bytes() == original


def test_configure_runtime_override_env_vars_sets_internal_runtime_path(tmp_path: Path):
    env_vars: dict[str, str] = {}
    source_config = tmp_path / "config.yaml"
    source_config.write_text("version: v0.5\n")
    effective_config = tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective_config.parent.mkdir(parents=True, exist_ok=True)
    effective_config.write_text("version: v0.5\n")

    configure_runtime_override_env_vars(env_vars, source_config, effective_config)

    assert env_vars["VLLM_SR_SOURCE_CONFIG_PATH"] == "/app/.vllm-sr/runtime-config.yaml"
    assert (
        env_vars["VLLM_SR_RUNTIME_CONFIG_PATH"] == "/app/.vllm-sr/runtime-config.yaml"
    )
    assert "VLLM_SR_STATE_ROOT_DIR" not in env_vars


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
                "version": "v0.5",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
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
