"""Tests for effective runtime config paths."""

from pathlib import Path

import yaml
from cli.commands.runtime_support import (
    configure_runtime_override_env_vars,
    resolve_effective_config_path,
    sync_runtime_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_resolve_effective_config_path_applies_the_algorithm_override(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config = {
        "version": "v0.5",
        "recipes": [{"name": "default", "decisions": [{"name": "default"}]}],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm="multi_factor",
        setup_mode=False,
    )

    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["recipes"][0]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    )


def test_sync_runtime_config_takes_the_call_the_dashboard_and_router_make(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config = {
        "version": "v0.5",
        "recipes": [{"name": "default", "decisions": [{"name": "default"}]}],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    effective_path = sync_runtime_config(config_path, algorithm="multi_factor")

    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["recipes"][0]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    )


def test_resolve_effective_config_path_injects_no_model_settings(tmp_path: Path):
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
    )

    effective = yaml.safe_load(effective_path.read_text())
    assert effective.get("global", {}).get("model_catalog", {}) == {}


def test_resolve_effective_config_path_keeps_a_recipe_catalog(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    balance_recipe = REPO_ROOT / "config" / "recipes" / "balance" / "config.yaml"
    config_path.write_text(balance_recipe.read_text(encoding="utf-8"))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        algorithm=None,
        setup_mode=False,
    )

    effective = yaml.safe_load(effective_path.read_text())
    source_catalog = yaml.safe_load(balance_recipe.read_text())["global"][
        "model_catalog"
    ]
    model_catalog = effective["global"]["model_catalog"]
    assert model_catalog.get("embeddings") == source_catalog.get("embeddings")
    assert model_catalog["deployments"] == source_catalog["deployments"]


def test_effective_config_keeps_the_embedding_deployment_as_authored(tmp_path: Path):
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
                "embeddings": {"default": {"deployment": "cpu-embedding"}},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    original = config_path.read_bytes()
    effective_path = resolve_effective_config_path(
        config_path=config_path, algorithm=None, setup_mode=False
    )
    effective = yaml.safe_load(effective_path.read_text())
    catalog = effective["global"]["model_catalog"]
    assert catalog["embeddings"] == {"default": {"deployment": "cpu-embedding"}}
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
    )

    assert effective_path == state_root / ".vllm-sr" / "runtime-config.yaml"
    assert effective_path.exists()
    assert not (config_dir / ".vllm-sr" / "runtime-config.yaml").exists()

    env_vars: dict[str, str] = {}
    configure_runtime_override_env_vars(env_vars, config_path, effective_path)
    assert (
        env_vars["VLLM_SR_RUNTIME_CONFIG_PATH"] == "/app/.vllm-sr/runtime-config.yaml"
    )
