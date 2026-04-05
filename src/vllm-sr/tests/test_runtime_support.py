from pathlib import Path

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


def test_resolve_effective_config_path_applies_amd_gpu_defaults(tmp_path: Path):
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

    assert effective_path != config_path
    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
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


def test_resolve_effective_config_path_combines_algorithm_and_platform_overrides(
    tmp_path: Path,
):
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
        algorithm="elo",
        setup_mode=False,
        platform="amd",
    )

    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    assert effective["routing"]["decisions"][0]["algorithm"]["type"] == "elo"
    assert (
        effective["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"]
        is False
    )


def test_resolve_effective_config_path_injects_missing_amd_gpu_defaults(
    tmp_path: Path,
):
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


def test_resolve_effective_config_path_does_not_reintroduce_deprecated_bert_for_amd(
    tmp_path: Path,
):
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


def test_resolve_effective_config_path_seeds_relative_kb_assets_into_runtime_store(
    tmp_path: Path,
):
    kb_root = tmp_path / "knowledge_bases" / "privacy"
    kb_root.mkdir(parents=True)
    (kb_root / "labels.json").write_text(
        '{"labels":{"safe":{"exemplars":["hello"]}}}', encoding="utf-8"
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "global": {
                    "model_catalog": {
                        "kbs": [
                            {
                                "name": "privacy_kb",
                                "source": {
                                    "path": "knowledge_bases/privacy/",
                                    "manifest": "labels.json",
                                },
                            }
                        ]
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

    assert effective_path == tmp_path / ".vllm-sr" / "runtime-config.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["global"]["model_catalog"]["kbs"][0]["source"]["path"]
        == "knowledge_bases/privacy/"
    )
    assert (
        effective_path.parent / "knowledge_bases" / "privacy" / "labels.json"
    ).exists()


def test_resolve_effective_config_path_seeds_builtin_kb_assets_once(tmp_path: Path):
    test_cases = [
        ("privacy_kb", "knowledge_bases/privacy/", "privacy"),
        ("mmlu_kb", "knowledge_bases/mmlu/", "mmlu"),
    ]

    for kb_name, source_path, bundled_dir in test_cases:
        case_dir = tmp_path / kb_name
        case_dir.mkdir(parents=True, exist_ok=True)
        builtin_manifest = (
            REPO_ROOT / "config" / "knowledge_bases" / bundled_dir / "labels.json"
        )
        assert builtin_manifest.exists()

        config_path = case_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "global": {
                        "model_catalog": {
                            "kbs": [
                                {
                                    "name": kb_name,
                                    "source": {
                                        "path": source_path,
                                        "manifest": "labels.json",
                                    },
                                }
                            ]
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

        assert effective_path == case_dir / ".vllm-sr" / "runtime-config.yaml"
        staged_manifest = (
            effective_path.parent / "knowledge_bases" / bundled_dir / "labels.json"
        )
        assert staged_manifest.exists()
        effective = yaml.safe_load(effective_path.read_text())
        assert (
            effective["global"]["model_catalog"]["kbs"][0]["source"]["path"]
            == source_path
        )

        staged_manifest.unlink()
        staged_manifest.parent.rmdir()
        effective_path = resolve_effective_config_path(
            config_path=config_path,
            algorithm=None,
            setup_mode=False,
            platform=None,
        )
        assert effective_path == case_dir / ".vllm-sr" / "runtime-config.yaml"
        assert not staged_manifest.exists()


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
