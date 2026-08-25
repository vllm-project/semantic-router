"""Tests for effective runtime config paths and platform overrides."""

from pathlib import Path

import yaml
from cli.commands.runtime_support import resolve_effective_config_path

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_resolve_effective_config_path_preserves_algorithm_and_applies_platform(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_AMD_PRESERVE_CPU", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "recipes": [
                    {
                        "name": "default",
                        "routing": {
                            "decisions": [
                                {
                                    "name": "default",
                                    "algorithm": {"type": "hybrid"},
                                }
                            ]
                        },
                    }
                ],
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
        platform="amd",
    )

    assert effective_path == tmp_path / ".vllm-sr" / "compiled-bootstrap.yaml"
    effective = yaml.safe_load(effective_path.read_text())
    assert (
        effective["recipes"][0]["routing"]["decisions"][0]["algorithm"]["type"]
        == "hybrid"
    )
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
                    "models": [
                        {
                            "name": "test-model",
                            "provider_model_id": "test-model",
                            "backend_refs": [
                                {
                                    "provider": "vllm",
                                    "base_url": "http://127.0.0.1:8000/v1",
                                }
                            ],
                        }
                    ]
                },
                "routing": {
                    "modelCards": [
                        {
                            "name": "test-model",
                            "capabilities": ["chat"],
                        }
                    ]
                },
                "recipes": [
                    {
                        "name": "default",
                        "routing": {
                            "decisions": [
                                {
                                    "name": "default-route",
                                    "priority": 1,
                                    "rules": {
                                        "operator": "AND",
                                        "conditions": [],
                                    },
                                }
                            ]
                        },
                    }
                ],
                "entrypoints": [
                    {
                        "model_names": ["vllm-sr/default", "auto"],
                        "recipe": "default",
                        "assignments": {
                            "default-route": {"models": [{"model": "test-model"}]}
                        },
                    }
                ],
                "global": {},
            },
            sort_keys=False,
        )
    )

    effective_path = resolve_effective_config_path(
        config_path=config_path,
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
    balance_recipe = REPO_ROOT / "config" / "recipes" / "balance" / "config.yaml"
    config_path.write_text(balance_recipe.read_text(encoding="utf-8"))

    effective_path = resolve_effective_config_path(
        config_path=config_path,
        platform="amd",
    )

    effective = yaml.safe_load(effective_path.read_text())
    model_catalog = effective.get("global", {}).get("model_catalog", {})
    embeddings = model_catalog.get("embeddings", {})
    assert "bert" not in embeddings
    assert embeddings["semantic"]["use_cpu"] is False


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
        platform=None,
    )

    assert effective_path == state_root / ".vllm-sr" / "compiled-bootstrap.yaml"
    assert effective_path.exists()
    assert not (config_dir / ".vllm-sr" / "compiled-bootstrap.yaml").exists()
