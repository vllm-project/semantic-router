from pathlib import Path

import yaml

from cli.commands.runtime_support import (
    apply_runtime_mode_env_vars,
    resolve_effective_config_path,
)


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

    effective = yaml.safe_load(effective_path.read_text())
    assert effective["routing"]["decisions"][0]["algorithm"]["type"] == "elo"
    assert (
        effective["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"]
        is False
    )
