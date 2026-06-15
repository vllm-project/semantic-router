from pathlib import Path

import yaml

from cli.bootstrap import (
    build_bootstrap_config,
    ensure_bootstrap_workspace,
    is_setup_mode_config,
)


def test_build_bootstrap_config_contains_setup_marker():
    config = build_bootstrap_config()

    assert config["version"] == "v0.3"
    assert config["listeners"][0]["port"] == 8899
    assert config["setup"]["mode"] is True
    assert config["setup"]["state"] == "bootstrap"


def test_ensure_bootstrap_workspace_creates_expected_files(tmp_path: Path):
    config_path = tmp_path / "config.yaml"

    result = ensure_bootstrap_workspace(config_path)

    assert result.created_config is True
    assert result.created_output_dir is True
    assert result.created_defaults is False
    assert result.setup_mode is True
    assert config_path.exists()
    assert (tmp_path / ".vllm-sr").exists()
    assert is_setup_mode_config(config_path) is True

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    assert data["listeners"][0]["port"] == 8899
    assert data["setup"]["mode"] is True


def test_ensure_bootstrap_workspace_preserves_existing_setup_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1",
                "listeners": [
                    {"name": "http-9999", "address": "0.0.0.0", "port": 9999}
                ],
                "setup": {"mode": True, "state": "bootstrap"},
            },
            sort_keys=False,
        )
    )

    result = ensure_bootstrap_workspace(config_path)

    assert result.created_config is False
    assert result.setup_mode is True

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    assert data["listeners"][0]["port"] == 9999
