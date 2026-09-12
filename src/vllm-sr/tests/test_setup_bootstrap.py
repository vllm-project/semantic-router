from pathlib import Path

import pytest
import yaml

from cli.bootstrap import (
    build_bootstrap_config,
    ensure_bootstrap_workspace,
    is_setup_mode_config,
)
from cli.config_schema.validation import validate_config_structure
from cli.parser import ConfigParseError, parse_user_config


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


def test_bootstrap_document_parses_without_relaxing_the_router_schema(tmp_path):
    path = tmp_path / "config.yaml"
    ensure_bootstrap_workspace(path)
    original = path.read_text()
    document = yaml.safe_load(original)

    config = parse_user_config(str(path), log_summary=False)

    assert config.setup == document["setup"]
    assert config.setup["mode"] is True
    assert path.read_text() == original
    assert is_setup_mode_config(path)
    assert any("config.setup" in error for error in validate_config_structure(document))
    assert (
        validate_config_structure(
            {key: value for key, value in document.items() if key != "setup"}
        )
        == []
    )


@pytest.mark.parametrize(
    "setup, error",
    [
        (True, "config.setup: must be an object"),
        ({"mode": "false"}, "config.setup.mode: must be a boolean"),
        ({"mode": 1}, "config.setup.mode: must be a boolean"),
        ({"mode": True, "state": []}, "config.setup.state: must be a string"),
        ({"mode": True, "typo": True}, "config.setup.typo: unknown setup metadata"),
    ],
)
def test_bootstrap_document_rejects_invalid_setup_metadata(tmp_path, setup, error):
    document = build_bootstrap_config()
    document["setup"] = setup
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigParseError, match=error):
        parse_user_config(str(path), log_summary=False)


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("version", "v0.1", "version"),
        ("unknown_root", True, "config.unknown_root"),
        ("providers", {"models": [{"name": "model", "typo": True}]}, "typo"),
        ("signals", {}, "Deprecated config fields"),
    ],
)
def test_setup_envelope_does_not_bypass_canonical_validation(
    tmp_path, field, value, error
):
    document = build_bootstrap_config()
    document[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))

    with pytest.raises(ConfigParseError, match=error):
        parse_user_config(str(path), log_summary=False)


@pytest.mark.parametrize("mode", [False, "false", 1])
def test_setup_mode_requires_an_explicit_boolean(tmp_path, mode):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"setup": {"mode": mode}}))
    assert is_setup_mode_config(path) is False
