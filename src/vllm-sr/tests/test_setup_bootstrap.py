from pathlib import Path

import pytest
import yaml

from cli.bootstrap import (
    build_bootstrap_config,
    ensure_bootstrap_workspace,
    is_setup_mode_config,
)
from cli.config_schema.validation import validate_config_structure
from cli.container_management_listener import _managed_management_listener
from cli.container_start import _render_split_envoy_config
from cli.parser import ConfigParseError, parse_user_config
from cli.runtime_stack import resolve_runtime_stack
from cli.validator_model_runtime import validate_model_runtime_references


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


def test_bootstrap_metadata_preserves_split_runtime_preparation(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    ensure_bootstrap_workspace(config_path)
    original = config_path.read_text()
    document = yaml.safe_load(original)

    # Bootstrap metadata belongs to the CLI/Dashboard, not the public Router
    # contract. Both container preparation consumers still need its listeners.
    assert any(
        "config.setup:" in error for error in validate_config_structure(document)
    )
    parsed = parse_user_config(str(config_path), log_summary=False)
    assert parsed.setup == document["setup"]
    stack = resolve_runtime_stack(stack_name="bootstrap-contract", port_offset=1000)
    assert _managed_management_listener(str(config_path), stack) == {
        "bind_address": "0.0.0.0",
        "port": 8080,
        "host_port": 9080,
    }
    envoy_path = tmp_path / "envoy.yaml"
    _render_split_envoy_config(str(config_path), str(envoy_path), stack)
    envoy = yaml.safe_load(envoy_path.read_text())
    listeners = envoy["static_resources"]["listeners"]
    assert any(
        listener["address"]["socket_address"]["port_value"] == 8899
        for listener in listeners
    )
    assert stack.router_container_name in envoy_path.read_text()
    assert config_path.read_text() == original
    assert is_setup_mode_config(config_path)


@pytest.mark.parametrize(
    "extra", [{"unknown_router_field": True}, {"global": {"typo": 1}}]
)
def test_bootstrap_metadata_does_not_bypass_router_schema(tmp_path: Path, extra):
    config_path = tmp_path / "config.yaml"
    document = build_bootstrap_config()
    document.update(extra)
    config_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigParseError, match="schema validation failed"):
        parse_user_config(str(config_path), log_summary=False)


def test_bootstrap_metadata_still_requires_a_mapping(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    document = build_bootstrap_config()
    document["setup"] = "not metadata"
    config_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigParseError, match=r"config\.setup: must be an object"):
        parse_user_config(str(config_path), log_summary=False)


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


@pytest.mark.parametrize("owner", ["default", "private"])
def test_bootstrap_metadata_preserves_typed_model_bindings(tmp_path: Path, owner):
    document = build_bootstrap_config()
    binding = {
        "deployment": "shared",
        "contract": "embedding.v1",
        "adapter": "mmbert",
    }
    profile = {"model_bindings": {"embedding": binding}}
    if owner == "default":
        document["routing"] = profile
    else:
        document["recipes"] = [{"name": owner, "routing": profile}]
    document["global"] = {
        "model_catalog": {
            "deployments": {
                "shared": {"provider": "candle", "artifact": "models/mmbert"}
            }
        }
    }
    path = tmp_path / "config.yaml"
    original = yaml.safe_dump(document)
    path.write_text(original)

    parsed = parse_user_config(str(path), log_summary=False)
    parsed_profile = parsed.routing if owner == "default" else parsed.recipes[0].routing

    assert (
        parsed_profile.model_bindings["embedding"].model_dump(exclude_none=True)
        == binding
    )
    assert validate_model_runtime_references(parsed) == []
    assert parsed.global_ == document["global"]
    assert parsed.setup == document["setup"]
    assert path.read_text() == original


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
