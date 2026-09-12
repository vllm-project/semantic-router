from __future__ import annotations

import json
from pathlib import Path

from cli.config_schema import routing_surface_catalog, schema_document, surface_types
from cli.config_schema.validation import validate_config_structure
from cli.config_schema.views import schema_view
from cli.config_yaml import safe_load_router_config
from cli.main import main
from click.testing import CliRunner
from jsonschema import Draft202012Validator


def test_bundled_schema_exposes_router_surface_catalog() -> None:
    document = schema_document()
    Draft202012Validator.check_schema(document)
    catalog = routing_surface_catalog()

    assert document["$id"].endswith("router-config-v0.3.schema.json")
    assert "setup" not in document["properties"]
    assert "hallucination" in surface_types("signals")
    assert "multi_factor" in surface_types("algorithms")
    assert "shadow_dispatch" in surface_types("plugins")
    assert catalog["validation"]["endpoint"] == "/api/v1/config/validate"


def test_config_schema_command_defaults_to_compact_index() -> None:
    result = CliRunner().invoke(main, ["config", "schema"])

    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["contract_version"] == "vllm-sr/config-schema/v1"
    assert document["default_view"] == "index"
    assert document["schema_etag"].startswith('"sha256:')
    assert "signals" not in document


def test_config_schema_command_supports_full_section_and_surface_views() -> None:
    runner = CliRunner()

    full = runner.invoke(main, ["config", "schema", "--full"])
    assert full.exit_code == 0, full.output
    assert json.loads(full.output)["x-vllm-sr"]["contract_version"] == (
        "vllm-sr/config-schema/v1"
    )

    section = runner.invoke(
        main, ["config", "schema", "--section", "global.router.learning"]
    )
    assert section.exit_code == 0, section.output
    section_document = json.loads(section.output)
    assert section_document["x-vllm-sr-view"]["path"] == "global.router.learning"
    assert section_document["x-vllm-sr-view"]["detail"] == "summary"
    assert section_document["fields"]
    assert "$defs" not in section_document
    assert len(section.output) < len(full.output)

    expanded = runner.invoke(
        main,
        [
            "config",
            "schema",
            "--section",
            "global.router.learning",
            "--expanded",
        ],
    )
    assert expanded.exit_code == 0, expanded.output
    expanded_document = json.loads(expanded.output)
    assert expanded_document["x-vllm-sr-view"]["detail"] == "expanded"
    assert "$defs" in expanded_document

    surface = runner.invoke(main, ["config", "schema", "--surface", "algorithm:static"])
    assert surface.exit_code == 0, surface.output
    assert json.loads(surface.output)["x-vllm-sr-surface"]["type"] == "static"

    incompatible = runner.invoke(
        main, ["config", "schema", "--full", "--section", "global"]
    )
    assert incompatible.exit_code != 0
    assert "use only one" in incompatible.output


def test_config_schema_command_uses_management_origin_and_auth_client(
    monkeypatch,
) -> None:
    captured = {}

    class Client:
        def __init__(self, endpoint, *, timeout, token_env):
            captured.update(
                endpoint=endpoint,
                timeout=timeout,
                token_env=token_env,
            )

        def get_config_schema(self, **kwargs):
            captured["view"] = kwargs
            return type("Response", (), {"payload": {"remote": True}})()

    monkeypatch.setattr("cli.commands.config.RouterManagementClient", Client)

    result = CliRunner().invoke(
        main,
        [
            "config",
            "schema",
            "--endpoint",
            "https://router.example",
            "--token-env",
            "ROUTER_TOKEN",
            "--timeout",
            "7",
            "--surface",
            "algorithm:multi_factor",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"remote": True}
    assert captured == {
        "endpoint": "https://router.example",
        "timeout": 7.0,
        "token_env": "ROUTER_TOKEN",
        "view": {
            "view": "surface",
            "path": None,
            "surface_kind": "algorithm",
            "surface_name": "multi_factor",
            "expanded": False,
        },
    }


def test_config_init_creates_valid_minimal_template_without_overwriting(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "config.yaml"
    runner = CliRunner()

    created = runner.invoke(main, ["config", "init", "--output", str(output)])

    assert created.exit_code == 0, created.output
    config = safe_load_router_config(output.read_text(encoding="utf-8"))
    assert validate_config_structure(config) == []
    assert config["providers"]["models"][0]["name"] == (
        config["routing"]["modelCards"][0]["name"]
    )
    assert config["routing"]["decisions"][0]["modelRefs"][0]["model"] == (
        config["providers"]["models"][0]["name"]
    )

    original = output.read_text(encoding="utf-8")
    refused = runner.invoke(main, ["config", "init", "--output", str(output)])
    assert refused.exit_code != 0
    assert "already exists" in refused.output
    assert output.read_text(encoding="utf-8") == original


def test_python_progressive_index_covers_every_surface_catalog() -> None:
    index = schema_view(schema_document())

    assert {"signal", "algorithm", "plugin", "projection"} == set(index["surfaces"])
    assert "global" in {entry["path"] for entry in index["sections"]}


def test_reference_config_matches_generated_structure() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    with (repository_root / "config" / "config.yaml").open(encoding="utf-8") as stream:
        config = safe_load_router_config(stream)

    assert validate_config_structure(config) == []
