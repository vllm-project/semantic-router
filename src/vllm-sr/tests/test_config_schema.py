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
    assert "setup" in document["properties"]
    assert "hallucination" in surface_types("signals")
    assert "multi_factor" in surface_types("algorithms")
    assert "shadow_dispatch" in surface_types("plugins")
    assert catalog["validation"]["endpoint"] == "/config/router/validate"


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
    assert len(section.output) < len(full.output)

    surface = runner.invoke(main, ["config", "schema", "--surface", "algorithm:static"])
    assert surface.exit_code == 0, surface.output
    assert json.loads(surface.output)["x-vllm-sr-surface"]["type"] == "static"

    incompatible = runner.invoke(
        main, ["config", "schema", "--full", "--section", "global"]
    )
    assert incompatible.exit_code != 0
    assert "use only one" in incompatible.output


def test_python_progressive_index_covers_every_surface_catalog() -> None:
    index = schema_view(schema_document())

    assert {"signal", "algorithm", "plugin", "projection"} == set(index["surfaces"])
    assert "global" in {entry["path"] for entry in index["sections"]}


def test_reference_config_matches_generated_structure() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    with (repository_root / "config" / "config.yaml").open(encoding="utf-8") as stream:
        config = safe_load_router_config(stream)

    assert validate_config_structure(config) == []
