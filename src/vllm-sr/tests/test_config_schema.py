from __future__ import annotations

import json
from pathlib import Path

from cli.config_schema import routing_surface_catalog, schema_document, surface_types
from cli.config_schema.validation import validate_config_structure
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


def test_config_schema_command_prints_machine_readable_contract() -> None:
    result = CliRunner().invoke(main, ["config", "schema"])

    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["x-vllm-sr"]["contract_version"] == "vllm-sr/config-schema/v1"


def test_reference_config_matches_generated_structure() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    with (repository_root / "config" / "config.yaml").open(encoding="utf-8") as stream:
        config = safe_load_router_config(stream)

    assert validate_config_structure(config) == []
