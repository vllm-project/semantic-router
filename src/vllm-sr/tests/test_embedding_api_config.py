"""The CLI preserves the generated standalone embedding API contract."""

from copy import deepcopy

import pytest
from cli.config_schema.validation import validate_config_structure
from cli.main import main
from cli.models import UserConfig
from click.testing import CliRunner


@pytest.mark.parametrize("enabled", [False, True])
def test_global_embedding_api_round_trip_preserves_scope(enabled):
    raw = {
        "version": "v0.3",
        "routing": {},
        "recipes": [{"name": "isolated", "routing": {}}],
        "global": {"services": {"api": {"embeddings": {"enabled": enabled}}}},
    }
    before = deepcopy(raw)
    assert validate_config_structure(raw) == []
    parsed = UserConfig.model_validate(raw)
    exported = parsed.model_dump(by_alias=True, exclude_none=True)
    assert exported["global"] == raw["global"]
    assert parsed.routing.model_bindings == {}
    assert parsed.recipes[0].routing.model_bindings == {}
    assert "services" not in exported["recipes"][0]
    assert raw == before


def test_embedding_api_opt_in_is_not_inserted_when_omitted():
    raw = {"version": "v0.3", "routing": {}}
    assert validate_config_structure(raw) == []
    parsed = UserConfig.model_validate(raw)
    assert "global" not in parsed.model_dump(by_alias=True, exclude_none=True)


@pytest.mark.parametrize(
    "global_settings",
    [
        {"services": {"api": {"embeddings": {"enabled": "yes"}}}},
        {"services": {"api": {"embeddings": {"enable": True}}}},
        {"router": {"api": {"embeddings": {"enabled": True}}}},
    ],
)
def test_embedding_api_rejects_invalid_fields_and_wrong_scope(global_settings):
    errors = validate_config_structure(
        {"version": "v0.3", "routing": {}, "global": global_settings}
    )
    assert errors
    assert any("global." in error for error in errors)


def test_cli_exposes_embedding_api_opt_in_from_generated_schema():
    result = CliRunner().invoke(
        main,
        ["config", "schema", "--section", "global.services.api.embeddings"],
    )
    assert result.exit_code == 0, result.output
    assert '"enabled"' in result.output
