"""The CLI's copy of the Router's API registry agrees with the generated config schema."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

CLI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLI_ROOT))

from cli.api_registry import (  # noqa: E402
    ALIASES,
    API_NAMES,
    SCHEMAS,
    accepted_names,
    canonical_api,
    upstream_schema,
)
from cli.models import Model  # noqa: E402

SCHEMA = (
    CLI_ROOT.parents[1]
    / "src/semantic-router/pkg/configschema/router-config-v0.5.schema.json"
)


def test_the_names_and_schemas_are_the_routers():
    definitions = json.loads(SCHEMA.read_text())["$defs"]
    model = definitions["CanonicalProviderModel"]["properties"]
    assert model["api_format"]["enum"] == list(accepted_names())
    assert sorted(model["schema"]["enum"]) == sorted(
        {schema for schemas in SCHEMAS.values() for schema in schemas}
    )
    listener = definitions["Listener"]["properties"]["apis"]
    assert listener["propertyNames"]["enum"] == list(accepted_names())
    assert definitions["ListenerAPI"]["properties"]["schema"]["enum"] == [
        "vllm-sr/5.0",
        "vllm-sr/4.0",
    ]


def test_aliases_read_as_their_api():
    assert canonical_api("openai") == "chat_completions"
    assert canonical_api("anthropic") == "messages"
    assert all(canonical_api(name) == name for name in API_NAMES)
    assert set(ALIASES.values()) <= set(API_NAMES)
    with pytest.raises(ValueError, match="api 'realtime' is not supported"):
        canonical_api("realtime")


def test_a_provider_model_names_its_api_and_schema():
    model = Model.model_validate(
        {"name": "m", "api_format": "anthropic", "schema": "anthropic/2023-06-01"}
    )
    assert model.api_format == "messages" and model.schema_ == "anthropic/2023-06-01"
    assert (
        Model.model_validate(
            {"name": "d3", "api_format": "systemone", "schema": "vllm-sr/4.0"}
        ).schema_
        == "vllm-sr/4.0"
    )
    assert upstream_schema("systemone", None) == "systemone/0.2"
    assert upstream_schema("openai", None) == "openai/v1"
    for model, message in (
        (
            {"name": "m", "api_format": "openai", "schema": "vllm-sr/5.0"},
            "is not a schema of api chat_completions",
        ),
        ({"name": "m", "api_format": "realtime"}, "api 'realtime' is not supported"),
        (
            {
                "name": "m",
                "api_format": "systemone",
                "deployment": "local",
                "schema": "vllm-sr/5.0",
            },
            "speaks that deployment's API",
        ),
    ):
        with pytest.raises(ValidationError, match=message):
            Model.model_validate(model)
