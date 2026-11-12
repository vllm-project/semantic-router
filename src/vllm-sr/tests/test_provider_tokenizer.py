"""providers.models[].tokenizer in the CLI's model, against the Router's schema."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

CLI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLI_ROOT))

from cli.models import Model  # noqa: E402

SCHEMA = (
    CLI_ROOT.parents[1]
    / "src/semantic-router/pkg/configschema/router-config-v0.5.schema.json"
)


def test_the_field_is_the_routers():
    definitions = json.loads(SCHEMA.read_text())["$defs"]
    properties = definitions["CanonicalProviderModel"]["properties"]
    assert properties["tokenizer"]["type"] == "string"
    assert "token_count" not in properties
    assert "tokenizer" in Model.model_fields
    window = definitions["ContextWindowConfig"]["properties"]
    assert set(window) == {"margin", "proxy_tokenizer"}


@pytest.mark.parametrize(
    "tokenizer",
    [
        "o200k_base",
        "cl100k_base",
        "/models/qwen/tokenizer.json",
        "./tokenizer.json",
        "vendor/qwen.json",
        "Qwen/Qwen3-8B",
        "gpt2",
        "zai-org/GLM-4.6@main",
        "org/model@refs/pr/1",
        "org/model@v1.0",
    ],
)
def test_a_provider_model_names_its_tokenizer(tokenizer):
    model = Model.model_validate(
        {"name": "m", "context_window_size": 1048576, "tokenizer": tokenizer}
    )
    assert model.tokenizer == tokenizer


@pytest.mark.parametrize(
    "tokenizer",
    [
        "",
        "  ",
        "org/model/extra",
        "has space",
        "-org/model",
        "org/model@../../etc",
        "org/model@refs/../../x",
        "org/model@/abs",
        "org/model@a//b",
        "org/model@a\\b",
    ],
)
def test_an_invalid_tokenizer_is_refused(tokenizer):
    with pytest.raises(ValidationError, match="tokenizer"):
        Model.model_validate({"name": "m", "tokenizer": tokenizer})


def test_the_backend_count_is_gone():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Model.model_validate({"name": "m", "token_count": {"endpoint": "/tokenize"}})
