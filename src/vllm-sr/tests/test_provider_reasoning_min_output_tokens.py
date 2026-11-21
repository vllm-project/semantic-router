"""providers.defaults.reasoning_min_output_tokens in the CLI's model, against the Router's schema."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

CLI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLI_ROOT))

from cli.models import ProviderDefaults  # noqa: E402

SCHEMA = (
    CLI_ROOT.parents[1]
    / "src/semantic-router/pkg/configschema/router-config-v0.5.schema.json"
)


def test_the_field_is_the_routers():
    properties = json.loads(SCHEMA.read_text())["$defs"]["CanonicalProviderDefaults"][
        "properties"
    ]
    assert properties["reasoning_min_output_tokens"] == {
        "type": "integer",
        "minimum": 0,
    }
    assert "reasoning_min_output_tokens" in ProviderDefaults.model_fields


@pytest.mark.parametrize("threshold", [0, 512])
def test_a_non_negative_threshold_is_accepted(threshold):
    defaults = ProviderDefaults(model="m", reasoning_min_output_tokens=threshold)
    assert defaults.reasoning_min_output_tokens == threshold


def test_the_threshold_is_optional():
    assert ProviderDefaults(model="m").reasoning_min_output_tokens is None


@pytest.mark.parametrize("threshold", [-1, "many"])
def test_an_invalid_threshold_is_refused(threshold):
    with pytest.raises(ValidationError):
        ProviderDefaults(model="m", reasoning_min_output_tokens=threshold)
