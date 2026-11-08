"""Canonical model deployment/binding preservation across CLI model boundaries.

The Router validates what a binding names; the CLI keeps its shape.
"""

import copy
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.config_schema.validation import validate_config_structure
from cli.models import ModelBinding, UserConfig
from cli.validator_recipe_contracts import _recipe_name_contract


def binding_document():
    return {
        "version": "v0.5",
        "recipes": [
            {"name": "default", "model_bindings": {"jailbreak": "shared"}},
            {"name": "private", "model_bindings": {"pii": "shared"}},
        ],
        "global": {
            "model_catalog": {
                "deployments": {
                    "shared": {
                        "provider": "model_runtime",
                        "artifact": "vllm-sr/maintained-checkpoint",
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                        "device": "cpu",
                        "input": {"max_tokens": 512, "overflow": "reject"},
                    }
                },
                "admission": {
                    "shared": {
                        "max_concurrency": 2,
                        "max_queue": 8,
                        "queue_timeout_ms": 1000,
                        "on_overflow": "shed",
                    }
                },
            }
        },
        "entrypoints": [
            {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True}
        ],
    }


def test_model_bindings_survive_typed_and_schema_roundtrip():
    document = binding_document()
    assert validate_config_structure(document) == []
    parsed = UserConfig.model_validate(document)
    emitted = parsed.model_dump(mode="json", by_alias=True, exclude_none=True)
    for index in (0, 1):
        assert (
            emitted["recipes"][index]["model_bindings"]
            == document["recipes"][index]["model_bindings"]
        )
    assert emitted["global"] == document["global"]


def test_model_binding_fields_are_strict():
    document = binding_document()
    document["recipes"][0]["model_bindings"]["jailbreak"] = {
        "deployment": "shared",
        "deploymnt": "typo",
    }
    assert validate_config_structure(document)
    with pytest.raises(ValidationError):
        UserConfig.model_validate(document)


def test_bindings_only_default_cannot_be_overwritten_by_explicit_default():
    document = binding_document()
    explicit_default = copy.deepcopy(document["recipes"][0])
    explicit_default["name"] = "default"
    document["recipes"].append(explicit_default)
    _, errors = _recipe_name_contract(UserConfig.model_validate(document))
    assert any("Duplicate recipe name 'default'" in error.message for error in errors)


@pytest.mark.parametrize(
    "written",
    ["shared", " shared ", {"deployment": "shared"}],
)
def test_a_binding_is_written_as_its_deployments_name(written):
    binding = ModelBinding.model_validate(written)

    assert binding.deployment == "shared"
    assert binding.model_dump(mode="json") == "shared"


@pytest.mark.parametrize("field", ["head", "contract", "adapter", "mapping_path"])
def test_a_binding_object_sets_no_field_but_its_deployment(field):
    with pytest.raises(ValidationError):
        ModelBinding.model_validate({"deployment": "shared", field: "pii"})
