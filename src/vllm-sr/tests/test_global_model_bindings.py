"""Global serving defaults keep their authored form; the Router resolves them."""

from copy import deepcopy

import pytest
from cli.config_schema.validation import validate_config_structure
from cli.models import UserConfig
from cli.validator_recipe_contracts import _recipe_name_contract
from pydantic import ValidationError


def document():
    return {
        "version": "v0.5",
        "global": {
            "model_catalog": {
                "deployments": {
                    "shared": {
                        "provider": "model_runtime",
                        "device": "rocm:0",
                        "artifact": "/models/encoder",
                        "input": {"max_tokens": 32768, "overflow": "truncate"},
                    }
                },
                "bindings": {
                    "pii": "shared",
                    "domain": "shared",
                    "jailbreak": "shared",
                },
            }
        },
        "recipes": [
            {"name": "default"},
            {
                "name": "active",
                "signals": {
                    "systemone": [
                        {
                            "name": "risk",
                            "question": {
                                "type": "choice",
                                "instructions": "Is the request risky?",
                                "choices": [{"key": "safe"}, {"key": "unsafe"}],
                            },
                        }
                    ],
                    "safety": [{"name": "policy", "predicate": {"gte": 0.8}}],
                },
            },
            {"name": "peer"},
        ],
        "entrypoints": [
            {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True}
        ],
    }


def test_global_bindings_keep_their_authored_form():
    raw = document()
    original = deepcopy(raw)
    assert validate_config_structure(raw) == []
    config = UserConfig.model_validate(raw)
    assert config.recipes[0].model_bindings == {}
    assert raw == original
    emitted = config.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert (
        emitted["global"]["model_catalog"]["bindings"]
        == raw["global"]["model_catalog"]["bindings"]
    )


def test_a_recipe_binding_overrides_only_its_recipe():
    raw = document()
    raw["global"]["model_catalog"]["deployments"]["local"] = {
        "provider": "model_runtime",
        "artifact": "/models/local-encoder",
    }
    raw["recipes"][0]["model_bindings"] = {"pii": {"deployment": "local"}}
    config = UserConfig.model_validate(raw)
    assert config.recipes[0].model_bindings["pii"].deployment == "local"
    assert config.recipes[1].model_bindings == {}
    assert config.global_["model_catalog"]["bindings"]["pii"] == "shared"


def test_unknown_binding_fields_are_rejected():
    raw = document()
    raw["recipes"][0]["model_bindings"] = {"pii": {"deploymnt": "typo"}}
    assert validate_config_structure(raw)
    with pytest.raises(ValidationError):
        UserConfig.model_validate(raw)


def test_global_service_scope_cannot_be_declared_as_a_recipe():
    raw = document()
    raw["recipes"][1]["name"] = "@global"
    _, errors = _recipe_name_contract(UserConfig.model_validate(raw))
    assert any("reserved" in error.message for error in errors)
