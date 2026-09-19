"""Global serving defaults are immutable, while consumer policy stays local."""

from copy import deepcopy

import pytest
from cli.config_schema.validation import validate_config_structure
from cli.model_runtime_defaults import effective_model_bindings
from cli.models import UserConfig
from cli.validator_classifier import validate_classifier_contracts
from cli.validator_model_runtime import validate_model_runtime_references
from cli.validator_recipe_contracts import _recipe_name_contract
from cli.validator_safety import validate_safety_contracts
from pydantic import ValidationError


def document():
    return {
        "version": "v0.3",
        "global": {
            "model_catalog": {
                "deployments": {
                    "shared": {
                        "provider": "ort",
                        "device": "rocm:0",
                        "artifact": "models/encoder",
                        "custom_ops_profile": "ck_flash_attention",
                        "input": {"max_tokens": 32768, "overflow": "truncate"},
                    }
                },
                "bindings": {
                    "embedding": {
                        "deployment": "shared",
                        "contract": "embedding.v1",
                        "adapter": "mmbert",
                    },
                    "classifier.risk": {
                        "deployment": "shared",
                        "contract": "label_distribution.v1",
                        "adapter": "modernbert",
                    },
                    "safety.policy": {
                        "deployment": "shared",
                        "contract": "label_distribution.v1",
                        "adapter": "modernbert",
                    },
                },
            }
        },
        "recipes": [
            {
                "name": "active",
                "routing": {
                    "signals": {
                        "classifiers": [
                            {
                                "name": "risk",
                                "type": "local",
                                "labels": ["safe", "unsafe"],
                            }
                        ],
                        "safety": [{"name": "policy", "threshold": 0.8}],
                    }
                },
            },
            {"name": "peer", "routing": {}},
        ],
    }


def test_global_bindings_preserve_authoring_and_resolve_only_local_consumers():
    raw = document()
    original = deepcopy(raw)
    assert validate_config_structure(raw) == []
    config = UserConfig.model_validate(raw)
    before = config.model_dump()
    assert validate_model_runtime_references(config) == []
    assert validate_classifier_contracts(config) == []
    assert validate_safety_contracts(config) == []
    active = effective_model_bindings(config, config.recipes[0].routing)
    peer = effective_model_bindings(config, config.recipes[1].routing)
    assert set(active) == {"embedding", "classifier.risk", "safety.policy"}
    assert set(peer) == {"embedding"}
    assert config.recipes[0].routing.model_bindings == {}
    assert config.model_dump() == before
    assert raw == original
    emitted = config.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert (
        emitted["global"]["model_catalog"]["bindings"]
        == raw["global"]["model_catalog"]["bindings"]
    )


def test_recipe_override_does_not_rewrite_shared_or_peer_serving():
    raw = document()
    local = {
        "deployment": "shared",
        "contract": "embedding.v1",
        "adapter": "mmbert",
        "head": "onnx/other.onnx",
    }
    raw["recipes"][0]["routing"]["model_bindings"] = {"embedding": local}
    config = UserConfig.model_validate(raw)
    assert (
        effective_model_bindings(config, config.recipes[0].routing)["embedding"].head
        == "onnx/other.onnx"
    )
    assert (
        effective_model_bindings(config, config.recipes[1].routing)["embedding"].head
        is None
    )
    assert "head" not in config.global_["model_catalog"]["bindings"]["embedding"]


def test_invalid_global_deployment_and_unknown_fields_are_rejected():
    raw = document()
    raw["global"]["model_catalog"]["bindings"]["embedding"]["deployment"] = "missing"
    errors = validate_model_runtime_references(UserConfig.model_validate(raw))
    assert any(
        error.field == "global.model_catalog.bindings.embedding.deployment"
        for error in errors
    )
    raw = document()
    raw["global"]["model_catalog"]["bindings"]["embedding"]["deploymnt"] = "typo"
    assert validate_config_structure(raw)
    with pytest.raises(ValidationError):
        UserConfig.model_validate(raw)


def test_global_service_scope_cannot_be_declared_as_a_recipe():
    raw = document()
    raw["recipes"][1]["name"] = "@global"
    _, errors = _recipe_name_contract(UserConfig.model_validate(raw))
    assert any("reserved" in error.message for error in errors)
