"""The CLI admits the same typed Safety deployments as the Go preparation path."""

import copy

import pytest
from cli.config_schema.validation import validate_config_structure
from cli.models import UserConfig
from cli.validator_model_runtime import validate_model_runtime_references
from cli.validator_safety import validate_safety_contracts


def safety_document():
    return {
        "version": "v0.3",
        "routing": {
            "signals": {
                "safety": [
                    {
                        "name": "risk",
                        "threshold": 0.5,
                        "hazard": {
                            "labels": ["a", "b"],
                            "categories": ["b"],
                            "threshold": 0.8,
                        },
                    }
                ]
            },
            "model_bindings": {
                "safety.risk": {
                    "deployment": "local",
                    "contract": "label_distribution.v1",
                    "adapter": "modernbert",
                },
                "safety.risk.hazard": {
                    "deployment": "local",
                    "contract": "label_scores.v1",
                    "adapter": "modernbert",
                    "head": "hazard",
                },
            },
        },
        "global": {
            "model_catalog": {
                "deployments": {
                    "local": {
                        "provider": "ort",
                        "artifact": "models/qualified",
                        "device": "rocm:0",
                        "precision": "native",
                        "custom_ops_profile": "ck_flash_attention",
                        "input": {"max_tokens": 32768},
                    }
                },
                "modules": {
                    "safety": {
                        "safety": {
                            "model_id": "",
                            "max_sequence_length": 512,
                            "window": {"size": 1024, "overlap": 128},
                        }
                    }
                },
            }
        },
    }


def test_safety_binding_and_rocm_profile_survive_public_schema_and_validation():
    document = safety_document()
    original = copy.deepcopy(document)
    assert validate_config_structure(document) == []
    parsed = UserConfig.model_validate(document)
    assert validate_model_runtime_references(parsed) == []
    assert validate_safety_contracts(parsed) == []
    assert document == original
    document["global"]["model_catalog"]["deployments"]["local"]["input"][
        "max_tokens"
    ] = 512
    assert validate_safety_contracts(UserConfig.model_validate(document))


@pytest.mark.parametrize(
    "change",
    ["wrong_contract", "foreign_head", "mapping", "cpu_profile", "bad_profile"],
)
def test_safety_invalid_binding_is_rejected_before_launch(change):
    document = safety_document()
    bindings = document["routing"]["model_bindings"]
    deployment = document["global"]["model_catalog"]["deployments"]["local"]
    if change == "wrong_contract":
        bindings["safety.risk.hazard"]["contract"] = "label_distribution.v1"
    elif change == "foreign_head":
        bindings["safety.foreign"] = bindings.pop("safety.risk")
    elif change == "mapping":
        bindings["safety.risk"]["mapping_path"] = "unused.json"
    elif change == "cpu_profile":
        deployment["device"] = "cpu"
    else:
        deployment["custom_ops_profile"] = "untrusted"
    assert validate_model_runtime_references(UserConfig.model_validate(document))
