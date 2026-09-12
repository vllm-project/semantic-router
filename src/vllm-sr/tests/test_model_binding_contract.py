"""Canonical model deployment/binding preservation across CLI model boundaries."""

import copy
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.config_schema.validation import validate_config_structure
from cli.models import UserConfig
from cli.validator_model_runtime import validate_model_runtime_references
from cli.validator_recipe_contracts import _recipe_name_contract


def binding_document():
    return {
        "version": "v0.3",
        "routing": {
            "model_bindings": {
                "embedding": {
                    "deployment": "shared",
                    "contract": "embedding.v1",
                    "adapter": "mmbert",
                }
            }
        },
        "recipes": [
            {
                "name": "private",
                "routing": {
                    "model_bindings": {
                        "pii_classifier": {
                            "deployment": "shared",
                            "contract": "token_spans.v1",
                            "adapter": "mmbert",
                            "head": "pii",
                            "mapping_path": "mappings/pii.json",
                        }
                    }
                },
            }
        ],
        "global": {
            "model_catalog": {
                "deployments": {
                    "shared": {
                        "artifact": "models/maintained-checkpoint",
                        "revision": "pinned-revision",
                        "provider": "candle",
                        "device": "cpu",
                        "precision": "fp32",
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
    }


def test_model_bindings_survive_typed_and_schema_roundtrip():
    document = binding_document()
    assert validate_config_structure(document) == []
    parsed = UserConfig.model_validate(document)
    emitted = parsed.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert emitted["routing"]["model_bindings"] == document["routing"]["model_bindings"]
    assert (
        emitted["recipes"][0]["routing"]["model_bindings"]
        == document["recipes"][0]["routing"]["model_bindings"]
    )
    assert emitted["global"] == document["global"]


def test_model_binding_fields_are_strict():
    document = binding_document()
    document["routing"]["model_bindings"]["embedding"]["deploymnt"] = "typo"
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


def test_recipe_binding_cannot_resolve_a_missing_deployment():
    document = binding_document()
    document["recipes"][0]["routing"]["model_bindings"]["pii_classifier"][
        "deployment"
    ] = "absent"
    errors = validate_model_runtime_references(UserConfig.model_validate(document))
    assert len(errors) == 1
    assert (
        errors[0].field
        == "recipes.private.routing.model_bindings.pii_classifier.deployment"
    )


@pytest.mark.parametrize(
    "consumer,provider,contract,adapter,error_fragment",
    [
        (
            "fact_check_classifier",
            "http",
            "label_distribution.v1",
            "http_classify",
            "no HTTP",
        ),
        (
            "feedback_detector",
            "http",
            "label_distribution.v1",
            "http_classify",
            "no HTTP",
        ),
        (
            "modality_detector",
            "http",
            "label_distribution.v1",
            "http_classify",
            "no HTTP",
        ),
        (
            "hallucination_explainer",
            "http",
            "text_pair_distribution.v1",
            "http_classify",
            "no HTTP",
        ),
        ("hallucination_detector", "ort", "token_spans.v1", "mmbert", "no ORT"),
        (
            "hallucination_explainer",
            "ort",
            "text_pair_distribution.v1",
            "mmbert",
            "no ORT",
        ),
        (
            "prompt_guard",
            "http",
            "label_distribution.v1",
            "http_chat",
            "label_decision.v1",
        ),
        ("unknown", "candle", "score.v1", "mmbert", "Unknown task"),
        ("complexity", "candle", "score.v1", "mmbert", "requires an HTTP"),
        ("complexity", "ort", "label_distribution.v1", "mmbert", "requires an HTTP"),
    ],
)
def test_unsupported_task_provider_binding_fails_before_startup(
    consumer, provider, contract, adapter, error_fragment
):
    document = binding_document()
    document["recipes"] = []
    document["routing"]["model_bindings"] = {
        consumer: {"deployment": "shared", "contract": contract, "adapter": adapter}
    }
    deployment = document["global"]["model_catalog"]["deployments"]["shared"]
    deployment["provider"] = provider
    if provider == "http":
        deployment.pop("artifact")
        deployment.pop("device")
        deployment.pop("precision")
        deployment["external_model"] = "remote"
        document["global"]["model_catalog"]["external"] = [{"name": "remote"}]
    deployment["input"] = {"overflow": "reject"}
    errors = validate_model_runtime_references(UserConfig.model_validate(document))
    assert len(errors) == 1
    assert error_fragment in errors[0].message


def test_classification_budget_is_not_expanded_by_embedding_capacity():
    document = binding_document()
    document["global"]["model_catalog"]["deployments"]["shared"]["input"][
        "max_tokens"
    ] = 513
    errors = validate_model_runtime_references(UserConfig.model_validate(document))
    assert len(errors) == 1
    assert errors[0].field == "recipes.private.routing.model_bindings.pii_classifier"
    assert "512 tokens" in errors[0].message


@pytest.mark.parametrize(
    "changes,fragment",
    [
        ({"provider": "unknown"}, "Unsupported provider"),
        ({"provider": "ort", "device": "cuda:0"}, "incompatible"),
        ({"device": "migraphx:0"}, "incompatible"),
        ({"input": {"max_tokens": -1}}, "negative"),
        ({"external_model": "remote"}, "cannot set external_model"),
    ],
)
def test_deployment_validation_matches_router_before_model_loading(changes, fragment):
    document = binding_document()
    document["global"]["model_catalog"]["deployments"]["shared"].update(changes)
    errors = validate_model_runtime_references(UserConfig.model_validate(document))
    assert any(fragment in error.message for error in errors)


def test_unregistered_custom_local_artifact_remains_valid():
    document = binding_document()
    document["global"]["model_catalog"]["deployments"]["shared"][
        "artifact"
    ] = "/mounted/custom/checkpoint"
    assert validate_model_runtime_references(UserConfig.model_validate(document)) == []
