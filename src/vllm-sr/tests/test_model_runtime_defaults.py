"""Offline model references share Go-generated deployment defaults."""

from copy import deepcopy

import pytest
from cli.config_schema import schema_document
from cli.model_runtime_defaults import effective_model_deployments
from cli.models import UserConfig
from cli.validator_classifier import validate_classifier_contracts
from cli.validator_model_runtime import (
    project_classifier_rule,
    validate_model_runtime_references,
)


def hazard_document():
    return {
        "version": "v0.3",
        "routing": {},
        "recipes": [
            {
                "name": "care",
                "routing": {
                    "signals": {
                        "classifiers": [
                            {
                                "name": "content-risk",
                                "type": "local",
                                "labels": ["violence", "self_harm"],
                            }
                        ]
                    },
                    "model_bindings": {
                        "classifier.content-risk": {
                            "deployment": "hazard",
                            "contract": "label_scores.v1",
                            "adapter": "modernbert",
                            "operating_point": {
                                "path": "operating_point.json",
                                "sha256": "e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5",
                            },
                        }
                    },
                },
            }
        ],
    }


def test_named_default_resolves_without_mutating_authoring_config():
    raw = hazard_document()
    original = deepcopy(raw)
    config = UserConfig.model_validate(raw)
    before = config.model_dump()
    deployments = effective_model_deployments(config)
    defaults = schema_document()["$defs"]["CanonicalModelCatalog"]["properties"][
        "deployments"
    ]["default"]
    assert deployments == defaults
    assert validate_model_runtime_references(config) == []
    assert validate_classifier_contracts(config) == []
    profile = config.recipes[0].routing
    source_rule = profile.signals.classifiers[0]
    projected = project_classifier_rule(
        source_rule, profile.model_bindings, deployments
    )
    assert projected.model_path == defaults["hazard"]["artifact"]
    assert projected.use_cpu is True
    assert not source_rule.model_path
    assert config.routing.model_bindings == {}
    deployments["hazard"]["input"]["max_tokens"] = 1
    assert effective_model_deployments(config) == defaults
    assert config.model_dump() == before
    assert raw == original


def test_explicit_amd_override_replaces_whole_entry_and_remains_immutable():
    raw = hazard_document()
    deployment = {
        "artifact": "models/operator-hazard",
        "provider": "ort",
        "device": "migraphx:0",
        "precision": "native",
        "input": {"max_tokens": 32768, "overflow": "reject"},
    }
    raw["global"] = {"model_catalog": {"deployments": {"hazard": deployment}}}
    original = deepcopy(raw)
    config = UserConfig.model_validate(raw)
    resolved = effective_model_deployments(config)
    assert resolved["hazard"] == deployment
    assert "revision" not in resolved["hazard"]
    assert validate_model_runtime_references(config) == []
    assert validate_classifier_contracts(config) == []
    profile = config.recipes[0].routing
    projected = project_classifier_rule(
        profile.signals.classifiers[0], profile.model_bindings, resolved
    )
    assert projected.model_path == deployment["artifact"]
    assert projected.use_cpu is False
    resolved["hazard"]["input"]["max_tokens"] = 1
    assert effective_model_deployments(config)["hazard"] == deployment
    assert raw == original


@pytest.mark.parametrize("scenario", ["unknown", "incomplete_override", "cleared"])
def test_deployment_defaults_do_not_rescue_invalid_explicit_bindings(scenario):
    raw = hazard_document()
    if scenario == "unknown":
        raw["recipes"][0]["routing"]["model_bindings"]["classifier.content-risk"][
            "deployment"
        ] = "not-registered"
    elif scenario == "incomplete_override":
        raw["global"] = {
            "model_catalog": {"deployments": {"hazard": {"provider": "ort"}}}
        }
    else:
        raw["global"] = {"model_catalog": {"deployments": None}}
    config = UserConfig.model_validate(raw)
    errors = validate_model_runtime_references(config)
    assert errors
    if scenario == "incomplete_override":
        assert effective_model_deployments(config)["hazard"] == {"provider": "ort"}
        assert "requires artifact" in str(errors[0])
    else:
        assert any("Unknown model deployment" in str(error) for error in errors)


def test_unbound_config_does_not_materialize_or_activate_defaults():
    config = UserConfig.model_validate({"version": "v0.3", "routing": {}})
    assert validate_model_runtime_references(config) == []
    assert config.global_ is None
    assert config.routing.model_bindings == {}
    assert config.routing.signals.classifiers == []
