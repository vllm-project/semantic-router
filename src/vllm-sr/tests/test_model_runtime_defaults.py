"""Offline model references share Go-generated deployment defaults."""

from copy import deepcopy

from cli.config_schema import schema_document
from cli.model_runtime_defaults import effective_model_deployments
from cli.models import UserConfig


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
                                "deployment": "hazard",
                                "labels": ["violence", "self_harm"],
                            }
                        ]
                    },
                },
            }
        ],
    }


def _defaults():
    return schema_document()["$defs"]["CanonicalModelCatalog"]["properties"][
        "deployments"
    ]["default"]


def test_named_default_resolves_without_mutating_authoring_config():
    raw = hazard_document()
    original = deepcopy(raw)
    config = UserConfig.model_validate(raw)
    before = config.model_dump()
    deployments = effective_model_deployments(config)
    assert deployments == _defaults()
    assert config.routing.model_bindings == {}
    deployments["hazard"]["input"]["max_tokens"] = 1
    assert effective_model_deployments(config) == _defaults()
    assert config.model_dump() == before
    assert raw == original


def test_explicit_amd_override_replaces_whole_entry_and_remains_immutable():
    raw = hazard_document()
    deployment = {
        "provider": "model_runtime",
        "artifact": "/models/operator-hazard",
        "device": "rocm:0",
        "input": {"max_tokens": 32768, "overflow": "reject"},
    }
    raw["global"] = {"model_catalog": {"deployments": {"hazard": deployment}}}
    original = deepcopy(raw)
    config = UserConfig.model_validate(raw)
    resolved = effective_model_deployments(config)
    assert resolved["hazard"] == deployment
    assert "revision" not in resolved["hazard"]
    resolved["hazard"]["input"]["max_tokens"] = 1
    assert effective_model_deployments(config)["hazard"] == deployment
    assert raw == original


def test_an_override_replaces_the_whole_default_and_null_clears_them():
    raw = hazard_document()
    raw["global"] = {
        "model_catalog": {"deployments": {"hazard": {"provider": "model_runtime"}}}
    }
    assert effective_model_deployments(UserConfig.model_validate(raw))["hazard"] == {
        "provider": "model_runtime"
    }
    raw["global"] = {"model_catalog": {"deployments": None}}
    assert effective_model_deployments(UserConfig.model_validate(raw)) == {}


def test_unbound_config_does_not_materialize_or_activate_defaults():
    config = UserConfig.model_validate({"version": "v0.3", "routing": {}})
    assert config.global_ is None
    assert config.routing.model_bindings == {}
    assert config.routing.signals.classifiers == []
