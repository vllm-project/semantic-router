"""Generic rule bindings select exact recipe-owned execution resources."""

import copy

import pytest
from cli.config_schema.validation import validate_config_structure
from cli.models import UserConfig
from cli.validator_classifier import validate_classifier_contracts
from cli.validator_model_runtime import (
    project_classifier_rule,
    validate_model_runtime_references,
)
from pydantic import ValidationError


def generic_document(provider="http", rule_type="local", named=False):
    rule = {"name": "risk.tenant", "type": rule_type, "labels": ["safe", "unsafe"]}
    if rule_type == "llm":
        rule["instructions"] = "Score all labels."
    deployment = {"provider": provider, "artifact": "models/selected"}
    adapter = "modernbert"
    if provider == "http":
        deployment = {"provider": provider, "external_model": "selected"}
        adapter = "http_chat" if rule_type == "llm" else "http_classify"
    profile = {
        "signals": {"classifiers": [rule]},
        "model_bindings": {
            "classifier.risk.tenant": {
                "deployment": "selected",
                "contract": "label_distribution.v1",
                "adapter": adapter,
            }
        },
    }
    document = {
        "version": "v0.3",
        "global": {
            "model_catalog": {
                "deployments": {"selected": deployment},
                "external": [
                    {
                        "name": "selected",
                        "model_role": "classification",
                        "llm_model_name": "served",
                        "llm_endpoint": {"address": "localhost", "port": 8080},
                    }
                ],
            }
        },
    }
    if named:
        document["recipes"] = [{"name": "private", "routing": profile}]
    else:
        document["routing"] = profile
    return document


@pytest.mark.parametrize("provider", ["candle", "ort", "http"])
@pytest.mark.parametrize("rule_type", ["local", "sequence_classifier"])
@pytest.mark.parametrize("named", [False, True])
def test_generic_binding_resolves_provider_without_default_selector(
    provider, rule_type, named
):
    document = generic_document(provider, rule_type, named)
    assert validate_config_structure(document) == []
    config = UserConfig.model_validate(document)
    assert validate_model_runtime_references(config) == []
    assert validate_classifier_contracts(config) == []
    profile = config.recipes[0].routing if named else config.routing
    rule = profile.signals.classifiers[0]
    original = rule.model_dump()
    resolved = project_classifier_rule(
        rule, profile.model_bindings, config.global_["model_catalog"]["deployments"]
    )
    assert resolved.type == ("sequence_classifier" if provider == "http" else "local")
    assert resolved.model == ("selected" if provider == "http" else None)
    assert resolved.model_path == (None if provider == "http" else "models/selected")
    assert rule.model_dump() == original


@pytest.mark.parametrize(
    "scenario",
    ["foreign rule", "mapping", "chat sequence", "local llm", "decision contract"],
)
def test_generic_binding_rejects_unknown_scope_and_incompatible_extraction(scenario):
    document = generic_document()
    binding = document["routing"]["model_bindings"]["classifier.risk.tenant"]
    if scenario == "foreign rule":
        document["recipes"] = [
            {
                "name": "private",
                "routing": {
                    "model_bindings": copy.deepcopy(
                        document["routing"]["model_bindings"]
                    )
                },
            }
        ]
    elif scenario == "mapping":
        binding["mapping_path"] = "ignored.json"
    elif scenario == "chat sequence":
        binding["adapter"] = "http_chat"
    elif scenario == "local llm":
        document = generic_document("candle", "llm")
    else:
        binding["contract"] = "label_decision.v1"
    assert validate_model_runtime_references(UserConfig.model_validate(document))


def test_generic_llm_binding_replaces_obsolete_named_endpoint():
    document = generic_document("http", "llm")
    document["routing"]["signals"]["classifiers"][0]["model"] = "removed-endpoint"
    config = UserConfig.model_validate(document)
    assert validate_model_runtime_references(config) == []
    assert validate_classifier_contracts(config) == []


def test_unbound_classifier_still_requires_execution_selector():
    document = generic_document()
    document["routing"]["model_bindings"] = {}
    with pytest.raises(ValidationError, match="model_path"):
        UserConfig.model_validate(document)


def test_multiple_local_multiclass_rules_roundtrip_without_process_limit():
    document = generic_document("candle")
    profile = document["routing"]
    first = profile["signals"]["classifiers"][0]
    first["labels"] = ["safe", "unsafe", "uncertain"]
    second = copy.deepcopy(first)
    second["name"] = "other.rule"
    second["model_path"] = "models/independent"
    profile["signals"]["classifiers"].append(second)
    config = UserConfig.model_validate(document)
    assert validate_model_runtime_references(config) == []
    assert validate_classifier_contracts(config) == []
    emitted = config.model_dump(mode="json", by_alias=True, exclude_none=True)
    reloaded = UserConfig.model_validate(emitted)
    assert reloaded.routing.signals.classifiers == config.routing.signals.classifiers
    assert reloaded.routing.signals.classifiers[1].model_path == "models/independent"


def test_bound_llm_rejects_unscored_parser():
    document = generic_document("http", "llm")
    document["global"]["model_catalog"]["external"][0]["parser_type"] = "qwen3guard"
    errors = validate_classifier_contracts(UserConfig.model_validate(document))
    assert any("parser_type must be json" in error.message for error in errors)
