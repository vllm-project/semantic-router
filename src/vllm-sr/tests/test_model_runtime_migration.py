"""Migration must preserve named backend identity and refuse role ambiguity."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.config_migration import migrate_config_data
from cli.config_schema.validation import validate_config_structure
from cli.models import UserConfig
from cli.validator_model_runtime import validate_model_runtime_references


def legacy_document(protocol="http_classify", name=None):
    model = {
        "model_role": "guardrail",
        "llm_model_name": "served-guard",
        "llm_endpoint": {"address": "guard.example", "port": 8080},
    }
    if name is not None:
        model["name"] = name
    return {
        "version": "v0.3",
        "listeners": [],
        "routing": {},
        "global": {
            "model_catalog": {
                "external": [model],
                "modules": {
                    "prompt_guard": {
                        "enabled": True,
                        "protocol": protocol,
                        "threshold": 0.7,
                        "on_error": "block",
                        "positive_labels": ["unsafe"],
                    }
                },
            }
        },
    }


@pytest.mark.parametrize("protocol", ["http_chat", "http_classify"])
@pytest.mark.parametrize("name", [None, "explicit-guard"])
def test_prompt_guard_protocol_migration_is_explicit_and_idempotent(protocol, name):
    source = legacy_document(protocol, name)
    original = copy.deepcopy(source)
    assert validate_model_runtime_references(UserConfig.model_validate(source))
    result = migrate_config_data(source)
    assert source == original
    catalog = result["global"]["model_catalog"]
    guard = catalog["modules"]["prompt_guard"]
    assert "protocol" not in guard
    assert guard["backend"] == {
        "protocol": protocol,
        "contract": (
            "label_decision.v1" if protocol == "http_chat" else "label_distribution.v1"
        ),
        "model": name or "guardrail_classifier",
    }
    assert catalog["external"][0]["name"] == (name or "guardrail_classifier")
    for field in ("enabled", "threshold", "on_error", "positive_labels"):
        assert (
            guard[field]
            == original["global"]["model_catalog"]["modules"]["prompt_guard"][field]
        )
    assert validate_config_structure(result) == []
    assert validate_model_runtime_references(UserConfig.model_validate(result)) == []
    assert migrate_config_data(result) == result


@pytest.mark.parametrize(
    "scenario", ["missing", "multiple", "conflict", "backend", "variant"]
)
def test_prompt_guard_protocol_migration_rejects_ambiguity(scenario):
    source = legacy_document()
    catalog = source["global"]["model_catalog"]
    if scenario == "missing":
        catalog["external"] = []
    elif scenario == "multiple":
        catalog["external"].append(copy.deepcopy(catalog["external"][0]))
    elif scenario == "conflict":
        catalog["external"].append(
            {"name": "guardrail_classifier", "model_role": "classification"}
        )
    else:
        catalog["modules"]["prompt_guard"][scenario] = (
            {"model": "named"} if scenario == "backend" else "mmbert32k"
        )
    with pytest.raises(ValueError, match="prompt_guard.protocol"):
        migrate_config_data(source)
