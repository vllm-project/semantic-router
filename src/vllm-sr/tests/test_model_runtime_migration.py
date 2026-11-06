"""Migration must preserve named backend identity and refuse role ambiguity."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.config_migration import migrate_config_data
from cli.config_migration_notes import MigrationNotes
from cli.config_schema.validation import validate_config_structure


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
    result = migrate_config_data(source)
    assert source == original
    catalog = result["global"]["model_catalog"]
    guard = catalog["modules"]["prompt_guard"]
    assert "protocol" not in guard
    assert "backend" not in guard
    assert "external" not in catalog
    deployment = name or "guardrail_classifier"
    assert catalog["deployments"][deployment] == (
        {
            "provider": "openai_chat",
            "endpoint": "http://guard.example:8080/v1",
            "model": "served-guard",
        }
        if protocol == "http_chat"
        else {
            "provider": "classify",
            "endpoint": "http://guard.example:8080",
            "model": "served-guard",
        }
    )
    assert catalog["bindings"] == {"jailbreak": deployment}
    for field in ("enabled", "threshold", "on_error", "positive_labels"):
        assert (
            guard[field]
            == original["global"]["model_catalog"]["modules"]["prompt_guard"][field]
        )
    assert validate_config_structure(result) == []
    assert migrate_config_data(result) == result


def test_prompt_guard_protocol_migration_keeps_chat_settings_on_the_backend():
    source = legacy_document("http_chat")
    source["global"]["model_catalog"]["external"][0]["parser_type"] = "json"
    notes = MigrationNotes()
    result = migrate_config_data(source, notes=notes)
    catalog = result["global"]["model_catalog"]
    assert catalog["modules"]["prompt_guard"]["backend"] == {
        "protocol": "http_chat",
        "contract": "label_decision.v1",
        "model": "guardrail_classifier",
    }
    assert catalog["external"][0]["name"] == "guardrail_classifier"
    assert "deployments" not in catalog
    assert any("parser_type" in note.message for note in notes)
    assert validate_config_structure(result) == []


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
    with pytest.raises(ValueError, match=r"prompt_guard\.protocol"):
        migrate_config_data(source)
