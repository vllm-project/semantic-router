"""Tests for the served-artifact inventory read from the router configuration."""

import pathlib
import sys

import pytest
import yaml

TEST_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

import artifact_inventory  # noqa: E402
from artifact_inventory import (  # noqa: E402
    load_config,
    ref_mismatches,
    registry_drift,
    served_artifacts,
    system_refs,
    uncovered_artifacts,
)

MINIMAL_CONFIG = {
    "global": {
        "model_catalog": {
            "system": {
                "prompt_guard": "models/mmbert32k-jailbreak-detector-merged",
                "domain_classifier": "models/mmbert32k-intent-classifier-merged",
            },
            "modules": {
                "prompt_guard": {
                    "enabled": True,
                    "model_ref": "prompt_guard",
                    "model_id": "models/mmbert32k-jailbreak-detector-merged",
                    "threshold": 0.7,
                    "jailbreak_mapping_path": (
                        "models/mmbert32k-jailbreak-detector-merged/"
                        "jailbreak_type_mapping.json"
                    ),
                },
                "classifier": {
                    "domain": {
                        "enabled": True,
                        "model_ref": "domain_classifier",
                        "model_id": "models/mmbert32k-intent-classifier-merged",
                        "threshold": 0.5,
                    }
                },
            },
        }
    }
}


def write_config(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_system_table_and_load_sites_agree(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL_CONFIG))
    assert system_refs(config)["prompt_guard"] == (
        "models/mmbert32k-jailbreak-detector-merged"
    )
    assert ref_mismatches(config) == []


def test_served_artifacts_are_grouped_by_task(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    assert set(inventory) == {"jailbreak", "domain"}
    jailbreak = inventory["jailbreak"]
    assert jailbreak.artifact_name == "mmbert32k-jailbreak-detector-merged"
    assert jailbreak.hf_repo == (
        "llm-semantic-router/mmbert32k-jailbreak-detector-merged"
    )
    assert jailbreak.thresholds == (0.7,)


def test_a_disabled_module_is_not_reported_as_served(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(MINIMAL_CONFIG))
    config["global"]["model_catalog"]["modules"]["prompt_guard"]["enabled"] = False
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert "jailbreak" not in inventory


def test_a_module_disagreeing_with_the_system_table_is_reported(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(MINIMAL_CONFIG))
    config["global"]["model_catalog"]["modules"]["prompt_guard"][
        "model_id"
    ] = "models/mmbert-jailbreak-detector-merged"
    findings = ref_mismatches(load_config(write_config(tmp_path, config)))
    assert len(findings) == 1
    assert "system table maps" in findings[0]


def test_two_sites_loading_different_artifacts_for_one_task_is_an_error(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(MINIMAL_CONFIG))
    config["global"]["model_catalog"]["modules"]["second_guard"] = {
        "model_ref": "prompt_guard",
        "model_id": "models/mmbert-jailbreak-detector-merged",
    }
    with pytest.raises(ValueError, match="conflicting artifacts"):
        served_artifacts(load_config(write_config(tmp_path, config)))


def test_registry_drift_names_both_sides(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    registry = {
        "jailbreak": {"id": "llm-semantic-router/mmbert-jailbreak-detector-merged"},
        "intent": {"id": "llm-semantic-router/mmbert32k-intent-classifier-merged"},
    }
    findings = registry_drift(inventory, registry)
    assert len(findings) == 1
    assert "mmbert32k-jailbreak-detector-merged" in findings[0]
    assert "mmbert-jailbreak-detector-merged" in findings[0]


def test_registry_entry_nothing_serves_is_reported(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    registry = {
        "jailbreak": {"id": "llm-semantic-router/mmbert32k-jailbreak-detector-merged"},
        "intent": {"id": "llm-semantic-router/mmbert32k-intent-classifier-merged"},
        "pii": {"id": "llm-semantic-router/mmbert-pii-detector-merged"},
    }
    findings = registry_drift(inventory, registry)
    assert findings == [
        "pii: measured by the evaluation registry but no maintained configuration "
        "loads it"
    ]


def test_an_artifact_with_no_evaluation_task_is_reported(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(MINIMAL_CONFIG))
    config["global"]["model_catalog"]["modules"]["extra"] = {
        "model_id": "models/mom-halugate-detector"
    }
    findings = uncovered_artifacts(load_config(write_config(tmp_path, config)))
    assert len(findings) == 1
    assert "mom-halugate-detector" in findings[0]


def test_the_maintained_config_still_parses():
    """The shipped configuration must stay readable by the inventory."""
    inventory = served_artifacts(load_config(artifact_inventory.DEFAULT_CONFIG))
    assert "jailbreak" in inventory
    assert inventory["jailbreak"].model_path.startswith("models/")
