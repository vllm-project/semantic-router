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
    registry_drift,
    served_artifacts,
    uncovered_artifacts,
)

GUARD = "vllm-sr/mmbert32k-jailbreak-detector-merged"
INTENT = "vllm-sr/mmbert32k-intent-classifier-merged"

MINIMAL_CONFIG = {
    "global": {
        "model_catalog": {
            "deployments": {
                "guard": {"provider": "model_runtime", "artifact": GUARD},
                "intent": {"provider": "model_runtime", "artifact": INTENT},
            },
            "bindings": {"jailbreak": "guard"},
        }
    },
    "recipes": [
        {
            "name": "default",
            "signals": {
                "jailbreak": [{"name": "attack", "predicate": {"gte": 0.7}}],
                "domains": [{"name": "law", "deployment": "intent"}],
            },
        }
    ],
}


def write_config(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def copy(config):
    return yaml.safe_load(yaml.safe_dump(config))


def test_served_artifacts_are_grouped_by_task(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    assert set(inventory) == {"jailbreak", "domain"}
    jailbreak = inventory["jailbreak"]
    assert jailbreak.artifact_name == "mmbert32k-jailbreak-detector-merged"
    assert jailbreak.hf_repo == GUARD
    assert jailbreak.thresholds == (0.7,)
    assert jailbreak.sites[0].location == "recipes[default].signals.jailbreak[attack]"


def test_a_rule_asks_its_deployment_then_its_bindings_then_the_decision_model(
    tmp_path,
):
    config = copy(MINIMAL_CONFIG)
    recipe = config["recipes"][0]
    recipe["model_bindings"] = {"jailbreak": {"deployment": "intent"}}
    recipe["signals"]["domains"][0].pop("deployment")
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    # The recipe's binding wins over the global one.
    assert inventory["jailbreak"].hf_repo == INTENT
    # A rule nothing binds asks the pre-declared primary decision model.
    assert inventory["domain"].model_path == "models/vela3-fast"

    config["global"]["model_catalog"]["system"] = {
        "decision_model": {"deployment": "guard"}
    }
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert inventory["domain"].hf_repo == GUARD


def test_a_remote_deployment_serves_no_artifact(tmp_path):
    config = copy(MINIMAL_CONFIG)
    config["global"]["model_catalog"]["deployments"]["guard"] = {
        "provider": "systemone",
        "endpoint": "https://systemone.example.com/v1/systemone",
        "model": "vllm-sr/d3-ultra",
    }
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert "jailbreak" not in inventory
    assert "domain" in inventory


def test_two_rules_asking_different_artifacts_for_one_task_is_an_error(tmp_path):
    config = copy(MINIMAL_CONFIG)
    config["recipes"][0]["signals"]["jailbreak"].append(
        {"name": "strict", "deployment": "intent"}
    )
    with pytest.raises(ValueError, match="conflicting artifacts"):
        served_artifacts(load_config(write_config(tmp_path, config)))


def test_registry_drift_names_both_sides(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    registry = {
        "jailbreak": {"id": "vllm-sr/mmbert-jailbreak-detector-merged"},
        "intent": {"id": INTENT},
    }
    findings = registry_drift(inventory, registry)
    assert len(findings) == 1
    assert "mmbert32k-jailbreak-detector-merged" in findings[0]
    assert "mmbert-jailbreak-detector-merged" in findings[0]


def test_registry_entry_nothing_serves_is_reported(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    registry = {
        "jailbreak": {"id": GUARD},
        "intent": {"id": INTENT},
        "pii": {"id": "vllm-sr/mmbert-pii-detector-merged"},
    }
    findings = registry_drift(inventory, registry)
    assert findings == [
        "pii: measured by the evaluation registry but no maintained configuration "
        "loads it"
    ]


def test_an_artifact_with_no_evaluation_task_is_reported(tmp_path):
    config = copy(MINIMAL_CONFIG)
    config["recipes"][0]["signals"]["systemone"] = [
        {"name": "tone", "question": {"type": "noul", "instructions": "Formal?"}}
    ]
    findings = uncovered_artifacts(load_config(write_config(tmp_path, config)))
    assert findings == [
        "models/vela3-fast is loaded at recipes[default].signals.systemone[tone] but "
        "no evaluation task covers it"
    ]


def test_positive_labels_are_read_from_the_gate(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    assert inventory["jailbreak"].positive_labels == ("jailbreak",)
    # An argmax classifier has none, and must not be reported as a gate.
    assert inventory["domain"].positive_labels == ()


def test_the_maintained_config_still_parses():
    """The shipped configuration must stay readable by the inventory."""
    inventory = served_artifacts(load_config(artifact_inventory.DEFAULT_CONFIG))
    assert "jailbreak" in inventory
    assert inventory["jailbreak"].model_path.startswith("models/")
