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
                    "positive_labels": ["jailbreak"],
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
    assert jailbreak.hf_repo == "vllm-sr/mmbert32k-jailbreak-detector-merged"
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
        "jailbreak": {"id": "vllm-sr/mmbert-jailbreak-detector-merged"},
        "intent": {"id": "vllm-sr/mmbert32k-intent-classifier-merged"},
    }
    findings = registry_drift(inventory, registry)
    assert len(findings) == 1
    assert "mmbert32k-jailbreak-detector-merged" in findings[0]
    assert "mmbert-jailbreak-detector-merged" in findings[0]


def test_registry_entry_nothing_serves_is_reported(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    registry = {
        "jailbreak": {"id": "vllm-sr/mmbert32k-jailbreak-detector-merged"},
        "intent": {"id": "vllm-sr/mmbert32k-intent-classifier-merged"},
        "pii": {"id": "vllm-sr/mmbert-pii-detector-merged"},
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


def test_generic_classifier_is_not_assumed_to_be_a_guard():
    config = yaml.safe_load(yaml.safe_dump(MINIMAL_CONFIG))
    safety_path = "models/Vela-1.0-Encoder-307M-Safety"
    config["routing"] = {
        "signals": {
            "classifiers": [
                {
                    "name": "generic-safety",
                    "model_path": safety_path,
                    "labels": ["safe", "unsafe"],
                }
            ]
        }
    }
    inventory = served_artifacts(config)
    assert inventory["jailbreak"].model_path == (
        MINIMAL_CONFIG["global"]["model_catalog"]["system"]["prompt_guard"]
    )
    assert all(site.model_path != safety_path for site in inventory["jailbreak"].sites)
    assert any(safety_path in finding for finding in uncovered_artifacts(config))


def test_the_maintained_config_still_parses():
    """The shipped configuration must stay readable by the inventory."""
    inventory = served_artifacts(load_config(artifact_inventory.DEFAULT_CONFIG))
    assert "jailbreak" in inventory
    assert inventory["jailbreak"].model_path.startswith("models/")


def test_positive_labels_are_read_from_the_gate_site(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, MINIMAL_CONFIG)))
    assert inventory["jailbreak"].positive_labels == ("jailbreak",)
    # An argmax classifier declares none, and must not be reported as a gate.
    assert inventory["domain"].positive_labels == ()


# Since #4721 the maintained modules declare only a model_ref; the artifact
# and threshold they run come from the system table the decision model fills.
REF_ONLY_CONFIG = {
    "global": {
        "model_catalog": {
            "system": {
                "decision_model": "Vela-2.0-0.3B",
                "hazard": "models/Vela-1.0-Encoder-307M-Hazard",
            },
            "modules": {
                "prompt_guard": {
                    "enabled": True,
                    "model_ref": "prompt_guard",
                    "positive_labels": ["jailbreak"],
                },
                "classifier": {
                    "domain": {
                        "enabled": True,
                        "model_ref": "domain_classifier",
                    }
                },
                "modality_detector": {
                    "enabled": True,
                    "classifier": {"model_path": ""},
                },
            },
        }
    }
}


def test_a_ref_only_module_resolves_to_the_decision_model(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, REF_ONLY_CONFIG)))
    assert set(inventory) == {"jailbreak", "domain", "modality"}
    for artifact in inventory.values():
        assert artifact.model_path == "models/Vela-2.0-0.3B"
    assert inventory["jailbreak"].positive_labels == ("jailbreak",)


def test_an_unset_threshold_takes_the_served_models_published_one(tmp_path):
    inventory = served_artifacts(load_config(write_config(tmp_path, REF_ONLY_CONFIG)))
    # module_thresholds of the 0.3B in vela2-decision-model-sizes.json.
    assert inventory["jailbreak"].thresholds == (0.75,)
    assert inventory["domain"].thresholds == (0.28,)


def test_the_decision_model_choice_moves_every_unset_line(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["system"]["decision_model"] = "Vela-2.0-4B"
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert inventory["jailbreak"].model_path == "models/Vela-2.0-4B"
    assert inventory["jailbreak"].thresholds == (0.63,)
    # Hazard is the one line a Vela 2.0 decision model does not move.
    assert system_refs(load_config(write_config(tmp_path, config)))["hazard"] == (
        "models/Vela-1.0-Encoder-307M-Hazard"
    )


def test_an_explicit_system_line_still_wins_over_the_decision_model(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["system"][
        "prompt_guard"
    ] = "models/Vela-1.0-Encoder-307M-Guard"
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert inventory["jailbreak"].model_path == "models/Vela-1.0-Encoder-307M-Guard"
    # A module on a non-Vela-2.0 model runs at the Vela 1.0 thresholds.
    assert inventory["jailbreak"].thresholds == (0.5,)
    assert inventory["domain"].model_path == "models/Vela-2.0-0.3B"


def test_vela1_restores_the_specialists(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["system"] = {"decision_model": "Vela-1.0"}
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert inventory["jailbreak"].model_path == "models/Vela-1.0-Encoder-307M-Guard"
    assert inventory["domain"].model_path == "models/Vela-1.0-Encoder-307M-Domain"
    assert inventory["modality"].model_path == ("models/Vela-1.0-Encoder-307M-Modality")


def test_an_unknown_decision_model_is_a_load_error(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["system"]["decision_model"] = "Vela-3.0-1B"
    with pytest.raises(ValueError, match="not a decision model"):
        served_artifacts(load_config(write_config(tmp_path, config)))


def test_an_unknown_model_ref_is_reported_and_not_served(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["modules"]["prompt_guard"][
        "model_ref"
    ] = "prompt_gard"
    loaded = load_config(write_config(tmp_path, config))
    assert "jailbreak" not in served_artifacts(loaded)
    findings = ref_mismatches(loaded)
    assert len(findings) == 1
    assert "does not define" in findings[0]


def test_a_disabled_ref_only_module_is_not_reported_as_served(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["modules"]["prompt_guard"]["enabled"] = False
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert "jailbreak" not in inventory
