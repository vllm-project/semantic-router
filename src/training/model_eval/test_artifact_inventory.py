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
                "decision_model": {"deployment": "primary"},
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
    config["global"]["model_catalog"]["deployments"] = {
        "vela-4b": {
            "provider": "model_runtime",
            "artifact": "vllm-sr/Vela-2.0-4B",
            "device": "cpu",
        }
    }
    config["global"]["model_catalog"]["system"]["decision_model"] = {
        "deployment": "vela-4b"
    }
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


def test_a_vela1_encoder_deployment_fills_the_lines_with_its_artifact(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["deployments"] = {
        "vela1-safety": {
            "provider": "model_runtime",
            "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Safety",
            "device": "cpu",
        }
    }
    config["global"]["model_catalog"]["system"]["decision_model"] = {
        "deployment": "vela1-safety"
    }
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert inventory["jailbreak"].model_path == "models/Vela-1.0-Encoder-307M-Safety"
    assert inventory["domain"].model_path == "models/Vela-1.0-Encoder-307M-Safety"
    assert system_refs(load_config(write_config(tmp_path, config)))["hazard"] == (
        "models/Vela-1.0-Encoder-307M-Hazard"
    )


def test_a_bare_string_decision_model_is_a_load_error(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["system"]["decision_model"] = "Vela-2.0-4B"
    with pytest.raises(ValueError, match="decision_model must be"):
        served_artifacts(load_config(write_config(tmp_path, config)))


def test_an_unknown_deployment_key_is_a_load_error(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["system"]["decision_model"] = {
        "deployment": "missing-deployment"
    }
    with pytest.raises(ValueError, match="unknown deployment"):
        served_artifacts(load_config(write_config(tmp_path, config)))


def test_a_non_model_runtime_deployment_is_a_load_error(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["deployments"] = {
        "remote": {"provider": "openai", "endpoint": "https://api.example.com"}
    }
    config["global"]["model_catalog"]["system"]["decision_model"] = {
        "deployment": "remote"
    }
    with pytest.raises(ValueError, match="must use provider model_runtime"):
        served_artifacts(load_config(write_config(tmp_path, config)))


def test_a_kai_binding_fills_the_lines_with_the_served_artifact(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["deployments"] = {
        "decision-kai": {
            "provider": "model_runtime",
            "artifact": "vllm-sr/Decision-2.0-Kai-0.6B",
            "device": "cpu",
        }
    }
    config["global"]["model_catalog"]["system"]["decision_model"] = {
        "deployment": "decision-kai"
    }
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    # An artifact the built-in registry does not know stays in the repo-id
    # form the deployment declares; the gold probe against DecisionModelSpec
    # pins this.
    for artifact in inventory.values():
        assert artifact.model_path == "vllm-sr/Decision-2.0-Kai-0.6B"
        # The declared Hub ID is the artifact's identity: the org is not
        # prepended a second time, so the drift check, the default revision
        # lookup and baseline matching all see the repository the deployment
        # declared.
        assert artifact.hf_repo == "vllm-sr/Decision-2.0-Kai-0.6B"
    assert system_refs(load_config(write_config(tmp_path, config)))["hazard"] == (
        "models/Vela-1.0-Encoder-307M-Hazard"
    )


def test_the_kai_identity_survives_the_drift_check(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(REF_ONLY_CONFIG))
    config["global"]["model_catalog"]["deployments"] = {
        "decision-kai": {
            "provider": "model_runtime",
            "artifact": "vllm-sr/Decision-2.0-Kai-0.6B",
            "device": "cpu",
        }
    }
    config["global"]["model_catalog"]["system"]["decision_model"] = {
        "deployment": "decision-kai"
    }
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    registry = {
        task: {"id": "vllm-sr/Decision-2.0-Kai-0.6B"}
        for task in ("jailbreak", "intent", "modality")
    }
    # The drift check matches the served artifact on its Hub repository. A
    # registry that measures the repository the deployment declared must not
    # fire, and must not see the org twice either.
    assert registry_drift(inventory, registry) == []


def test_a_registered_alias_keeps_its_registered_repo_id(tmp_path):
    config = yaml.safe_load(yaml.safe_dump(MINIMAL_CONFIG))
    # A deployment declaring vllm-sr/feedback-detector is rewritten to the
    # local path the registry registers it under, and the Hub id has to
    # survive the rewrite: the drift check compares it against the
    # evaluation registry, and the revision lookup and baseline matching
    # resolve through it. Inventing vllm-sr/mom-feedback-detector from the
    # path reports a mismatch against a registry that measures the
    # repository the deployment declared.
    config["global"]["model_catalog"]["system"][
        "prompt_guard"
    ] = "models/mom-feedback-detector"
    config["global"]["model_catalog"]["modules"]["prompt_guard"][
        "model_id"
    ] = "models/mom-feedback-detector"
    config["global"]["model_catalog"]["system"][
        "domain_classifier"
    ] = "models/mom-halugate-detector"
    config["global"]["model_catalog"]["modules"]["classifier"]["domain"][
        "model_id"
    ] = "models/mom-halugate-detector"
    inventory = served_artifacts(load_config(write_config(tmp_path, config)))
    assert inventory["jailbreak"].hf_repo == "vllm-sr/feedback-detector"
    # A registered artifact under a foreign org keeps that org too.
    assert inventory["domain"].hf_repo == "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    registry = {
        "jailbreak": {"id": "vllm-sr/feedback-detector"},
        "intent": {"id": "KRLabsOrg/lettucedect-base-modernbert-en-v1"},
    }
    assert registry_drift(inventory, registry) == []


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
