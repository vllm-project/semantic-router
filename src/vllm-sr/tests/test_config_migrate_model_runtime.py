"""`vllm-sr config migrate`: removed native providers, legacy models and NLI paths."""

import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_migration import migrate_config_data  # noqa: E402
from cli.config_migration_model_runtime import runtime_device  # noqa: E402
from cli.config_migration_notes import MigrationNotes  # noqa: E402
from cli.main import main  # noqa: E402

REMOVED_FIELDS = {"precision", "custom_ops_profile", "compilation_cache_dir"}
REMOVED_PROVIDERS = {"candle", "ort", "openvino"}


def _config(global_config=None, routing=None, recipes=None):
    config = {
        "version": "v0.3",
        "listeners": [],
        "providers": {"defaults": {"model": "general"}},
        "routing": {"modelCards": [{"name": "general"}], **(routing or {})},
        "global": global_config or {},
    }
    if recipes is not None:
        config["recipes"] = recipes
    return config


def _migrate(config):
    notes = MigrationNotes()
    return migrate_config_data(config, notes), notes


def _catalog(migrated):
    return migrated["global"]["model_catalog"]


def _note_paths(notes):
    return {note.path for note in notes}


@pytest.mark.parametrize(
    ("provider", "device", "expected"),
    [
        ("candle", None, "cpu"),
        ("candle", "cpu", "cpu"),
        ("candle", "cuda:1", "cuda:1"),
        ("candle", "metal:0", "mps"),
        ("ort", "rocm:0", "rocm:0"),
        ("ort", "migraphx:2", "rocm:2"),
        ("openvino", None, "cpu"),
        ("openvino", "CPU", "cpu"),
        ("openvino", "GPU.0", "cpu"),
    ],
)
def test_legacy_devices_map_to_runtime_accelerators(provider, device, expected):
    assert runtime_device(provider, device)[0] == expected


def test_openvino_gpu_device_reports_cpu_fallback():
    device, problem = runtime_device("openvino", "GPU.0")

    assert device == "cpu"
    assert "xpu" in problem


def test_native_deployments_become_model_runtime_deployments():
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "deployments": {
                        "risk": {
                            "provider": "candle",
                            "artifact": "models/Vela-1.0-Encoder-307M-Hazard",
                            "device": "cpu",
                            "precision": "fp32",
                            "input": {"max_tokens": 32768, "overflow": "reject"},
                        },
                        "ranker": {
                            "provider": "ort",
                            "artifact": "models/Vela-1.0-Encoder-307M-Reranker",
                            "device": "migraphx:0",
                            "precision": "fp16",
                            "custom_ops_profile": "ck_flash_attention",
                            "compilation_cache_dir": "/var/cache/migraphx",
                        },
                        "embedder": {
                            "provider": "openvino",
                            "artifact": "/models/vela-embedding-ir",
                            "device": "GPU",
                        },
                    }
                }
            }
        )
    )

    deployments = _catalog(migrated)["deployments"]
    assert deployments["risk"] == {
        "provider": "model_runtime",
        "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Hazard",
        "device": "cpu",
        "input": {"max_tokens": 32768, "overflow": "reject"},
    }
    assert deployments["ranker"] == {
        "provider": "model_runtime",
        "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Reranker",
        "device": "rocm:0",
        "profile": "max_speed",
    }
    assert deployments["embedder"] == {
        "provider": "model_runtime",
        "artifact": "/models/vela-embedding-ir",
        "device": "cpu",
    }
    paths = _note_paths(notes)
    assert "global.model_catalog.deployments.ranker.precision" in paths
    assert "global.model_catalog.deployments.ranker.custom_ops_profile" in paths
    assert "global.model_catalog.deployments.ranker.compilation_cache_dir" in paths
    assert "global.model_catalog.deployments.embedder.device" in paths
    assert not any(note.action_required for note in notes)


def test_existing_profile_wins_over_fp16_precision():
    migrated, _ = _migrate(
        _config(
            {
                "model_catalog": {
                    "deployments": {
                        "fast": {
                            "provider": "candle",
                            "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Domain",
                            "device": "cuda:0",
                            "precision": "fp16",
                            "profile": "batching",
                        }
                    }
                }
            }
        )
    )

    assert _catalog(migrated)["deployments"]["fast"]["profile"] == "batching"


def test_legacy_alias_artifact_moves_to_its_vela_replacement_and_drops_revision():
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "deployments": {
                        "guard": {
                            "provider": "candle",
                            "artifact": "models/mom-jailbreak-classifier",
                            "revision": "0" * 40,
                        }
                    }
                }
            }
        )
    )

    guard = _catalog(migrated)["deployments"]["guard"]
    assert guard["artifact"] == "vllm-sr/Vela-1.0-Encoder-307M-Guard"
    assert "revision" not in guard
    assert any("benign / jailbreak" in note.message for note in notes)


def test_relative_custom_artifact_needs_operator_action():
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "deployments": {
                        "custom": {
                            "provider": "candle",
                            "artifact": "models/my-domain-classifier",
                        }
                    }
                }
            }
        )
    )

    assert (
        _catalog(migrated)["deployments"]["custom"]["artifact"]
        == "models/my-domain-classifier"
    )
    actions = [note for note in notes if note.action_required]
    assert [note.path for note in actions] == [
        "global.model_catalog.deployments.custom.artifact"
    ]


def test_graph_heads_and_the_explainer_binding_leave_task_bindings():
    deployments = {
        "domain": {
            "provider": "ort",
            "artifact": "models/Vela-1.0-Encoder-307M-Domain",
            "device": "rocm:0",
        },
        "nli": {
            "provider": "candle",
            "artifact": "models/mom-halugate-explainer",
        },
    }
    graph_binding = {
        "deployment": "domain",
        "contract": "label_distribution.v1",
        "adapter": "modernbert",
        "head": "onnx/model_rocm_32k.onnx",
    }
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "deployments": deployments,
                    "bindings": {
                        "domain_classifier": dict(graph_binding),
                        "hallucination_explainer": {
                            "deployment": "nli",
                            "contract": "text_pair_distribution.v1",
                            "adapter": "modernbert",
                        },
                    },
                }
            },
            recipes=[
                {
                    "name": "amd",
                    "routing": {
                        "model_bindings": {"domain_classifier": dict(graph_binding)}
                    },
                }
            ],
        )
    )

    catalog = _catalog(migrated)
    assert "nli" not in catalog["deployments"]
    assert catalog["bindings"] == {
        "domain_classifier": {
            "deployment": "domain",
            "contract": "label_distribution.v1",
            "adapter": "modernbert",
        }
    }
    recipe_binding = migrated["recipes"][0]["routing"]["model_bindings"][
        "domain_classifier"
    ]
    assert "head" not in recipe_binding
    paths = _note_paths(notes)
    assert "global.model_catalog.bindings.hallucination_explainer" in paths
    assert "recipes[0].routing.model_bindings.domain_classifier.head" in paths


def test_system_models_and_module_models_move_to_vela_with_their_label_maps():
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "system": {
                        "domain_classifier": "models/mom-domain-classifier",
                        "pii_classifier": "models/mom-mmbert-pii-detector",
                        "prompt_guard": "models/mmbert32k-jailbreak-detector-merged",
                        "fact_check_classifier": "models/mom-halugate-sentinel",
                        "hallucination_detector": "models/mom-halugate-detector",
                        "hallucination_explainer": "models/mom-halugate-explainer",
                        "feedback_detector": "models/mom-feedback-detector",
                        "safety": "models/Vela-1.0-Encoder-307M-Safety",
                    },
                    "modules": {
                        "classifier": {
                            "domain": {
                                "model_ref": "domain_classifier",
                                "model_id": "models/mom-domain-classifier",
                                "use_modernbert": True,
                                "category_mapping_path": "models/mom-domain-classifier/category_mapping.json",
                            },
                            "pii": {
                                "model_id": "models/mom-pii-classifier",
                                "use_mmbert_32k": True,
                                "pii_mapping_path": "models/mom-pii-classifier/pii_type_mapping.json",
                            },
                        },
                        "prompt_guard": {
                            "enabled": True,
                            "model_id": "models/mom-jailbreak-classifier",
                            "variant": "candle",
                            "model_type": "candle",
                            "jailbreak_mapping_path": "models/mom-jailbreak-classifier/jailbreak_type_mapping.json",
                        },
                        "feedback_detector": {
                            "enabled": True,
                            "model_id": "models/mmbert32k-feedback-detector-merged",
                            "use_modernbert": False,
                            "use_mmbert_32k": True,
                            "feedback_mapping_path": "models/mmbert32k-feedback-detector-merged/label_mapping.json",
                        },
                        "modality_detector": {
                            "classifier": {
                                "model_path": "models/mmbert32k-modality-router-merged"
                            }
                        },
                    },
                }
            }
        )
    )

    catalog = _catalog(migrated)
    assert catalog["system"] == {
        "domain_classifier": "models/Vela-1.0-Encoder-307M-Domain",
        "pii_classifier": "models/Vela-1.0-Encoder-307M-PII",
        "prompt_guard": "models/Vela-1.0-Encoder-307M-Guard",
        "fact_check_classifier": "models/Vela-1.0-Encoder-307M-FactCheck",
        "hallucination_detector": "models/Vela-1.0-Encoder-307M-Halu",
        "feedback_detector": "models/Vela-1.0-Encoder-307M-Feedback",
        "safety": "models/Vela-1.0-Encoder-307M-Safety",
    }
    modules = catalog["modules"]
    assert modules["classifier"]["domain"] == {
        "model_ref": "domain_classifier",
        "model_id": "models/Vela-1.0-Encoder-307M-Domain",
        "category_mapping_path": "models/Vela-1.0-Encoder-307M-Domain/category_mapping.json",
    }
    assert modules["classifier"]["pii"] == {
        "model_id": "models/Vela-1.0-Encoder-307M-PII",
        "pii_mapping_path": "models/Vela-1.0-Encoder-307M-PII/pii_mapping.json",
    }
    assert modules["prompt_guard"] == {
        "enabled": True,
        "model_id": "models/Vela-1.0-Encoder-307M-Guard",
        "jailbreak_mapping_path": "models/Vela-1.0-Encoder-307M-Guard/jailbreak_type_mapping.json",
    }
    assert modules["feedback_detector"] == {
        "enabled": True,
        "model_id": "models/Vela-1.0-Encoder-307M-Feedback",
        "feedback_mapping_path": "",
    }
    assert modules["modality_detector"]["classifier"]["model_path"] == (
        "models/Vela-1.0-Encoder-307M-Modality"
    )
    assert any("NO_FEEDBACK" in note.message for note in notes)
    assert "global.model_catalog.system.hallucination_explainer" in _note_paths(notes)


def test_local_hallucination_detector_loses_nli_and_the_explainer_retires():
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "modules": {
                        "hallucination_mitigation": {
                            "enabled": True,
                            "fact_check": {
                                "model_id": "models/mmbert32k-factcheck-classifier-merged",
                                "use_mmbert_32k": True,
                            },
                            "detector": {
                                "backend": "candle",
                                "model_id": "models/lettucedect-v2-mmbert-base",
                                "include_explanation": True,
                                "enable_nli_filtering": True,
                                "nli_entailment_threshold": 0.75,
                                "threshold": 0.5,
                            },
                            "explainer": {
                                "model_id": "models/mom-halugate-explainer",
                                "threshold": 0.9,
                            },
                        }
                    }
                }
            }
        )
    )

    hallucination = _catalog(migrated)["modules"]["hallucination_mitigation"]
    assert hallucination == {
        "enabled": True,
        "fact_check": {"model_id": "models/Vela-1.0-Encoder-307M-FactCheck"},
        "detector": {"model_id": "models/Vela-1.0-Encoder-307M-Halu", "threshold": 0.5},
    }
    paths = _note_paths(notes)
    for field in (
        "explainer",
        "detector.enable_nli_filtering",
        "detector.include_explanation",
    ):
        assert f"global.model_catalog.modules.hallucination_mitigation.{field}" in paths


def test_endpoint_hallucination_detector_is_left_to_its_service():
    detector = {
        "backend": "endpoint",
        "endpoint": "http://127.0.0.1:8077/v1",
        "include_explanation": True,
        "model_id": "KRLabsOrg/lettucedect-v2-qwen-2b",
    }
    migrated, _ = _migrate(
        _config(
            {
                "model_catalog": {
                    "modules": {
                        "hallucination_mitigation": {"detector": dict(detector)}
                    }
                }
            }
        )
    )

    assert (
        _catalog(migrated)["modules"]["hallucination_mitigation"]["detector"]
        == detector
    )


def test_embedding_backends_and_retired_embedders_move_to_the_runtime():
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "embeddings": {
                        "semantic": {
                            "qwen3_model_path": "models/mom-embedding-pro",
                            "gemma_model_path": "models/mom-embedding-flash",
                            "mmbert_model_path": "models/mmbert-embed-32k-2d-matryoshka",
                            "multimodal_model_path": "models/mom-embedding-multimodal",
                            "bert_model_path": "models/mom-embedding-light",
                            "use_cpu": True,
                            "embedding_config": {
                                "backend": "candle",
                                "model_type": "gemma",
                                "target_layer": 22,
                            },
                        }
                    }
                },
                "stores": {
                    "response_cache": {
                        "enabled": True,
                        "embedding_model": "bert",
                        "polarity_guard": {
                            "mode": "lexical+nli",
                            "nli": {"contradiction_threshold": 0.5},
                        },
                    },
                    "memory": {"embedding_model": "qwen3"},
                },
            }
        )
    )

    semantic = _catalog(migrated)["embeddings"]["semantic"]
    assert semantic == {
        "qwen3_model_path": "models/mom-embedding-pro",
        "mmbert_model_path": "models/Vela-1.0-Encoder-307M-Embedding",
        "multimodal_model_path": "models/vela-1.0-omni-nano",
        "use_cpu": True,
        "embedding_config": {"model_type": "mmbert", "target_layer": 22},
    }
    stores = migrated["global"]["stores"]
    assert stores["response_cache"]["embedding_model"] == "mmbert"
    assert stores["response_cache"]["polarity_guard"] == {"mode": "lexical"}
    assert stores["memory"]["embedding_model"] == "qwen3"
    assert sum("re-embedded" in note.message for note in notes) >= 3


def test_signal_and_plugin_nli_options_and_mlp_devices_are_removed():
    decision = {
        "name": "grounded",
        "priority": 10,
        "rules": {"operator": "AND", "conditions": []},
        "modelRefs": [{"model": "general"}],
        "plugins": [
            {
                "type": "hallucination",
                "configuration": {
                    "enabled": True,
                    "use_nli": True,
                    "hallucination_action": "header",
                },
            }
        ],
        "algorithm": {
            "type": "mlp",
            "mlp": {"device": "cpu", "pretrained_path": "/models/mlp.json"},
            "fusion": {
                "grounding": {"enabled": True, "nli_contradiction_penalty": 0.5}
            },
        },
    }
    migrated, notes = _migrate(
        _config(
            routing={
                "signals": {
                    "classifiers": [
                        {
                            "name": "intent",
                            "type": "local",
                            "model_path": "models/mmbert32k-intent-classifier-merged",
                        }
                    ],
                    "hallucination": [{"name": "ungrounded", "use_nli": True}],
                },
                "decisions": [decision],
            }
        )
    )

    routing = migrated["routing"]
    assert routing["signals"]["classifiers"][0]["model_path"] == (
        "models/Vela-1.0-Encoder-307M-Domain"
    )
    assert routing["signals"]["hallucination"] == [{"name": "ungrounded"}]
    migrated_decision = routing["decisions"][0]
    assert migrated_decision["plugins"][0]["configuration"] == {
        "enabled": True,
        "hallucination_action": "header",
    }
    assert migrated_decision["algorithm"]["mlp"] == {
        "pretrained_path": "/models/mlp.json"
    }
    assert migrated_decision["algorithm"]["fusion"]["grounding"] == {
        "enabled": True,
        "contradiction_penalty": 0.5,
    }
    assert "routing.decisions[0].algorithm.mlp.device" in _note_paths(notes)


def test_runtime_migration_is_idempotent_and_quiet_the_second_time():
    source = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text())
    first, _ = _migrate(source)
    second, notes = _migrate(first)

    assert second == first
    assert len(notes) == 0


@pytest.mark.parametrize(
    "path", ["config/config.yaml", "config/recipes/vela-amd/config.yaml"]
)
def test_maintained_configs_leave_no_removed_provider_or_field(path):
    migrated, notes = _migrate(yaml.safe_load((REPO_ROOT / path).read_text()))

    leftovers = list(_removed_values(migrated))
    assert leftovers == []
    assert not any(note.action_required for note in notes)


def _removed_values(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in REMOVED_FIELDS:
                yield child_path
            if key == "provider" and child in REMOVED_PROVIDERS:
                yield child_path
            if key == "backend" and child in REMOVED_PROVIDERS:
                yield child_path
            if key in {
                "explainer",
                "hallucination_explainer",
                "use_nli",
                "nli",
                "nli_contradiction_penalty",
            }:
                yield child_path
            yield from _removed_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _removed_values(child, f"{path}[{index}]")


def test_migrate_command_lists_changes_and_actions(tmp_path):
    source = tmp_path / "legacy.yaml"
    source.write_text(
        yaml.safe_dump(
            _config(
                {
                    "model_catalog": {
                        "deployments": {
                            "risk": {
                                "provider": "candle",
                                "artifact": "models/Vela-1.0-Encoder-307M-Hazard",
                            },
                            "custom": {
                                "provider": "ort",
                                "artifact": "models/my-model",
                            },
                        }
                    }
                }
            )
        )
    )
    output = tmp_path / "migrated.yaml"

    result = CliRunner().invoke(
        main, ["config", "migrate", "--config", str(source), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert "Changes to review" in result.output
    assert "global.model_catalog.deployments.risk" in result.output
    assert "global.model_catalog.deployments.custom.artifact" in result.output
    deployments = yaml.safe_load(output.read_text())["global"]["model_catalog"][
        "deployments"
    ]
    assert deployments["risk"]["provider"] == "model_runtime"


def test_omni_deployments_name_the_prepared_bundle_router_images_ship():
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "deployments": {
                        "nano": {
                            "provider": "ort",
                            "artifact": "models/vela-1.0-omni-nano",
                            "device": "cpu",
                        },
                        "legacy": {
                            "provider": "candle",
                            "artifact": "models/mom-embedding-multimodal",
                        },
                    }
                }
            }
        )
    )

    deployments = _catalog(migrated)["deployments"]
    assert (
        deployments["nano"]["artifact"]
        == "/opt/router-model-artifacts/vela-1.0-omni-nano"
    )
    assert deployments["legacy"]["artifact"] == (
        "/opt/router-model-artifacts/vela-1.0-omni-nano"
    )
    assert any("prepare.py" in note.message for note in notes)
    assert any("re-embedded" in note.message for note in notes)
