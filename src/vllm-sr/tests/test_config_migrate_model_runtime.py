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
from cli.model_runtime_retired import retired_model_fields  # noqa: E402

REMOVED_FIELDS = {"precision", "custom_ops_profile", "compilation_cache_dir"}
REMOVED_PROVIDERS = {"candle", "ort", "openvino"}


def _config(global_config=None, routing=None, recipes=None, providers=None):
    config = {
        "version": "v0.3",
        "listeners": [],
        "providers": {"defaults": {"model": "general"}, **(providers or {})},
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


def test_system_models_and_module_models_move_to_vela_without_legacy_label_maps():
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
    # The legacy label maps described the legacy models; the router reads the
    # replacements' labels from the models themselves.
    assert modules["classifier"]["domain"] == {
        "model_ref": "domain_classifier",
        "model_id": "models/Vela-1.0-Encoder-307M-Domain",
    }
    assert modules["classifier"]["pii"] == {
        "model_id": "models/Vela-1.0-Encoder-307M-PII",
    }
    assert modules["prompt_guard"] == {
        "enabled": True,
        "model_id": "models/Vela-1.0-Encoder-307M-Guard",
    }
    assert modules["feedback_detector"] == {
        "enabled": True,
        "model_id": "models/Vela-1.0-Encoder-307M-Feedback",
    }
    assert modules["modality_detector"]["classifier"]["model_path"] == (
        "models/Vela-1.0-Encoder-307M-Modality"
    )
    assert any("NO_FEEDBACK" in note.message for note in notes)
    assert "global.model_catalog.system.hallucination_explainer" in _note_paths(notes)
    assert (
        "global.model_catalog.modules.classifier.domain.category_mapping_path"
        in _note_paths(notes)
    )


def test_an_operators_own_label_map_stays_with_the_replacement_model():
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "modules": {
                        "classifier": {
                            "domain": {
                                "model_id": "models/mom-domain-classifier",
                                "category_mapping_path": "config/domains.json",
                            }
                        }
                    }
                }
            }
        )
    )

    assert _catalog(migrated)["modules"]["classifier"]["domain"] == {
        "model_id": "models/Vela-1.0-Encoder-307M-Domain",
        "category_mapping_path": "config/domains.json",
    }
    assert (
        "global.model_catalog.modules.classifier.domain.category_mapping_path"
        not in _note_paths(notes)
    )


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


def _endpoint_config(endpoint, bindings=None):
    catalog = {
        "modules": {
            "hallucination_mitigation": {
                "detector": {
                    "backend": "endpoint",
                    "endpoint": endpoint,
                    "include_explanation": True,
                    "model_id": "KRLabsOrg/lettucedect-v2-qwen-2b",
                    "enable_nli_filtering": True,
                }
            }
        }
    }
    if bindings is not None:
        catalog["bindings"] = bindings
    return _config({"model_catalog": catalog})


def test_endpoint_hallucination_detector_becomes_an_http_chat_binding():
    migrated, notes = _migrate(_endpoint_config("http://detector.example:8077/v1/"))

    catalog = _catalog(migrated)
    assert catalog["modules"]["hallucination_mitigation"]["detector"] == {
        "include_explanation": True
    }
    assert catalog["external"] == [
        {
            "name": "hallucination-detector",
            "model_role": "classification",
            "llm_endpoint": {
                "address": "detector.example",
                "port": 8077,
                "protocol": "http",
            },
            "llm_model_name": "KRLabsOrg/lettucedect-v2-qwen-2b",
            "llm_timeout_seconds": 10,
        }
    ]
    assert catalog["deployments"]["hallucination-detector"] == {
        "provider": "http",
        "external_model": "hallucination-detector",
    }
    assert catalog["bindings"]["hallucination_detector"] == {
        "deployment": "hallucination-detector",
        "contract": "token_spans.v1",
        "adapter": "http_chat",
    }
    detector = "global.model_catalog.modules.hallucination_mitigation.detector"
    assert {detector + ".backend", detector + ".enable_nli_filtering"} <= _note_paths(
        notes
    )
    assert retired_model_fields(migrated) == []


def test_endpoint_on_another_path_needs_a_hand_written_binding():
    with pytest.raises(ValueError, match="/v1/chat/completions"):
        _migrate(_endpoint_config("http://detector.example:8077/detect"))


def test_an_existing_detector_binding_wins_over_the_endpoint_shorthand():
    binding = {
        "deployment": "grounding",
        "contract": "token_spans.v1",
        "adapter": "http_classify",
    }
    migrated, _ = _migrate(
        _endpoint_config(
            "http://detector.example:8077/v1", {"hallucination_detector": binding}
        )
    )

    catalog = _catalog(migrated)
    assert catalog["bindings"] == {"hallucination_detector": binding}
    assert "external" not in catalog
    assert catalog["modules"]["hallucination_mitigation"]["detector"] == {
        "include_explanation": True
    }


def test_every_field_the_router_rejects_leaves_the_configuration():
    migrated, _ = _migrate(
        _config(
            {
                "model_catalog": {
                    "deployments": {
                        "served": {
                            "provider": "model_runtime",
                            "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Domain",
                            "precision": "fp16",
                            "custom_ops_profile": "none",
                        }
                    },
                    "modules": {
                        "prompt_guard": {"model_type": "modernbert"},
                        "hallucination_mitigation": {
                            "nli_model": {"model_id": "models/mom-halugate-explainer"}
                        },
                    },
                },
                "stores": {"response_cache": {"polarity_guard": {"mode": "lexical"}}},
            }
        )
    )

    catalog = _catalog(migrated)
    assert catalog["deployments"]["served"] == {
        "provider": "model_runtime",
        "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Domain",
        "profile": "max_speed",
    }
    assert catalog["modules"] == {"prompt_guard": {}, "hallucination_mitigation": {}}
    assert migrated["global"]["stores"]["response_cache"] == {}


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
    assert "polarity_guard" not in stores["response_cache"]
    assert stores["memory"]["embedding_model"] == "qwen3"
    assert sum("re-embedded" in note.message for note in notes) >= 3


def test_stores_selection_and_rag_leave_the_retired_minilm_default():
    rag = {
        "type": "rag",
        "configuration": {
            "enabled": True,
            "backend": "milvus",
            "backend_config": {"collection": "docs"},
        },
    }
    hosted_rag = {
        "type": "rag",
        "configuration": {
            "enabled": True,
            "backend": "openai",
            "backend_config": {"vector_store_id": "vs-1"},
        },
    }
    migrated, notes = _migrate(
        _config(
            {
                "model_catalog": {
                    "embeddings": {
                        "semantic": {"bert_model_path": "models/mom-embedding-light"}
                    }
                },
                "router": {
                    "model_selection": {
                        "ml": {"models_path": "models/selection", "model_type": "bert"}
                    }
                },
                "stores": {
                    "response_cache": {"enabled": True, "backend_type": "memory"},
                    "memory": {
                        "enabled": True,
                        "backend": "milvus",
                        "embedding_model": "gemma",
                    },
                    "vector_store": {"enabled": True, "backend_type": "milvus"},
                    "tool_sessions": {"backend": "redis"},
                },
            },
            routing={
                "decisions": [
                    {"name": "docs", "plugins": [rag]},
                    {"name": "hosted", "plugins": [hosted_rag]},
                ]
            },
        )
    )

    stores = migrated["global"]["stores"]
    assert stores["response_cache"]["embedding_model"] == "mmbert"
    assert stores["memory"]["embedding_model"] == "mmbert"
    assert stores["vector_store"]["embedding_model"] == "mmbert"
    assert "embedding_model" not in stores["tool_sessions"]
    assert (
        migrated["global"]["router"]["model_selection"]["ml"]["model_type"] == "mmbert"
    )
    by_path = {note.path: note for note in notes}
    cache = by_path["global.stores.response_cache.embedding_model"]
    assert not cache.action_required
    assert "bert was this store's default" in cache.message
    for path in (
        "global.stores.memory.embedding_model",
        "global.stores.vector_store.embedding_model",
        "global.router.model_selection.ml.model_type",
        "routing.decisions[0].plugins[0].configuration.backend",
    ):
        assert by_path[path].action_required, path
    assert (
        "'docs'"
        in by_path["routing.decisions[0].plugins[0].configuration.backend"].message
    )
    assert "routing.decisions[1].plugins[0].configuration.backend" not in by_path


def test_store_default_follows_a_served_semantic_model():
    source = _config(
        {
            "model_catalog": {
                "embeddings": {
                    "semantic": {"qwen3_model_path": "models/mom-embedding-pro"}
                }
            },
            "stores": {"response_cache": {"enabled": True, "backend_type": "redis"}},
        }
    )

    migrated, notes = _migrate(source)

    assert "embedding_model" not in migrated["global"]["stores"]["response_cache"]
    assert len(notes) == 0


def test_vela_embedding_stores_get_a_vector_size_it_serves():
    migrated, notes = _migrate(
        _config(
            {
                "stores": {
                    "response_cache": {
                        "enabled": True,
                        "backend_type": "redis",
                        "embedding_model": "bert",
                        "redis": {"index": {"vector_field": {"dimension": 384}}},
                    },
                    "memory": {
                        "enabled": True,
                        "backend": "milvus",
                        "embedding_model": "mmbert",
                        "milvus": {"dimension": 384},
                        "valkey": {"dimension": 512},
                    },
                    "vector_store": {
                        "enabled": True,
                        "backend_type": "milvus",
                        "embedding_dimension": 384,
                    },
                }
            }
        )
    )

    stores = migrated["global"]["stores"]
    assert stores["response_cache"]["redis"]["index"]["vector_field"] == {
        "dimension": 768
    }
    assert stores["memory"]["milvus"] == {"dimension": 256}
    assert stores["memory"]["valkey"] == {"dimension": 512}
    assert stores["vector_store"]["embedding_model"] == "mmbert"
    assert stores["vector_store"]["embedding_dimension"] == 768
    by_path = {note.path: note for note in notes}
    for path in (
        "global.stores.response_cache.redis.index.vector_field.dimension",
        "global.stores.memory.milvus.dimension",
        "global.stores.vector_store.embedding_dimension",
    ):
        assert by_path[path].action_required, path
        assert by_path[path].message.startswith("384 -> "), path
    assert "global.stores.memory.valkey.dimension" not in by_path


def test_other_embedding_models_keep_their_vector_size():
    source = _config(
        {
            "stores": {
                "memory": {
                    "enabled": True,
                    "backend": "qdrant",
                    "embedding_model": "multimodal",
                    "qdrant": {"dimension": 384},
                }
            }
        }
    )

    migrated, notes = _migrate(source)

    assert migrated["global"]["stores"]["memory"]["qdrant"] == {"dimension": 384}
    assert len(notes) == 0


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


def _is_reembed_reminder(note):
    return note.action_required and "re-embed" in note.message


def test_runtime_migration_is_idempotent_and_quiet_the_second_time():
    source = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text())
    first, _ = _migrate(source)
    second, notes = _migrate(first)

    assert second == first
    # No value records that a RAG collection was re-embedded, so its reminder repeats.
    assert [note.path for note in notes if not _is_reembed_reminder(note)] == []
    assert all(note.path.endswith(".configuration.backend") for note in notes)


@pytest.mark.parametrize(
    "path", ["config/config.yaml", "config/recipes/vela-amd/config.yaml"]
)
def test_maintained_configs_leave_no_removed_provider_or_field(path):
    migrated, notes = _migrate(yaml.safe_load((REPO_ROOT / path).read_text()))

    leftovers = list(_removed_values(migrated))
    assert leftovers == []
    assert [
        note.path
        for note in notes
        if note.action_required and not _is_reembed_reminder(note)
    ] == []


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
                "nli_model",
                "polarity_guard",
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
                },
                providers={
                    "models": [{"name": "general", "provider_model_id": "gpt-4o-mini"}]
                },
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


def test_omni_deployments_name_the_published_repositories():
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
                        "bundle": {
                            "provider": "model_runtime",
                            "artifact": "/opt/router-model-artifacts/vela-1.0-omni-mini",
                        },
                    }
                }
            }
        )
    )

    deployments = _catalog(migrated)["deployments"]
    assert deployments["nano"]["artifact"] == "vllm-sr/Vela-1.0-Omni-Nano"
    assert deployments["legacy"]["artifact"] == "vllm-sr/Vela-1.0-Omni-Nano"
    assert deployments["bundle"]["artifact"] == "vllm-sr/Vela-1.0-Omni-Mini"
    assert any("no longer ship" in note.message for note in notes)
    assert any("re-embedded" in note.message for note in notes)
