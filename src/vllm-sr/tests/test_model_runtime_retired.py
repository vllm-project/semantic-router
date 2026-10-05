"""The CLI refuses the model execution settings the router refuses, and
``vllm-sr config migrate`` leaves none of them behind."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_migration import migrate_config_data  # noqa: E402
from cli.config_migration_notes import MigrationNotes  # noqa: E402
from cli.model_runtime_retired import (  # noqa: E402
    REMOVED_DEPLOYMENT_FIELDS,
    REMOVED_EMBEDDING_BACKENDS,
    REMOVED_MODULE_FIELDS,
    REMOVED_PROVIDERS,
    RETIRED_EMBEDDING_PATHS,
    RETIRED_EMBEDDING_TYPES,
    retired_model_fields,
)
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402

ROUTER_PARSER = REPO_ROOT / "src/semantic-router/pkg/config/loader_model_execution.go"


def _legacy() -> dict:
    """A configuration that sets every retired model execution setting."""
    hallucination_rule = {"name": "grounded", "use_nli": True}
    hallucination_plugin = {
        "type": "hallucination",
        "configuration": {"enabled": True, "use_nli": True},
    }
    routing = {
        "modelCards": [{"name": "general"}],
        "signals": {"hallucination": [hallucination_rule]},
        "decisions": [
            {
                "name": "default-route",
                "priority": 1,
                "rules": {"operator": "AND", "conditions": []},
                "modelRefs": [{"model": "general"}],
                "plugins": [hallucination_plugin],
                "algorithm": {
                    "type": "fusion",
                    "fusion": {"grounding": {"nli_contradiction_penalty": 0.5}},
                },
            }
        ],
    }
    return {
        "version": "v0.3",
        "listeners": [],
        "providers": {"defaults": {"model": "general"}},
        "routing": routing,
        "recipes": [
            {"name": "isolated", "routing": yaml.safe_load(yaml.safe_dump(routing))}
        ],
        "global": {
            "model_catalog": {
                "system": {"hallucination_explainer": "models/mom-halugate-explainer"},
                "embeddings": {
                    "semantic": {
                        "gemma_model_path": "models/embeddinggemma-300m",
                        "bert_model_path": "models/all-MiniLM-L12-v2",
                        "embedding_config": {
                            "backend": "OpenVINO",
                            "model_type": "gemma",
                        },
                    }
                },
                "deployments": {
                    "candle-domain": {
                        "provider": "candle",
                        "artifact": "models/Vela-1.0-Encoder-307M-Domain",
                        "precision": "fp16",
                    },
                    "ort-embedding": {
                        "provider": "ORT",
                        "artifact": "models/Vela-1.0-Encoder-307M-Embedding",
                        "device": "migraphx:0",
                        "custom_ops_profile": "ck_flash_attention",
                        "compilation_cache_dir": "/var/cache/migraphx",
                    },
                    "openvino-guard": {
                        "provider": "openvino",
                        "artifact": "models/Vela-1.0-Encoder-307M-Guard",
                    },
                },
                "modules": {
                    "prompt_guard": {
                        "enabled": True,
                        "variant": "candle",
                        "model_type": "modernbert",
                        "use_modernbert": True,
                        "use_mmbert_32k": False,
                    },
                    "classifier": {
                        "domain": {
                            "variant": "",
                            "use_modernbert": True,
                            "use_mmbert_32k": False,
                        },
                        "pii": {"use_modernbert": False, "use_mmbert_32k": True},
                    },
                    "feedback_detector": {
                        "use_modernbert": False,
                        "use_mmbert_32k": True,
                    },
                    "hallucination_mitigation": {
                        "enabled": True,
                        "nli_model": {"model_id": "models/mom-halugate-explainer"},
                        "explainer": {"model_id": "models/mom-halugate-explainer"},
                        "fact_check": {"use_modernbert": False, "use_mmbert_32k": True},
                        "detector": {
                            "backend": "candle",
                            "enable_nli_filtering": True,
                            "nli_entailment_threshold": 0.5,
                        },
                    },
                },
            },
            "stores": {
                "response_cache": {
                    "enabled": True,
                    "embedding_model": "bert",
                    "polarity_guard": {"mode": "lexical"},
                }
            },
            "router": {"model_selection": {"ml": {"model_type": "bert"}}},
        },
    }


def test_every_retired_setting_is_found_by_its_path():
    found = retired_model_fields(_legacy())

    deployments = "global.model_catalog.deployments"
    modules = "global.model_catalog.modules"
    expected = [
        f"{deployments}.candle-domain.provider: candle",
        f"{deployments}.candle-domain.precision",
        f"{deployments}.openvino-guard.provider: openvino",
        f"{deployments}.ort-embedding.provider: ORT",
        f"{deployments}.ort-embedding.custom_ops_profile",
        f"{deployments}.ort-embedding.compilation_cache_dir",
        *(
            f"{modules}.{'.'.join(path)}.{field}"
            for path, fields in REMOVED_MODULE_FIELDS
            for field in fields
        ),
        f"{modules}.hallucination_mitigation.detector.backend: candle",
        "global.model_catalog.embeddings.semantic.embedding_config.backend: OpenVINO",
        "global.model_catalog.embeddings.semantic.gemma_model_path",
        "global.model_catalog.embeddings.semantic.bert_model_path",
        "global.model_catalog.embeddings.semantic.embedding_config.model_type: gemma",
        "global.stores.response_cache.embedding_model: bert",
        "global.router.model_selection.ml.model_type: bert",
        "global.model_catalog.system.hallucination_explainer",
        "global.stores.response_cache.polarity_guard",
    ]
    for prefix in ("routing", "recipes[0].routing"):
        expected += [
            f"{prefix}.signals.hallucination[0].use_nli",
            f"{prefix}.decisions[0].algorithm.fusion.grounding.nli_contradiction_penalty",
            f"{prefix}.decisions[0].plugins[0].configuration.use_nli",
        ]
    assert found == expected


def test_parsing_refuses_retired_settings_and_names_the_migration(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(_legacy(), sort_keys=False))

    with pytest.raises(ConfigParseError) as refused:
        parse_user_config(str(path), log_summary=False)

    message = str(refused.value)
    assert "global.model_catalog.deployments.candle-domain.provider: candle" in message
    assert "global.stores.response_cache.polarity_guard" in message
    assert f"vllm-sr config migrate --config {path}" in message


def test_migration_leaves_no_retired_setting():
    migrated = migrate_config_data(_legacy(), MigrationNotes())

    assert retired_model_fields(migrated) == []


def test_empty_retired_embedding_paths_are_refused_until_migrated(tmp_path):
    semantic = {"mmbert_model_path": "models/Vela-1.0-Encoder-307M-Embedding"}
    document = {
        "version": "v0.3",
        "listeners": [],
        "providers": {"defaults": {"model": "general"}},
        "global": {
            "model_catalog": {
                "embeddings": {
                    "semantic": {
                        **semantic,
                        **dict.fromkeys(RETIRED_EMBEDDING_PATHS, ""),
                    }
                }
            }
        },
    }
    path = tmp_path / "empty-paths.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    with pytest.raises(ConfigParseError) as refused:
        parse_user_config(str(path), log_summary=False)

    message = str(refused.value)
    for field in RETIRED_EMBEDDING_PATHS:
        assert f"global.model_catalog.embeddings.semantic.{field}" in message
    assert f"vllm-sr config migrate --config {path}" in message

    migrated = migrate_config_data(document, MigrationNotes())
    assert migrated["global"]["model_catalog"]["embeddings"]["semantic"] == semantic
    path.write_text(yaml.safe_dump(migrated, sort_keys=False))
    parse_user_config(str(path), log_summary=False)


def test_the_inventory_is_the_router_parsers():
    """The router's parser and the CLI refuse the same settings."""
    source = ROUTER_PARSER.read_text(encoding="utf-8")
    providers = re.search(
        r"removedModelProviders\s*=\s*map\[string\]bool\{(.*?)\}", source
    )
    embedding = re.search(
        r"removedEmbeddingBackends\s*=\s*map\[string\]bool\{(.*?)\}", source
    )
    fields = re.search(r"removedDeploymentFields\s*=\s*\[\]string\{(.*?)\}", source)
    retired_types = re.search(
        r"retiredEmbeddingTypes\s*=\s*map\[string\]bool\{(.*?)\}", source
    )
    retired_paths = re.search(
        r"removedEmbeddingPaths\s*=\s*\[\]string\{(.*?)\}", source
    )
    modules = re.findall(r"\{\[\]string\{([^}]*)\}, \[\]string\{([^}]*)\}\}", source)

    def strings(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r'"([^"]+)"', text))

    assert providers and set(strings(providers.group(1))) == REMOVED_PROVIDERS
    assert embedding and set(strings(embedding.group(1))) == REMOVED_EMBEDDING_BACKENDS
    assert fields and strings(fields.group(1)) == REMOVED_DEPLOYMENT_FIELDS
    assert (
        retired_types
        and set(strings(retired_types.group(1))) == RETIRED_EMBEDDING_TYPES
    )
    assert retired_paths and strings(retired_paths.group(1)) == RETIRED_EMBEDDING_PATHS
    assert tuple((strings(path), strings(names)) for path, names in modules) == (
        REMOVED_MODULE_FIELDS
    )


def test_the_hallucination_endpoint_shorthand_is_retired():
    document = _legacy()
    modules = document["global"]["model_catalog"]["modules"]
    modules["hallucination_mitigation"]["detector"] = {
        "backend": "endpoint",
        "endpoint": "http://detector:8000/v1",
        "model_id": "lettucedect",
    }

    found = retired_model_fields(document)

    detector = "global.model_catalog.modules.hallucination_mitigation.detector"
    assert f"{detector}.backend: endpoint" in found
    assert f"{detector}.endpoint" in found
