"""Recipe-local pair scoring survives the canonical CLI transport."""

import copy

import pytest
from cli.config_schema.validation import validate_config_structure
from cli.models import UserConfig
from cli.validator import validate_plugin_configurations
from cli.validator_model_runtime import validate_model_runtime_references


def document():
    return {
        "version": "v0.3",
        "global": {
            "model_catalog": {
                "deployments": {
                    "rank": {
                        "artifact": "models/reranker",
                        "provider": "candle",
                        "device": "cpu",
                        "input": {"max_tokens": 4096, "overflow": "reject"},
                    }
                }
            }
        },
        "routing": {
            "model_bindings": {
                "rag.reranker": {
                    "deployment": "rank",
                    "contract": "relevance_scores.v1",
                    "adapter": "vela_reranker",
                    "pair_scorer": {"layer": 22, "dimension": 768},
                }
            },
            "decisions": [
                {
                    "name": "retrieve",
                    "priority": 1,
                    "modelRefs": [{"model": "test-backend"}],
                    "rules": {"operator": "AND", "conditions": []},
                    "plugins": [
                        {
                            "type": "rag",
                            "configuration": {
                                "enabled": True,
                                "backend": "vectorstore",
                                "top_k": 10,
                                "rerank": {"top_k": 3},
                                "backend_config": {"vector_store_id": "vs-test"},
                            },
                        }
                    ],
                }
            ],
        },
    }


def test_pair_selection_survives_public_schema_and_cli():
    raw = document()
    original = copy.deepcopy(raw)
    assert validate_config_structure(raw) == []
    parsed = UserConfig.model_validate(raw)
    assert validate_model_runtime_references(parsed) == []
    assert validate_plugin_configurations(parsed) == []
    assert parsed.routing.model_bindings["rag.reranker"].pair_scorer.dimension == 768
    assert raw == original


@pytest.mark.parametrize(
    "case",
    [
        "missing_binding",
        "probability",
        "truncation",
        "remote",
        "foreign_task",
        "other_backend",
        "too_many_results",
    ],
)
def test_incompatible_pair_scoring_is_rejected(case):
    raw = document()
    binding = raw["routing"]["model_bindings"]["rag.reranker"]
    deployment = raw["global"]["model_catalog"]["deployments"]["rank"]
    plugin = raw["routing"]["decisions"][0]["plugins"][0]["configuration"]
    if case == "missing_binding":
        raw["routing"]["model_bindings"] = {}
    elif case == "probability":
        binding["contract"] = "label_distribution.v1"
    elif case == "truncation":
        deployment["input"]["overflow"] = "truncate"
    elif case == "remote":
        deployment["provider"] = "http"
    elif case == "foreign_task":
        binding["contract"] = "embedding.v1"
        raw["routing"]["model_bindings"]["embedding"] = raw["routing"][
            "model_bindings"
        ].pop("rag.reranker")
    elif case == "other_backend":
        plugin["backend"] = "external_api"
    else:
        plugin["rerank"]["top_k"] = 11
    parsed = UserConfig.model_validate(raw)
    assert validate_model_runtime_references(parsed) + validate_plugin_configurations(
        parsed
    )
