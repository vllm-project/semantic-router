"""Recipe-local pair scoring survives the canonical CLI transport.

The Router validates the deployment that scores the pairs.
"""

import copy

import pytest
from cli.config_schema.validation import validate_config_structure
from cli.models import UserConfig
from cli.validator import validate_plugin_configurations


def document():
    return {
        "version": "v0.3",
        "global": {
            "model_catalog": {
                "deployments": {
                    "rank": {
                        "provider": "model_runtime",
                        "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Reranker",
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
    assert validate_plugin_configurations(parsed) == []
    assert parsed.routing.model_bindings["rag.reranker"].pair_scorer.dimension == 768
    assert raw == original


def test_rerank_names_the_deployment_that_scores_the_pairs():
    raw = document()
    raw["routing"]["model_bindings"] = {}
    plugin = raw["routing"]["decisions"][0]["plugins"][0]["configuration"]
    plugin["rerank"] = {"deployment": "rank", "top_k": 3, "layer": 22, "dimension": 768}
    assert validate_plugin_configurations(UserConfig.model_validate(raw)) == []


@pytest.mark.parametrize("case", ["other_backend", "too_many_results"])
def test_incompatible_pair_scoring_is_rejected(case):
    raw = document()
    plugin = raw["routing"]["decisions"][0]["plugins"][0]["configuration"]
    if case == "other_backend":
        plugin["backend"] = "external_api"
    else:
        plugin["rerank"]["top_k"] = 11
    assert validate_plugin_configurations(UserConfig.model_validate(raw))
