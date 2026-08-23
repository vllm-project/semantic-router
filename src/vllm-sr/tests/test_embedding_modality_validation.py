"""Tests for embedding query_modality compatibility validation.

Mirrors the Go-side `validateEmbeddingContracts` so the CLI catches the same
misconfiguration the router would reject at config-load.
"""

import os
import tempfile

import yaml
from cli.parser import parse_user_config
from cli.validator import validate_embedding_modality_compatibility


def _parse_config_from_yaml(config_yaml: str):
    """Parse a v0.4 canonical config YAML string into a UserConfig."""
    data = yaml.safe_load(config_yaml)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(data, f, sort_keys=False)
        temp_path = f.name
    try:
        return parse_user_config(temp_path)
    finally:
        os.unlink(temp_path)


def _base_config_with_embeddings(
    *,
    query_modality: str,
    model_type: str = "qwen3",
) -> str:
    """Return a minimal v0.4 canonical config containing one embedding rule.

    The embedding model_type is configured under the canonical v0.4 path
    (global.model_catalog.embeddings.semantic.embedding_config.model_type).
    """
    document = {
        "version": "v0.4",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "models": [
            {
                "name": "qwen3-8b",
                "card": {
                    "description": "Model used by embedding validation tests.",
                    "capabilities": ["chat"],
                },
                "connections": [
                    {
                        "provider": "openai-compatible",
                        "endpoint": "http://127.0.0.1:8000/v1",
                        "model": "qwen3-8b",
                    }
                ],
            }
        ],
        "recipes": [
            {
                "name": "default",
                "document": {
                    "signals": {
                        "embeddings": [
                            {
                                "name": "example_rule",
                                "threshold": 0.7,
                                "candidates": [
                                    "example anchor one",
                                    "example anchor two",
                                ],
                                "query_modality": query_modality,
                            }
                        ]
                    },
                    "decisions": [
                        {
                            "name": "default-route",
                            "description": "fallback",
                            "priority": 100,
                            "rules": {"operator": "AND", "conditions": []},
                        }
                    ],
                },
            }
        ],
        "entrypoints": [
            {
                "name": "vllm-sr/default",
                "aliases": ["default"],
                "recipe": "default",
                "assignments": {"default-route": {"models": [{"model": "qwen3-8b"}]}},
            }
        ],
        "global": {
            "services": {
                "backend_egress": {
                    "policy_file": "/app/config/backend-egress-policy.yaml"
                }
            },
            "model_catalog": {
                "embeddings": {
                    "semantic": {"embedding_config": {"model_type": model_type}}
                }
            },
        },
    }
    return yaml.safe_dump(document, sort_keys=False)


def test_text_modality_passes_with_any_model_type():
    for model_type in ("qwen3", "gemma", "mmbert", "multimodal"):
        config = _parse_config_from_yaml(
            _base_config_with_embeddings(query_modality="text", model_type=model_type)
        )
        errors = validate_embedding_modality_compatibility(config)
        assert (
            errors == []
        ), f"text modality should pass under model_type={model_type}, got: {errors}"


def test_image_modality_requires_multimodal_model_type():
    config = _parse_config_from_yaml(
        _base_config_with_embeddings(query_modality="image", model_type="qwen3")
    )
    errors = validate_embedding_modality_compatibility(config)
    assert len(errors) == 1, f"expected one error, got {errors}"
    msg = str(errors[0])
    assert "example_rule" in msg
    assert "multimodal" in msg
    assert "qwen3" in msg


def test_image_modality_passes_with_multimodal_model_type():
    config = _parse_config_from_yaml(
        _base_config_with_embeddings(query_modality="image", model_type="multimodal")
    )
    errors = validate_embedding_modality_compatibility(config)
    assert errors == [], f"image modality should pass under multimodal, got: {errors}"


def test_audio_modality_rejected_with_planned_message():
    config = _parse_config_from_yaml(
        _base_config_with_embeddings(query_modality="audio", model_type="multimodal")
    )
    errors = validate_embedding_modality_compatibility(config)
    assert len(errors) == 1
    msg = str(errors[0])
    assert "example_rule" in msg
    assert "MultiModalEncodeAudioFromBase64" in msg
    assert "planned" in msg


def test_no_embeddings_returns_no_errors():
    data = yaml.safe_load(_base_config_with_embeddings(query_modality="text"))
    data["recipes"][0]["document"]["signals"] = {}
    config = _parse_config_from_yaml(yaml.safe_dump(data, sort_keys=False))
    assert validate_embedding_modality_compatibility(config) == []


def test_omitted_query_modality_defaults_to_text():
    """A rule that doesn't set query_modality should be treated as text and pass."""
    data = yaml.safe_load(_base_config_with_embeddings(query_modality="text"))
    rule = data["recipes"][0]["document"]["signals"]["embeddings"][0]
    rule["name"] = "implicit_text"
    rule.pop("query_modality")
    config = _parse_config_from_yaml(yaml.safe_dump(data, sort_keys=False))
    errors = validate_embedding_modality_compatibility(config)
    assert (
        errors == []
    ), f"omitted query_modality should default to text and pass, got: {errors}"


def test_image_modality_when_model_type_path_is_absent():
    """If global.model_catalog.embeddings.semantic.embedding_config is missing,
    treat model_type as empty and reject image-modality rules accordingly."""
    data = yaml.safe_load(_base_config_with_embeddings(query_modality="image"))
    data["recipes"][0]["document"]["signals"]["embeddings"][0][
        "name"
    ] = "image_rule_no_model_type"
    data["global"].pop("model_catalog")
    config = _parse_config_from_yaml(yaml.safe_dump(data, sort_keys=False))
    errors = validate_embedding_modality_compatibility(config)
    assert len(errors) == 1
    assert "image_rule_no_model_type" in str(errors[0])
