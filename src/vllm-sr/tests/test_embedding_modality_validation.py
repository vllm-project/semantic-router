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
    """Parse a v0.3 canonical config YAML string into a UserConfig."""
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
    """Return a minimal v0.3 canonical config containing one embedding rule.

    The embedding model_type is configured under the canonical v0.3 path
    (global.model_catalog.embeddings.semantic.embedding_config.model_type).
    """
    return f"""
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen3-8b
  models:
    - name: qwen3-8b
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
routing:
  signals:
    embeddings:
      - name: example_rule
        threshold: 0.7
        candidates:
          - example anchor one
          - example anchor two
        query_modality: {query_modality}
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          model_type: {model_type}
"""


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
    config_yaml = """
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen3-8b
  models:
    - name: qwen3-8b
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
routing:
  signals: {}
"""
    config = _parse_config_from_yaml(config_yaml)
    assert validate_embedding_modality_compatibility(config) == []


def test_omitted_query_modality_defaults_to_text():
    """A rule that doesn't set query_modality should be treated as text and pass."""
    config_yaml = """
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen3-8b
  models:
    - name: qwen3-8b
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
routing:
  signals:
    embeddings:
      - name: implicit_text
        threshold: 0.7
        candidates: ["example one", "example two"]
"""
    config = _parse_config_from_yaml(config_yaml)
    errors = validate_embedding_modality_compatibility(config)
    assert (
        errors == []
    ), f"omitted query_modality should default to text and pass, got: {errors}"


def test_image_modality_when_model_type_path_is_absent():
    """If global.model_catalog.embeddings.semantic.embedding_config is missing,
    treat model_type as empty and reject image-modality rules accordingly."""
    config_yaml = """
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen3-8b
  models:
    - name: qwen3-8b
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
routing:
  signals:
    embeddings:
      - name: image_rule_no_model_type
        threshold: 0.55
        candidates: ["example image anchor"]
        query_modality: image
"""
    config = _parse_config_from_yaml(config_yaml)
    errors = validate_embedding_modality_compatibility(config)
    assert len(errors) == 1
    assert "image_rule_no_model_type" in str(errors[0])
