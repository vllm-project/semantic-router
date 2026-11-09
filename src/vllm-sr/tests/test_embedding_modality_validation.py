"""Embedding query_modality is the Router's to match against the encoder.

The Router refuses at load a media rule whose embedding deployment cannot
encode that media; the CLI parses the rule and leaves the judgment to it.
"""

import os
import tempfile

import pytest
import yaml
from cli.models import EmbeddingSignal
from cli.parser import ConfigParseError, parse_user_config
from cli.validator import validate_user_config
from pydantic import ValidationError


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
    model: qwen3-8b
  models:
    - name: qwen3-8b
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
          provider: vllm
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


@pytest.mark.parametrize("query_modality", ["text", "image", "audio"])
@pytest.mark.parametrize("model_type", ["qwen3", "mmbert", "multimodal"])
def test_the_cli_leaves_media_capability_to_the_router(query_modality, model_type):
    config = _parse_config_from_yaml(
        _base_config_with_embeddings(
            query_modality=query_modality, model_type=model_type
        )
    )
    assert validate_user_config(config, log_summary=False) == []


def test_image_candidates_are_left_to_the_router():
    raw = yaml.safe_load(_base_config_with_embeddings(query_modality="text"))
    raw["routing"]["signals"]["embeddings"][0]["image_candidates"] = ["./positive.png"]
    config = _parse_config_from_yaml(yaml.safe_dump(raw))
    assert validate_user_config(config, log_summary=False) == []


def test_an_unknown_query_modality_is_refused_structurally():
    with pytest.raises(ConfigParseError, match="query_modality"):
        _parse_config_from_yaml(_base_config_with_embeddings(query_modality="video"))


def test_embedding_candidate_banks_typed_contract():
    rule = EmbeddingSignal(
        name="images",
        threshold=-0.1,
        image_candidates=["./positive.png"],
        negative_candidates=["a different subject"],
        negative_image_candidates=["./negative.png"],
    )
    assert rule.candidates == []
    assert rule.model_dump()["negative_image_candidates"] == ["./negative.png"]
    for values in (
        {"negative_candidates": ["negative"]},
        {"candidates": [""]},
        {"candidates": [17]},
    ):
        with pytest.raises(ValidationError):
            EmbeddingSignal(name="bad", threshold=0.1, **values)
    with pytest.raises(ValidationError):
        EmbeddingSignal(name="bad", threshold=1.1, candidates=["positive"])
