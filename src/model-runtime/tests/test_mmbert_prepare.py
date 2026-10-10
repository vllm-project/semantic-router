"""Issue #2440: an mmBERT-shaped classifier is verified by task_heads, not the retired Candle loader.

The old unified initializer could park every thread on darwin/arm64. Those
bindings are gone. Vela 1.0 task models use this same config shape, so
verification has to return a package error when a file is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.families.task_heads.family import TaskHeadsFamily
from vllm_srun.plugins.base import PackageRef

# vocab >= 200000 and sans_pos is the mmBERT checkpoint signature.
CONFIG = {
    "model_type": "modernbert",
    "architectures": ["ModernBertForSequenceClassification"],
    "vocab_size": 256000,
    "position_embedding_type": "sans_pos",
    "max_position_embeddings": 32768,
    "problem_type": "single_label_classification",
    "id2label": {"0": "other", "1": "math"},
}


def test_mmbert_classifier_verify_returns_before_weights(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    with pytest.raises(PackageError, match="package file is missing: model.safetensors"):
        TaskHeadsFamily().verify(PackageRef(root=tmp_path))
    assert not (tmp_path / "model.safetensors").exists()
