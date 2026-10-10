"""Issue #2440: mmBERT-shaped classifier initialization has to return.

The retired Candle initializer could park every thread on darwin/arm64, and a
timeout inside that process could not interrupt it. ``task_heads`` is the
loader current main uses for these classifiers, including Vela 1.0. This test
runs that load in a child process. The parent kills the child if it is still
running after 60 seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from vllm_srun.errors import PackageError
from vllm_srun.families.task_heads.family import TaskHeadsFamily
from vllm_srun.heads.task import ClassifierHead
from vllm_srun.plugins.base import PackageRef
from vllm_srun.testing.fixtures import modernbert_config, random_backbone, save
from vllm_srun.testing.task_heads import SEQUENCE, encoder_tokenizer

PACKAGE = "VLLM_SR_MMBERT_PACKAGE"
DEADLINE_SECONDS = 60

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

LOAD = """
import os
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.runtime import Runtime

runtime = Runtime(
    ServeConfig(
        models=(ModelConfig(model=os.environ[%r], device="cpu"),),
        load_attempts=1,
    )
)
runtime.start(background=False)
runtime.stop()
""" % (
    PACKAGE,
)


def test_mmbert_classifier_verify_returns_before_weights(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    with pytest.raises(
        PackageError, match="package file is missing: model.safetensors"
    ):
        TaskHeadsFamily().verify(PackageRef(root=tmp_path))
    assert not (tmp_path / "model.safetensors").exists()


def test_mmbert_classifier_load_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / "classifier"
    write_mmbert_classifier(root)
    env = os.environ.copy()
    env[PACKAGE] = str(root)
    runtime_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (runtime_root, env.get("PYTHONPATH", "")) if item
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", LOAD],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = proc.communicate(timeout=DEADLINE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        pytest.fail(
            f"mmBERT classifier load exceeded {DEADLINE_SECONDS}s; parent killed it\n{output}"
        )
    assert proc.returncode == 0, output


def write_mmbert_classifier(root: Path) -> None:
    """A tiny sequence classifier with the mmBERT config signature."""
    root.mkdir()
    encoder_tokenizer(root)
    labels = ["other", "math"]
    config = modernbert_config(
        200_000,
        architectures=[SEQUENCE],
        classifier_activation="gelu",
        classifier_bias=False,
        classifier_pooling="cls",
        id2label={str(index): label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
        position_embedding_type="sans_pos",
        problem_type="single_label_classification",
    )
    (root / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    weights = {
        f"model.{name}": value
        for name, value in random_backbone("modernbert", config, 0).items()
    }
    torch.manual_seed(1)
    head = ClassifierHead(config, len(labels))
    for name, parameter in head.named_parameters():
        if name.startswith("norm."):
            torch.nn.init.normal_(parameter, mean=1.0, std=0.05)
        else:
            torch.nn.init.normal_(parameter, std=0.3)
    prefixes = {
        "dense.": "head.dense.",
        "norm.": "head.norm.",
        "classifier.": "classifier.",
    }
    for name, value in head.state_dict().items():
        prefix = next(item for item in prefixes if name.startswith(item))
        weights[prefixes[prefix] + name[len(prefix) :]] = value.detach().float()
    save(weights, root / "model.safetensors")
