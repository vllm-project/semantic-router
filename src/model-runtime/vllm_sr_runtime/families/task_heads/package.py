"""HF ModernBERT task packages: what the family loads and which head a checkpoint carries.

A package is a Transformers checkpoint directory: ``config.json`` (a
ModernBERT task architecture and its labels), ``model.safetensors`` (the
backbone under ``model.`` plus ``head.*`` and ``classifier.*``), the tokenizer
files and, for calibrated heads, ``operating_point.json``. Only these files
are downloaded and hashed; bundled code, ONNX graphs and training artifacts
are never read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import sha256_json

MODEL_TYPE = "modernbert"
SEQUENCE = "ModernBertForSequenceClassification"
TOKEN = "ModernBertForTokenClassification"
ARCHITECTURES = (SEQUENCE, TOKEN)
WEIGHTS = "model.safetensors"
REQUIRED = (
    "config.json",
    WEIGHTS,
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
OPERATING_POINT = "operating_point.json"
BACKBONE_PREFIX = "model."
IDENTITY_FORMAT = "task-heads/1"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"{path.name} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageError(f"{path.name} must hold a JSON object")
    return value


def is_bio(labels: tuple[str, ...]) -> bool:
    return any(label.startswith(("B-", "I-")) for label in labels)


@dataclass(frozen=True)
class TaskPackage:
    """A recognised task checkpoint: its config, head kind, labels and policy, before any weights load."""

    root: Path
    config: dict[str, Any]
    kind: str
    labels: tuple[str, ...]
    operating_point: dict[str, Any] | None

    @property
    def files(self) -> tuple[str, ...]:
        """Every file the family loads from this package."""
        return REQUIRED + (
            (OPERATING_POINT,) if self.operating_point is not None else ()
        )

    @property
    def max_positions(self) -> int:
        return int(self.config["max_position_embeddings"])


def labels_of(config: dict[str, Any]) -> tuple[str, ...]:
    """``id2label`` in index order; the indices must be exactly ``0..n-1``."""
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        raise PackageError("config.json declares no labels")
    try:
        indexed = {int(key): str(value) for key, value in id2label.items()}
    except ValueError as exc:
        raise PackageError("config.json id2label keys must be integers") from exc
    if sorted(indexed) != list(range(len(indexed))):
        raise PackageError("config.json id2label must cover 0..n-1")
    return tuple(indexed[index] for index in range(len(indexed)))


def detect(root: Path) -> bool:
    """Cheap ownership test: a ModernBERT config with a task architecture."""
    try:
        config = read_json(root / "config.json")
    except PackageError:
        return False
    return config.get("model_type") == MODEL_TYPE and bool(
        set(config.get("architectures") or ()) & set(ARCHITECTURES)
    )


def read(root: Path) -> TaskPackage:
    """The package's head kind and labels; refuses checkpoints the family cannot serve exactly."""
    config = read_json(root / "config.json")
    if config.get("model_type") != MODEL_TYPE:
        raise PackageError(f"{root.name} is not a ModernBERT checkpoint")
    architectures = [a for a in config.get("architectures") or () if a in ARCHITECTURES]
    if len(architectures) != 1:
        raise PackageError(
            "config.json must name exactly one ModernBERT task architecture"
        )
    labels = labels_of(config)
    path = root / OPERATING_POINT
    policy = read_json(path) if path.is_file() else None
    if architectures[0] == SEQUENCE:
        problem = config.get("problem_type") or "single_label_classification"
        kinds = {
            "single_label_classification": "sequence",
            "multi_label_classification": "scores",
        }
        if problem not in kinds:
            raise PackageError(f"unsupported problem_type {problem!r}")
        kind = kinds[problem]
        if kind == "sequence" and policy is not None:
            raise PackageError("a softmax sequence head has no operating point")
    elif policy is not None and "input_pair" in policy:
        kind = "grounded"
    elif is_bio(labels):
        kind = "token"
    else:
        raise PackageError("a token classifier needs BIO labels or a grounding policy")
    return TaskPackage(root, config, kind, labels, policy)


def identity(files: dict[str, str]) -> str:
    """The model identity: a digest of the loaded files' digests."""
    return sha256_json({"format": IDENTITY_FORMAT, "files": files})
