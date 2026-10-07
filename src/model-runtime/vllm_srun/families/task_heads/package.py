"""HF task packages: what the family loads and which head a checkpoint carries.

A classifier is a Transformers checkpoint directory: ``config.json`` (a
ModernBERT task architecture and its labels), ``model.safetensors`` (the
backbone under ``model.`` plus ``head.*`` and ``classifier.*``), the tokenizer
files and, for calibrated heads, ``operating_point.json``. An embedder is a
sentence-transformers package over a ModernBERT or Qwen3 backbone (pooling,
modules and prompt files, ``heads.pooled``); a reranker a ``ModernBertModel``
with a Matryoshka layout and trained pair scorers (``heads.relevance``); both
keep their backbone at the root of ``model.safetensors`` and may ship ONNX
exit graphs. Only the files a kind loads are downloaded and hashed; bundled
code and training artifacts are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...heads.pooled import PooledLayout, is_pooled
from ...heads.relevance import RelevanceLayout, is_reranker
from ...registry.artifacts import read_json, sha256_json

MODEL_TYPE = "modernbert"
DECODER_TYPE = "qwen3"
SEQUENCE = "ModernBertForSequenceClassification"
TOKEN = "ModernBertForTokenClassification"
ENCODER = "ModernBertModel"
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
# What fetch() downloads for a Hub package without a built-in entry (patterns for every kind).
FETCH_PATTERNS = (
    *REQUIRED,
    OPERATING_POINT,
    "1_Pooling/config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "matryoshka_config.json",
    "classification_heads.safetensors",
    "onnx/model.onnx",
    "onnx/model_layer_*.onnx",
    "onnx/weights.data",
)


def is_bio(labels: tuple[str, ...]) -> bool:
    return any(label.startswith(("B-", "I-")) for label in labels)


@dataclass(frozen=True)
class TaskPackage:
    """A recognised task checkpoint: its config, head kind, labels and policy, before any weights load.

    ``layout`` is an embedder's ``PooledLayout`` or a reranker's
    ``RelevanceLayout``; classifiers have none.
    """

    root: Path
    config: dict[str, Any]
    kind: str
    labels: tuple[str, ...]
    operating_point: dict[str, Any] | None
    layout: PooledLayout | RelevanceLayout | None = None

    @property
    def model_type(self) -> str:
        return str(self.config["model_type"])

    @property
    def weight_prefix(self) -> str:
        """Classifiers keep the backbone under ``model.``; embedders and rerankers at the root."""
        return BACKBONE_PREFIX if self.layout is None else ""

    @property
    def files(self) -> tuple[str, ...]:
        """Every file the family loads from this package."""
        if isinstance(self.layout, PooledLayout):
            base = tuple(name for name in REQUIRED if (self.root / name).is_file())
            return base + self.layout.files
        if isinstance(self.layout, RelevanceLayout):
            return REQUIRED + self.layout.files(self.root, self.layout.exits)
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
    """Cheap ownership test: a ModernBERT task checkpoint, embedder or reranker, or a Qwen3 embedder."""
    try:
        config = read_json(root / "config.json", mapping=True)
    except PackageError:
        return False
    model_type, architectures = config.get("model_type"), set(
        config.get("architectures") or ()
    )
    if model_type == MODEL_TYPE and architectures & set(ARCHITECTURES):
        return True
    if model_type == MODEL_TYPE and architectures == {ENCODER}:
        return is_pooled(root) or is_reranker(root)
    return model_type == DECODER_TYPE and is_pooled(root)


def read(root: Path, selection: tuple[int, int] | None = None) -> TaskPackage:
    """The package's head kind and labels; refuses checkpoints the family cannot serve exactly.

    ``selection`` pins a reranker's served pair-scorer exit.
    """
    config = read_json(root / "config.json", mapping=True)
    if config.get("model_type") in (MODEL_TYPE, DECODER_TYPE) and not set(
        config.get("architectures") or ()
    ) & set(ARCHITECTURES):
        if is_reranker(root) and config.get("model_type") == MODEL_TYPE:
            layout = RelevanceLayout.read(root, config, selection)
            return TaskPackage(root, config, "relevance", (), None, layout)
        if is_pooled(root):
            return TaskPackage(
                root, config, "pooled", (), None, PooledLayout.read(root, config)
            )
        raise PackageError(f"{root.name} is neither an embedder nor a reranker package")
    if config.get("model_type") != MODEL_TYPE:
        raise PackageError(f"{root.name} is not a ModernBERT checkpoint")
    architectures = [a for a in config.get("architectures") or () if a in ARCHITECTURES]
    if len(architectures) != 1:
        raise PackageError(
            "config.json must name exactly one ModernBERT task architecture"
        )
    labels = labels_of(config)
    path = root / OPERATING_POINT
    policy = read_json(path, mapping=True) if path.is_file() else None
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
