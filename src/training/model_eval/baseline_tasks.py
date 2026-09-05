"""Task definitions and artifact loading for the Router Model quality baseline.

The class order used to score a model comes from the artifact itself. Any list
kept elsewhere is a copy that can drift, so a copy is only ever used to
cross-check, never as the source of truth.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from artifact_inventory import ServedArtifact
from constants import MODEL_REGISTRY
from datasets import load_dataset
from peft import PeftModel
from provenance.manifest import load_manifest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("QualityBaseline")

# Evaluation-registry keys that name the same task under a different word.
REGISTRY_ALIASES = {"domain": "intent"}
MAX_REPORTED_UNMAPPED = 10


@dataclass(frozen=True)
class TaskSpec:
    """How to obtain held-out data for one task."""

    dataset_repo: str
    split: str
    text_field: str
    label_field: str


# Only text-classification tasks with a published held-out split are wired up.
# PII is token classification and modality has no public eval split, so both are
# reported as coverage gaps rather than measured with a stand-in dataset.
TASK_SPECS: dict[str, TaskSpec] = {
    "jailbreak": TaskSpec(
        dataset_repo="llm-semantic-router/jailbreak-detection-dataset",
        split="test",
        text_field="text",
        label_field="label",
    ),
    "fact-check": TaskSpec(
        dataset_repo="llm-semantic-router/fact-check-classification-dataset",
        split="test",
        text_field="text",
        label_field="label_id",
    ),
    "feedback": TaskSpec(
        dataset_repo="llm-semantic-router/feedback-detector-dataset",
        split="validation",
        text_field="text",
        # The published split names this column label_name. The evaluation
        # registry declares "label" and only works through a silent auto-detect
        # fallback, so the field is pinned here instead.
        label_field="label_name",
    ),
    "domain": TaskSpec(
        dataset_repo="TIGER-Lab/MMLU-Pro",
        split="test",
        text_field="question",
        label_field="category",
    ),
}


class BaselineError(RuntimeError):
    """Raised when the run cannot produce a trustworthy baseline."""


def resolve_label_mapping(model_dir: Path, artifact: ServedArtifact) -> dict[str, int]:
    """Take the class order from the artifact itself, not from the harness.

    The artifact ships the order its classifier head was trained with. Any list
    kept elsewhere is a copy that can drift, so a copy is only ever used to
    cross-check, never as the source of truth.
    """
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    mapping: dict[str, int] = {}
    if isinstance(id2label, dict):
        try:
            mapping = {
                str(id2label[str(index)]): index for index in range(len(id2label))
            }
        except KeyError:
            mapping = {}

    if not mapping:
        mapping = _mapping_from_sidecar(model_dir)
    if not mapping:
        raise BaselineError(
            f"{artifact.artifact_name} does not publish a usable label order; "
            f"neither {config_path.name} id2label nor a mapping sidecar could be read"
        )
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise BaselineError(
            f"{artifact.artifact_name} label order is not contiguous: {mapping}"
        )
    return mapping


def _mapping_from_sidecar(model_dir: Path) -> dict[str, int]:
    for name in (
        "label_mapping.json",
        "category_mapping.json",
        "jailbreak_type_mapping.json",
        "feedback_mapping.json",
    ):
        path = model_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("label_to_id", "category_to_idx", "label_to_idx"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate:
                return {str(name): int(index) for name, index in candidate.items()}
        for key in ("idx_to_label", "idx_to_category", "id_to_label"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate:
                return {str(name): int(index) for index, name in candidate.items()}
    return {}


def check_registry_label_order(task: str, mapping: dict[str, int]) -> list[str]:
    """Compare the artifact's class order against the evaluation registry copy.

    A permuted order still yields plausible accuracy, so this comparison is the
    only place the mismatch becomes visible.
    """
    registry_key = REGISTRY_ALIASES.get(task, task)
    entry = MODEL_REGISTRY.get(registry_key)
    if not entry:
        return [f"{task}: no evaluation registry entry to cross-check"]
    registry_labels = list(entry.get("labels", []))
    artifact_labels = [
        name for name, _ in sorted(mapping.items(), key=lambda item: item[1])
    ]
    if registry_labels == artifact_labels:
        return []
    if sorted(registry_labels) == sorted(artifact_labels):
        moved = [
            f"{name}: artifact={mapping[name]} registry={registry_labels.index(name)}"
            for name in artifact_labels
            if registry_labels.index(name) != mapping[name]
        ]
        return [
            f"{task}: the evaluation registry lists the same labels in a different "
            f"order than the artifact ({'; '.join(moved)}); every affected class is "
            "scored against the wrong logit"
        ]
    return [
        f"{task}: the evaluation registry labels {registry_labels} do not match the "
        f"artifact labels {artifact_labels}"
    ]


def load_rows(
    spec: TaskSpec, mapping: dict[str, int], limit: int | None
) -> tuple[list[str], np.ndarray, int]:
    """Load the held-out split and map every row onto the artifact's class order."""
    dataset = load_dataset(spec.dataset_repo, split=spec.split)
    available = len(dataset)
    if limit is not None:
        dataset = dataset.select(range(min(available, limit)))

    texts: list[str] = []
    labels: list[int] = []
    unmapped: set[str] = set()
    for row in dataset:
        text = row.get(spec.text_field)
        raw = row.get(spec.label_field)
        if text is None or raw is None:
            continue
        if isinstance(raw, str):
            if raw not in mapping:
                unmapped.add(raw)
                continue
            index = mapping[raw]
        else:
            index = int(raw)
            if index not in mapping.values():
                unmapped.add(str(raw))
                continue
        texts.append(str(text))
        labels.append(index)

    if not texts:
        raise BaselineError(
            f"{spec.dataset_repo}:{spec.split} produced no rows the artifact can "
            f"score; unmapped label values: {sorted(unmapped)[:MAX_REPORTED_UNMAPPED]}"
        )
    if unmapped:
        logger.warning(
            "dropped rows with labels the artifact does not define: %s",
            ", ".join(sorted(unmapped)[:MAX_REPORTED_UNMAPPED]),
        )
    return texts, np.array(labels, dtype=np.int64), available


def tokenizer_class(model_dir: Path) -> str | None:
    """Read the tokenizer class the artifact declares, for runtime qualification."""
    config_path = Path(model_dir) / "tokenizer_config.json"
    if not config_path.is_file():
        return None
    declared = json.loads(config_path.read_text(encoding="utf-8")).get(
        "tokenizer_class"
    )
    return str(declared) if declared else None


def referenced_artifact(path: Path | None) -> dict[str, Any] | None:
    """Load an artifact manifest a training run already published."""
    if path is None:
        return None
    return load_manifest(path, expected_kind="artifact")


def artifact_config(model_dir: Path, model) -> dict[str, Any]:
    """Read the artifact config, falling back to the loaded model for adapters."""
    config_path = model_dir / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text(encoding="utf-8"))
    config = getattr(model, "config", None)
    base = getattr(config, "to_dict", lambda: {})()
    if not base.get("architectures"):
        base["architectures"] = ["ModernBertForSequenceClassification"]
    return base


def load_artifact(model_dir: Path, mapping: dict[str, int]):
    """Load a merged checkpoint, or a LoRA adapter on top of its declared base."""
    adapter_config = model_dir / "adapter_config.json"
    if not adapter_config.is_file():
        return (
            AutoTokenizer.from_pretrained(model_dir),
            AutoModelForSequenceClassification.from_pretrained(model_dir),
        )

    adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
    base_repo = adapter.get("base_model_name_or_path")
    if not base_repo:
        raise BaselineError(
            f"{adapter_config} does not name a base model, so the adapter cannot "
            "be evaluated"
        )
    index_to_label = {index: name for name, index in mapping.items()}
    base = AutoModelForSequenceClassification.from_pretrained(
        base_repo,
        num_labels=len(mapping),
        id2label={index: index_to_label[index] for index in sorted(index_to_label)},
        label2id=dict(mapping),
    )
    tokenizer_dir = model_dir if (model_dir / "tokenizer.json").is_file() else base_repo
    return (
        AutoTokenizer.from_pretrained(tokenizer_dir),
        PeftModel.from_pretrained(base, model_dir).eval(),
    )
