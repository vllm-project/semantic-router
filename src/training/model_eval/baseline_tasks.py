"""Task definitions and artifact loading for the Router Model quality baseline.

The class order used to score a model comes from the artifact itself. Any list
kept elsewhere is a copy that can drift, so a copy is only ever used to
cross-check, never as the source of truth.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import transformers
from artifact_inventory import REGISTRY_ALIASES, ServedArtifact
from baseline_artifact import BaselineError
from constants import LEGACY_MODEL_REGISTRY, MODEL_REGISTRY
from datasets import concatenate_datasets, load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("QualityBaseline")

MAX_REPORTED_UNMAPPED = 10
# Draws the class-matched selection, so the same pinned file always yields the
# same scored rows for every artifact and every rerun.
MATCHED_SELECTION_SEED: int = 20261011


@dataclass(frozen=True)
class TaskSpec:
    """How to obtain held-out data for one task.

    ``split_rule`` is the manifest's own vocabulary for how the rows were held
    out. ``predefined`` is the split the upstream publishes, which for an
    artifact trained on that upstream measures fit rather than generalisation.
    ``by_source`` holds a whole corpus out and is the rule an external
    held-out set carries.

    ``revision`` pins the dataset, so the rows scored are the rows the manifest
    names. ``data_files`` reads the split from these files of the repository
    instead of from a published split. ``match_strata`` names the columns the
    scored rows are class-matched inside, ``qmark`` being computed from the
    text: a stratum that does not carry every label contributes no rows, so no
    single feature of the corpus hands one class to the model.
    """

    dataset_repo: str
    split: str
    text_field: str
    label_field: str
    split_rule: str = "predefined"
    revision: str | None = None
    data_files: tuple[str, ...] = ()
    compatible_artifact_repos: tuple[str, ...] = ()
    # Why other artifacts are refused, completed with {repo}.
    restriction: str = ""
    # Rows whose ``exclude_prefix[0]`` value starts with ``exclude_prefix[1]`` are
    # left out. ``trained_on_repos`` names artifacts trained on the dataset
    # itself, for which none of its rows is held out.
    exclude_prefix: tuple[str, str] | None = None
    trained_on_repos: tuple[str, ...] = ()
    # Columns the scored rows are class-matched inside before any model sees
    # them; ``qmark`` is computed from the text rather than read.
    match_strata: tuple[str, ...] = ()

    def validate_artifact(self, repo: str) -> None:
        """Refuse source labels that cannot rank this artifact."""
        if repo in self.trained_on_repos:
            raise BaselineError(
                f"{repo} was trained on {self.dataset_repo}, so no split of it is "
                "held out for this artifact."
            )
        if (
            self.compatible_artifact_repos
            and repo not in self.compatible_artifact_repos
        ):
            raise BaselineError(
                f"{self.dataset_repo} {self.restriction.format(repo=repo)}"
            )


# Only text-classification tasks with a published held-out split are wired up.
# PII is token classification and modality has no public eval split, so both are
# reported as coverage gaps rather than measured with a stand-in dataset.
TASK_SPECS: dict[str, TaskSpec] = {
    "jailbreak": TaskSpec(
        dataset_repo="vllm-sr/jailbreak-detection-dataset",
        split="test",
        text_field="text",
        label_field="label",
        compatible_artifact_repos=(
            LEGACY_MODEL_REGISTRY["jailbreak"]["id"],
            LEGACY_MODEL_REGISTRY["jailbreak"]["lora_id"],
            "vllm-sr/mmbert-jailbreak-detector-merged",
            "vllm-sr/mmbert-jailbreak-detector-lora",
        ),
        restriction=(
            "is a legacy toxicity/jailbreak diagnostic, not an instruction-attack "
            "benchmark for {repo}. Use mom_collection_eval.py --custom_dataset with "
            "attack-reviewed benign/jailbreak gold for Guard."
        ),
    ),
    # The corpus-matched fact-check test of #4305. Inside every corpus, script,
    # length, question-mark and capitalisation stratum the two classes are equal,
    # so neither the corpus nor those cues give the label away. Dolly is left out
    # because the mmBERT fact-check checkpoint trained on it, and neither
    # fact-check checkpoint names the other four corpora.
    "fact-check": TaskSpec(
        dataset_repo="vllm-sr/router-signal-suite",
        revision="3f95fc2d0dbdd8abdd2a0fba0387f4a925fe934a",
        split="test",
        data_files=(
            "text/nf-cats/fact_check/test.jsonl",
            "text/open-question-type/fact_check/test.jsonl",
            "text/search-arena/fact_check/test.jsonl",
            "text/urs/fact_check/test.jsonl",
        ),
        text_field="text",
        label_field="label",
        split_rule="by_source",
    ),
    # The fresh held-out feedback set of #4305: CrossWOZ booking dialogues, a
    # corpus the suite never touched, deduplicated against every suite row, so
    # it is new to both feedback checkpoints - the legacy detector trained on
    # the feedback-detector dataset and Vela Feedback on WildFeedback and
    # Schema-Guided Dialogue. The fresh build keeps the corpus's own label
    # proportions, and here every satisfied turn is a question-mark-free thank
    # while most other turns are questions, so the punctuation alone separates
    # the classes; the runner scores a class-matched subset inside the source,
    # script, length and question-mark strata instead. The dialogue act the
    # label is read from (subsource) names the class rather than matching it.
    # The set carries SAT and NO_FEEDBACK; the classes with no published
    # held-out text keep #4301 open.
    "feedback": TaskSpec(
        dataset_repo="vllm-sr/router-signal-suite",
        revision="fa08b2a642df30955ad2ad2206d74050c7f12b5c",
        split="test",
        data_files=("text/crosswoz/feedback/fresh-crosswoz.jsonl",),
        text_field="text",
        label_field="label",
        split_rule="by_source",
        match_strata=("source", "script", "length_bin", "qmark"),
    ),
    # Vela Domain trains on Global-MMLU and moves the MMLU questions that match
    # MMLU-Pro into training, so the MMLU-derived rows are left out. The legacy
    # intent classifier trained on MMLU-Pro itself.
    "domain": TaskSpec(
        dataset_repo="TIGER-Lab/MMLU-Pro",
        split="test",
        text_field="question",
        label_field="category",
        split_rule="by_source",
        exclude_prefix=("src", "ori_mmlu"),
        trained_on_repos=(
            LEGACY_MODEL_REGISTRY["intent"]["id"],
            LEGACY_MODEL_REGISTRY["intent"]["lora_id"],
        ),
    ),
}


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


def check_registry_label_order(
    task: str, repo: str, mapping: dict[str, int]
) -> list[str]:
    """Compare the artifact's class order against its own registry copy.

    A permuted order still yields plausible accuracy, so this comparison is the
    only place the mismatch becomes visible. A legacy checkpoint is compared
    with the legacy registry, since the served model may add classes it never
    had, as Vela Feedback adds NO_FEEDBACK.
    """
    registry_key = REGISTRY_ALIASES.get(task, task)
    legacy = LEGACY_MODEL_REGISTRY.get(registry_key, {})
    registry = (
        LEGACY_MODEL_REGISTRY
        if repo in (legacy.get("id"), legacy.get("lora_id"))
        else MODEL_REGISTRY
    )
    entry = registry.get(registry_key)
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


def load_split(spec: TaskSpec, revision: str):
    """Read the split at ``revision``, from its files when the spec names them."""
    if not spec.data_files:
        return load_dataset(spec.dataset_repo, split=spec.split, revision=revision)
    # A row leaves out the fields it has no value for, so the files need not share
    # one column set. Each is read on its own, keeping the fields the baseline reads.
    fields = [spec.text_field, spec.label_field]
    if spec.exclude_prefix is not None:
        fields.append(spec.exclude_prefix[0])
    for name in spec.match_strata:
        if name != "qmark" and name not in fields:
            fields.append(name)
    return concatenate_datasets(
        [
            load_dataset(
                spec.dataset_repo, data_files=name, split="train", revision=revision
            ).select_columns(fields)
            for name in spec.data_files
        ]
    )


def stratum_of(row: dict[str, Any], spec: TaskSpec) -> tuple[Any, ...]:
    """The row's values for ``spec.match_strata``; ``qmark`` comes from the text."""
    text = str(row.get(spec.text_field) or "")
    return tuple(
        ("q" if "?" in text else "n") if name == "qmark" else row.get(name)
        for name in spec.match_strata
    )


def select_matched_rows(
    dataset: Any, spec: TaskSpec, seed: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep a class-balanced subset inside every ``spec.match_strata`` stratum.

    A fresh held-out file keeps its corpus's own label proportions, and a
    corpus can tie one class to a single surface feature: in CrossWOZ every
    satisfied turn is a question-mark-free thank while most other turns are
    questions, so the punctuation alone separates the classes. Scoring the raw
    file would measure that feature instead of the task, so each stratum
    contributes only as many rows of each label as its rarest label has, drawn
    in a seeded order, and the record names the pinned distribution beside the
    counts that were kept.
    """
    rows = list(dataset)
    labels = sorted({str(row.get(spec.label_field)) for row in rows})
    before = dict.fromkeys(labels, 0)
    for row in rows:
        before[str(row.get(spec.label_field))] += 1
    grouped: dict[tuple[tuple[Any, ...], str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (stratum_of(row, spec), str(row.get(spec.label_field)))
        grouped.setdefault(key, []).append(index)
    draw = MATCHED_SELECTION_SEED if seed is None else seed
    rng = random.Random(draw)
    keep: list[int] = []
    selected = dict.fromkeys(labels, 0)
    strata = sorted(
        {stratum for stratum, _ in grouped}, key=lambda s: tuple(str(v) for v in s)
    )
    for stratum in strata:
        groups = {label: grouped.get((stratum, label), []) for label in labels}
        quota = min(len(group) for group in groups.values())
        for label in labels:
            group = groups[label]
            rng.shuffle(group)
            keep.extend(group[:quota])
            selected[label] += quota
    record = {
        "strata": list(spec.match_strata),
        "seed": draw,
        "rows_by_label": before,
        "selected_by_label": selected,
    }
    return [rows[index] for index in sorted(keep)], record


def load_rows(
    spec: TaskSpec, mapping: dict[str, int], limit: int | None, revision: str
) -> tuple[list[str], np.ndarray, int, dict[str, Any] | None]:
    """Load the held-out split and map every row onto the artifact's class order."""
    dataset = load_split(spec, revision)
    if spec.exclude_prefix is not None:
        field, prefix = spec.exclude_prefix
        dataset = dataset.filter(lambda row: not str(row[field]).startswith(prefix))
    available = len(dataset)
    if limit is not None:
        dataset = dataset.select(range(min(available, limit)))
    matched: dict[str, Any] | None = None
    if spec.match_strata:
        dataset, matched = select_matched_rows(dataset, spec)

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
    return texts, np.array(labels, dtype=np.int64), available, matched


def tokenizer_class(model_dir: Path) -> str | None:
    """Read the tokenizer class the artifact declares, for runtime qualification."""
    config_path = Path(model_dir) / "tokenizer_config.json"
    if not config_path.is_file():
        return None
    declared = json.loads(config_path.read_text(encoding="utf-8")).get(
        "tokenizer_class"
    )
    return str(declared) if declared else None


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


def load_tokenizer(model_dir: Path, artifact_name: str):
    """Load the tokenizer, reporting a runtime mismatch as the gap that it is.

    An artifact declares the tokenizer class the library has to know. When the
    installed transformers does not know it, that is a runtime gap between the
    artifact and the portfolio it is served with, and naming both sides is what
    lets a maintainer act on it. The raw ValueError names neither.
    """
    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except ValueError as exc:
        declared = tokenizer_class(model_dir)
        if declared is None or declared not in str(exc):
            raise
        raise BaselineError(
            f"{artifact_name} declares tokenizer_class {declared!r}, which "
            f"transformers {transformers.__version__} does not provide. That is "
            "a runtime gap rather than a quality one, so the artifact cannot be "
            "measured on this interpreter"
        ) from exc


def load_artifact(model_dir: Path, mapping: dict[str, int], artifact_name: str):
    """Load a merged checkpoint, or a LoRA adapter on top of its declared base."""
    adapter_config = model_dir / "adapter_config.json"
    if not adapter_config.is_file():
        return (
            load_tokenizer(model_dir, artifact_name),
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
        load_tokenizer(Path(tokenizer_dir), artifact_name),
        PeftModel.from_pretrained(base, model_dir).eval(),
    )
