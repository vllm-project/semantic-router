"""Provenance manifest emission for the prompt-guard LoRA training workflow.

This is the first training script wired into the Router Model provenance
contract. It records what the run actually consumed -- pinned upstream dataset
revisions, the base model revision, the code revision, the seed, and the
hyperparameters -- next to the adapter it produced, so a later evaluation can
prove it measured this artifact and not a similarly named one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_EVAL = REPO_ROOT / "src" / "training" / "model_eval"
if str(_MODEL_EVAL) not in sys.path:
    sys.path.insert(0, str(_MODEL_EVAL))

from jailbreak_training_assets import DATASET_CONFIGS  # noqa: E402
from provenance.emit import (  # noqa: E402
    ProvenanceError,
    build_artifact_manifest,
    build_dataset_manifest,
    build_run_manifest,
    code_revision,
    resolve_hf_revision,
    split_digest,
    write_manifest,
)

TASK = "jailbreak"
TRAINING_ENTRYPOINT = (
    "src/training/model_classifier/prompt_guard_fine_tuning_lora/"
    "jailbreak_bert_finetuning_lora.py"
)
DATASET_BUILDER = (
    "src/training/model_classifier/prompt_guard_fine_tuning_lora/"
    "jailbreak_bert_finetuning_lora.py::create_jailbreak_dataset"
)
PATTERN_ASSETS = (
    "src/training/model_classifier/prompt_guard_fine_tuning_lora/"
    "jailbreak_training_assets.py"
)


def emit_training_manifests(
    *,
    output_dir: str | Path,
    manifest_dir: str | Path | None,
    model_name: str,
    base_model_repo: str,
    label_to_id: dict[str, int],
    seed: int,
    training_args: Any,
    lora_config: dict[str, Any],
    max_samples: int,
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    model: Any,
    logger,
) -> dict[str, Path]:
    """Write the dataset, run, and artifact manifests for one completed run.

    Returns the written paths. Raises :class:`ProvenanceError` rather than
    writing a partial bundle, so a run that cannot prove its inputs fails
    visibly instead of shipping an unverifiable adapter.
    """
    output_dir = Path(output_dir)
    manifest_dir = Path(manifest_dir) if manifest_dir else output_dir / "manifests"
    rank = lora_config["rank"]
    run_id = f"prompt-guard-{model_name}-r{rank}-seed{seed}"
    dataset_id = f"prompt-guard-mixture-max{max_samples}"
    artifact_id = f"prompt-guard-{model_name}-r{rank}-lora"
    label_mapping = dict(label_to_id)

    dataset = _build_dataset(
        dataset_id=dataset_id,
        label_mapping=label_mapping,
        train_rows=_rows(train_data),
        validation_rows=_rows(val_data),
    )
    run = build_run_manifest(
        manifest_id=run_id,
        task=TASK,
        base_model_repo=base_model_repo,
        base_model_revision=resolve_hf_revision(base_model_repo),
        entrypoint=TRAINING_ENTRYPOINT,
        repo_root=REPO_ROOT,
        dataset_refs=[
            {
                "id": dataset_id,
                "revision": dataset["source"]["revision"],
                "splits": ["train", "validation"],
            }
        ],
        seed=seed,
        hyperparameters=_hyperparameters(training_args, lora_config, max_samples),
        label_mapping=label_mapping,
    )
    artifact = build_artifact_manifest(
        manifest_id=artifact_id,
        task=TASK,
        repo=f"local/{artifact_id}",
        revision=_code_sha(),
        artifact_dir=output_dir,
        label_mapping=label_mapping,
        architecture=_architecture(model),
        max_position_embeddings=int(model.config.max_position_embeddings),
        run_id=run_id,
        description="LoRA adapter produced by the prompt-guard training workflow.",
    )

    written = {
        "dataset": write_manifest(
            dataset, manifest_dir / f"{dataset_id}.manifest.yaml"
        ),
        "run": write_manifest(run, manifest_dir / f"{run_id}.manifest.yaml"),
        "artifact": write_manifest(
            artifact, manifest_dir / f"{artifact_id}.manifest.yaml"
        ),
    }
    for kind, path in written.items():
        logger.info(f"Wrote {kind} manifest: {path}")
    return written


def _build_dataset(
    *,
    dataset_id: str,
    label_mapping: dict[str, int],
    train_rows: list[tuple[str, int]],
    validation_rows: list[tuple[str, int]],
) -> dict[str, Any]:
    """Describe the mixture the workflow built, with every upstream pinned."""
    code_sha = _code_sha()
    dataset = build_dataset_manifest(
        manifest_id=dataset_id,
        task=TASK,
        source_type="composite",
        locator=DATASET_BUILDER,
        revision=code_sha,
        license_id="mixed-upstream",
        splits=[
            _split("train", train_rows, label_mapping),
            _split("validation", validation_rows, label_mapping),
        ],
        text_field="text",
        label_field="label",
        preprocessing_steps=[
            "sample upstream jailbreak and toxicity corpora up to --max-samples",
            "append in-repo short and long jailbreak pattern assets",
            "balance classes to the smaller of the two label counts",
            "split 80/20 into train and validation without shuffling",
        ],
        label_mapping=label_mapping,
        description=(
            "Composite prompt-guard training mixture built by the LoRA workflow."
        ),
    )
    dataset["source"]["components"] = _components(code_sha)
    return dataset


def _components(code_sha: str) -> list[dict[str, str]]:
    components = []
    for entry in DATASET_CONFIGS.values():
        component = {
            "type": "huggingface",
            "locator": entry["name"],
            "revision": resolve_hf_revision(entry["name"], repo_type="dataset"),
        }
        if entry.get("config"):
            component["config"] = entry["config"]
        components.append(component)
    components.append(
        {"type": "in-repo", "locator": PATTERN_ASSETS, "revision": code_sha}
    )
    return components


def _split(
    name: str, rows: list[tuple[str, int]], label_mapping: dict[str, int]
) -> dict[str, Any]:
    return {
        "name": name,
        "rows": len(rows),
        "digest": split_digest(rows),
        "label_counts": _label_counts(rows, label_mapping),
    }


def _hyperparameters(
    training_args: Any, lora_config: dict[str, Any], max_samples: int
) -> dict[str, Any]:
    return {
        "lora_rank": lora_config["rank"],
        "lora_alpha": lora_config["alpha"],
        "lora_dropout": lora_config["dropout"],
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "max_samples": max_samples,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "weight_decay": training_args.weight_decay,
        "max_grad_norm": training_args.max_grad_norm,
    }


def _rows(samples: list[dict[str, Any]]) -> list[tuple[str, int]]:
    return [(row["text"], row["label"]) for row in samples]


def _architecture(model: Any) -> str:
    declared = getattr(model.config, "architectures", None)
    return declared[0] if declared else type(model).__name__


def _label_counts(
    rows: list[tuple[str, int]], label_to_id: dict[str, int]
) -> dict[str, int]:
    id_to_label = {index: name for name, index in label_to_id.items()}
    counts = dict.fromkeys(label_to_id, 0)
    for _, label in rows:
        counts[id_to_label[int(label)]] += 1
    return counts


def _code_sha() -> str:
    return code_revision(TRAINING_ENTRYPOINT, REPO_ROOT)["revision"]


__all__ = ["ProvenanceError", "emit_training_manifests"]
