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
    build_evaluation_manifest,
    build_run_manifest,
    code_revision,
    resolve_hf_revision,
    split_digest,
    write_manifest,
)
from provenance.manifest import load_manifest  # noqa: E402
from provenance.metrics import (  # noqa: E402
    abstention_curve,
    calibration_metrics,
    classification_metrics,
    latency_percentiles,
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


def resolve_training_pins(*, base_model_repo: str) -> dict[str, Any]:
    """Resolve every upstream revision once, before anything is loaded.

    The manifests have to describe the bytes the run actually read. Resolving
    after training records whatever the ref points at by then, so a moving ref
    or a stale cache would produce provenance that looks valid and is not. The
    caller passes these revisions into the loads and back into the manifests.
    """
    return {
        "base_model": {
            "repo": base_model_repo,
            "revision": resolve_hf_revision(base_model_repo),
        },
        "datasets": {
            key: resolve_hf_revision(entry["name"], repo_type="dataset")
            for key, entry in DATASET_CONFIGS.items()
        },
    }


def emit_training_manifests(
    *,
    pins: dict[str, Any],
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
        dataset_pins=pins["datasets"],
    )
    run = build_run_manifest(
        manifest_id=run_id,
        task=TASK,
        base_model_repo=base_model_repo,
        base_model_revision=pins["base_model"]["revision"],
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


def emit_evaluation_manifest(
    *,
    manifest_dir: str | Path,
    artifact_manifest_path: str | Path,
    dataset_manifest_path: str | Path,
    label_to_id: dict[str, int],
    seed: int,
    batch_size: int,
    max_length: int,
    device: str,
    device_name: str | None,
    sample_limit: int | None,
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    latencies_ms: list[float],
    peak_memory_mb: float,
    logger,
) -> Path:
    """Write the evaluation manifest for the run that just finished.

    The adapter is measured on the validation split the same run held out, so
    the manifest records ``by_row``: this workflow splits its own mixture
    positionally rather than holding out whole sources, which is the leak a
    reader has to be able to see from the manifest alone.
    """
    manifest_dir = Path(manifest_dir)
    artifact = load_manifest(Path(artifact_manifest_path), expected_kind="artifact")
    dataset = load_manifest(Path(dataset_manifest_path), expected_kind="dataset")
    label_mapping = dict(label_to_id)
    evaluation_id = f"{artifact['id']}-validation"

    evaluation = build_evaluation_manifest(
        manifest_id=evaluation_id,
        task=TASK,
        artifact_ref={
            "id": artifact["id"],
            "revision": artifact["identity"]["revision"],
            "digest": artifact["identity"]["digest"],
        },
        dataset_ref={
            "id": dataset["id"],
            "revision": dataset["source"]["revision"],
            "splits": ["validation"],
        },
        split_rule="by_row",
        entrypoint=TRAINING_ENTRYPOINT,
        repo_root=REPO_ROOT,
        device=device,
        device_name=device_name,
        batch_size=batch_size,
        max_length=max_length,
        sample_limit=sample_limit,
        seed=seed,
        label_mapping=label_mapping,
        metrics=classification_metrics(y_true, y_pred, label_mapping),
        calibration=calibration_metrics(y_true, y_pred, confidences),
        abstention=abstention_curve(y_true, y_pred, confidences),
        performance={
            "latency_ms": latency_percentiles(latencies_ms),
            "peak_memory_mb": peak_memory_mb,
        },
        description=(
            "Validation of the adapter this run produced, on the split it held out."
        ),
    )
    path = write_manifest(evaluation, manifest_dir / f"{evaluation_id}.manifest.yaml")
    logger.info(f"Wrote evaluation manifest: {path}")
    return path


def _build_dataset(
    *,
    dataset_id: str,
    label_mapping: dict[str, int],
    train_rows: list[tuple[str, int]],
    validation_rows: list[tuple[str, int]],
    dataset_pins: dict[str, str],
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
    dataset["source"]["components"] = _components(code_sha, dataset_pins)
    return dataset


def _components(code_sha: str, dataset_pins: dict[str, str]) -> list[dict[str, str]]:
    components = []
    for key, entry in DATASET_CONFIGS.items():
        component = {
            "type": "huggingface",
            "locator": entry["name"],
            "revision": dataset_pins[key],
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


__all__ = [
    "ProvenanceError",
    "emit_evaluation_manifest",
    "emit_training_manifests",
    "resolve_training_pins",
]
