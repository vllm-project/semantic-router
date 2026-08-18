"""Train the reconstructed mmBERT-32K safety classifiers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_CONTRACT_PATH,
    contract_sha256,
    distributed_batch_parameters,
    id2label,
    load_contract,
    task_contract,
)

RELEASE_WORLD_SIZE = 8


@dataclass(frozen=True)
class _TrainingRuntime:
    """Resolved process-local settings shared by the training helpers."""

    stack: dict[str, Any]
    torch: Any
    world_size: int
    per_device_batch: int
    gradient_accumulation: int
    use_bf16: bool
    output_root: Path


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any, *, runtime_values: bool = False) -> None:
    default = _json_default if runtime_values else None
    path.write_text(
        json.dumps(payload, default=default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def validate_materialized_data(data_dir: Path, task_name: str) -> dict[str, Any]:
    task_dir = data_dir / task_name
    missing = [
        str(task_dir / filename)
        for filename in ("train.jsonl", "validation.jsonl", "test.jsonl")
        if not (task_dir / filename).is_file()
    ]
    manifest_path = task_dir / "data_manifest.json"
    if not manifest_path.is_file():
        missing.append(str(manifest_path))
    if missing:
        raise FileNotFoundError("Missing prepared data files: " + ", ".join(missing))

    manifest = _read_json(manifest_path)
    if manifest.get("task") != task_name:
        raise ValueError(
            f"Data manifest task {manifest.get('task')!r} does not match {task_name!r}"
        )
    return manifest


def _load_training_stack() -> dict[str, Any]:
    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from datasets import load_dataset  # noqa: PLC0415
        from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install requirements.txt inside "
            "the pinned ROCm container."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _trainer_metrics(task_name: str):
    from .metrics import (  # noqa: PLC0415
        compute_level1_metrics,
        compute_level2_metrics,
    )

    def compute(eval_prediction):
        logits, labels = eval_prediction
        if task_name == "level1":
            return compute_level1_metrics(labels, logits)
        return compute_level2_metrics(labels, logits)

    return compute


def synchronize_best_checkpoint(trainer: Any, torch: Any, trial: Any) -> None:
    """Make rank-local best-checkpoint state consistent after a DDP save.

    Transformers 4.55 updates ``best_model_checkpoint`` only when the newly
    created checkpoint directory is already visible. On a shared single-node
    filesystem, non-zero ranks can perform that check before rank zero creates
    the directory, skip the later load-best barrier, and deadlock rank zero.
    Synchronize after the save and derive the path from the best global step on
    every rank instead of relying on that racy existence check.
    """
    distributed = getattr(torch, "distributed", None)
    is_distributed = bool(
        distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
    )
    if is_distributed:
        device_ids = None
        if torch.cuda.is_available():
            device_ids = [torch.cuda.current_device()]
        distributed.barrier(device_ids=device_ids)

    best_step = getattr(trainer.state, "best_global_step", None)
    if not best_step:
        return
    checkpoint_name = f"checkpoint-{best_step}"
    checkpoint = Path(trainer._get_output_dir(trial)) / checkpoint_name
    if checkpoint.is_dir():
        trainer.state.best_model_checkpoint = str(checkpoint)
        return

    resumed_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if resumed_checkpoint:
        resumed_path = Path(resumed_checkpoint)
        if resumed_path.name == checkpoint_name and resumed_path.is_dir():
            trainer.state.best_model_checkpoint = str(resumed_path)
            return
    raise FileNotFoundError(
        f"best checkpoint is missing after synchronized save: {checkpoint}"
    )


def synchronized_checkpoint_trainer(base_trainer: type[Any], torch: Any) -> type[Any]:
    """Wrap the pinned Trainer with a post-save distributed checkpoint fence."""

    class SynchronizedCheckpointTrainer(base_trainer):
        def _save_checkpoint(self, model: Any, trial: Any) -> None:
            super()._save_checkpoint(model, trial)
            synchronize_best_checkpoint(self, torch, trial)

    return SynchronizedCheckpointTrainer


def _tokenize_datasets(
    stack: dict[str, Any], tokenizer: Any, task_dir: Path, max_length: int
) -> dict[str, Any]:
    data_files = {
        split: str(task_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    datasets = stack["load_dataset"]("json", data_files=data_files)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = batch["label_id"]
        return encoded

    for split in datasets:
        required = {"text", "label_id"}
        missing = required.difference(datasets[split].column_names)
        if missing:
            raise ValueError(f"{split} is missing columns: {sorted(missing)}")
        datasets[split] = datasets[split].map(
            tokenize,
            batched=True,
            remove_columns=datasets[split].column_names,
            desc=f"Tokenizing {split}",
        )
    return datasets


def _load_training_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    contract = load_contract(args.contract)
    task = task_contract(contract, args.task)
    data_dir = Path(args.data_dir).resolve()
    data_manifest = validate_materialized_data(data_dir, args.task)
    manifest_contract_sha = data_manifest.get("contract_sha256")
    if manifest_contract_sha is None:
        manifest_contract_sha = data_manifest.get("provenance", {}).get(
            "contract_sha256"
        )
    if manifest_contract_sha != contract_sha256(contract):
        raise ValueError("Prepared data contract hash does not match training contract")
    return contract, task, data_dir, data_manifest


def _prepare_output_root(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir).resolve()
    output_is_reusable = args.overwrite_output_dir or args.resume_from_checkpoint
    if output_root.exists() and any(output_root.iterdir()) and not output_is_reusable:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Use --overwrite-output-dir or --resume-from-checkpoint."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _prepare_runtime(
    contract: dict[str, Any], args: argparse.Namespace
) -> _TrainingRuntime:
    stack = _load_training_stack()
    torch = stack["torch"]
    accelerator_available = torch.cuda.is_available()
    if not accelerator_available and not args.allow_cpu:
        raise RuntimeError(
            "No accelerator detected; pass --allow-cpu only for a smoke run"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.expected_world_size and world_size != args.expected_world_size:
        raise RuntimeError(
            f"Expected WORLD_SIZE={args.expected_world_size}, found {world_size}"
        )
    per_device_batch, gradient_accumulation = distributed_batch_parameters(
        contract, world_size
    )
    if args.per_device_train_batch_size is not None:
        per_device_batch = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        gradient_accumulation = args.gradient_accumulation_steps
    stack["set_seed"](int(contract["training"]["seed"]))
    return _TrainingRuntime(
        stack=stack,
        torch=torch,
        world_size=world_size,
        per_device_batch=per_device_batch,
        gradient_accumulation=gradient_accumulation,
        use_bf16=bool(accelerator_available and not args.disable_bf16),
        output_root=_prepare_output_root(args),
    )


def _build_model_and_tokenizer(
    contract: dict[str, Any], task: dict[str, Any], runtime: _TrainingRuntime
) -> tuple[Any, Any]:
    stack = runtime.stack
    base = contract["base_model"]
    model_config = contract["model"]
    labels = task["label2id"]
    inverse_labels = id2label(task)
    dtype = runtime.torch.bfloat16 if runtime.use_bf16 else runtime.torch.float32
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        base["id"],
        revision=base["revision"],
        model_max_length=int(model_config["max_length"]),
        use_fast=True,
    )
    model = stack["AutoModelForSequenceClassification"].from_pretrained(
        base["id"],
        revision=base["revision"],
        num_labels=int(task["num_labels"]),
        id2label=inverse_labels,
        label2id=labels,
        problem_type="single_label_classification",
        torch_dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.reference_compile = bool(model_config["reference_compile"])
    lora = model_config["lora"]
    peft_config = stack["LoraConfig"](
        task_type=stack["TaskType"].SEQ_CLS,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias=lora["bias"],
    )
    model = stack["get_peft_model"](model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _build_training_arguments(
    contract: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    runtime: _TrainingRuntime,
) -> Any:
    training = contract["training"]
    epochs = args.num_train_epochs or float(training["epochs"])
    learning_rate = args.learning_rate or float(training["learning_rate"])
    return runtime.stack["TrainingArguments"](
        output_dir=str(runtime.output_root / "checkpoints"),
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=runtime.per_device_batch,
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=runtime.gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=float(training["warmup_ratio"]),
        weight_decay=float(training["weight_decay"]),
        optim=training["optimizer"],
        lr_scheduler_type=training["scheduler"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model=task["selection_metric"],
        greater_is_better=True,
        bf16=runtime.use_bf16,
        fp16=False,
        seed=int(training["seed"]),
        data_seed=int(training["data_seed"]),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=runtime.torch.cuda.is_available(),
        ddp_find_unused_parameters=False if runtime.world_size > 1 else None,
        report_to=[],
        run_name=f"{contract['reconstruction_name']}-{args.task}",
        save_safetensors=True,
    )


def _build_trainer(
    contract: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    runtime: _TrainingRuntime,
    model: Any,
    tokenizer: Any,
    datasets: dict[str, Any],
) -> Any:
    stack = runtime.stack
    trainer_class = synchronized_checkpoint_trainer(stack["Trainer"], runtime.torch)
    return trainer_class(
        model=model,
        args=_build_training_arguments(contract, task, args, runtime),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=stack["DataCollatorWithPadding"](
            tokenizer=tokenizer, pad_to_multiple_of=8
        ),
        compute_metrics=_trainer_metrics(args.task),
        callbacks=[
            stack["EarlyStoppingCallback"](
                early_stopping_patience=int(
                    contract["training"]["early_stopping_patience"]
                )
            )
        ],
    )


def _release_eligible(
    args: argparse.Namespace, world_size: int, use_bf16: bool
) -> bool:
    """Return whether a run used the unmodified release training contract."""
    return bool(
        args.max_steps < 0
        and args.num_train_epochs is None
        and args.learning_rate is None
        and args.per_device_train_batch_size is None
        and args.gradient_accumulation_steps is None
        and world_size == RELEASE_WORLD_SIZE
        and use_bf16
    )


def _build_training_manifest(
    contract: dict[str, Any],
    task_name: str,
    source_manifest_path: Path,
    model: Any,
    runtime: _TrainingRuntime,
    is_release_eligible: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": task_name,
        "contract_sha256": contract_sha256(contract),
        "data_manifest_sha256": _file_sha256(source_manifest_path),
        "source_commit": _source_commit(),
        "base_model": contract["base_model"],
        "taxonomy_version": contract["taxonomy_version"],
        "world_size": runtime.world_size,
        "global_train_batch_size": (
            runtime.per_device_batch
            * runtime.world_size
            * runtime.gradient_accumulation
        ),
        "per_device_train_batch_size": runtime.per_device_batch,
        "gradient_accumulation_steps": runtime.gradient_accumulation,
        "precision": "bf16" if runtime.use_bf16 else "fp32",
        "reference_compile": model.config.reference_compile,
        "distributed_runtime": {
            name: os.environ.get(name)
            for name in (
                "HSA_NO_SCRATCH_RECLAIM",
                "NCCL_MAX_NCHANNELS",
                "NCCL_P2P_DISABLE",
            )
        },
        "release_eligible": is_release_eligible,
        "python": platform.python_version(),
        "packages": _package_versions(
            (
                "torch",
                "transformers",
                "peft",
                "datasets",
                "accelerate",
                "huggingface-hub",
                "safetensors",
            )
        ),
    }


def _write_run_receipts(
    contract: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    runtime: _TrainingRuntime,
    data_dir: Path,
    data_manifest: dict[str, Any],
    model: Any,
    metrics: dict[str, Any],
    adapter_dir: Path,
) -> None:
    inverse_labels = id2label(task)
    _write_json(
        adapter_dir / "label_mapping.json",
        {
            "label2id": task["label2id"],
            "id2label": {str(key): value for key, value in inverse_labels.items()},
            "taxonomy_version": contract["taxonomy_version"],
        },
    )
    _write_json(runtime.output_root / "metrics.json", metrics, runtime_values=True)
    _write_json(runtime.output_root / "data_manifest.json", data_manifest)
    source_manifest_path = data_dir / args.task / "data_manifest.json"
    manifest = _build_training_manifest(
        contract,
        args.task,
        source_manifest_path,
        model,
        runtime,
        _release_eligible(args, runtime.world_size, runtime.use_bf16),
    )
    _write_json(runtime.output_root / "training_manifest.json", manifest)
    _write_json(runtime.output_root / "reconstruction_contract.json", contract)


def _save_training_output(
    contract: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    runtime: _TrainingRuntime,
    data_dir: Path,
    data_manifest: dict[str, Any],
    model: Any,
    tokenizer: Any,
    trainer: Any,
    metrics: dict[str, Any],
) -> Path:
    adapter_dir = runtime.output_root / "adapter"
    trainer.save_model(str(adapter_dir))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_dir)
        _write_run_receipts(
            contract,
            task,
            args,
            runtime,
            data_dir,
            data_manifest,
            model,
            metrics,
            adapter_dir,
        )
    return adapter_dir


def train(args: argparse.Namespace) -> Path:
    contract, task, data_dir, data_manifest = _load_training_inputs(args)
    if args.validate_only:
        print(json.dumps(data_manifest, indent=2, sort_keys=True))
        return data_dir / args.task
    runtime = _prepare_runtime(contract, args)
    model, tokenizer = _build_model_and_tokenizer(contract, task, runtime)
    datasets = _tokenize_datasets(
        runtime.stack,
        tokenizer,
        data_dir / args.task,
        int(contract["model"]["max_length"]),
    )
    trainer = _build_trainer(contract, task, args, runtime, model, tokenizer, datasets)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    validation_metrics = trainer.evaluate(
        datasets["validation"], metric_key_prefix="validation"
    )
    test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    metrics = {
        "train": train_result.metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    return _save_training_output(
        contract,
        task,
        args,
        runtime,
        data_dir,
        data_manifest,
        model,
        tokenizer,
        trainer,
        metrics,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("level1", "level2"))
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--disable-bf16", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = train(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Training output: {output}")


if __name__ == "__main__":
    main()
