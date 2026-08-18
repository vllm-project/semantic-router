"""Train the reconstructed mmBERT-32K safety classifiers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
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
        import numpy as np
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
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
    from .metrics import compute_level1_metrics, compute_level2_metrics

    def compute(eval_prediction):
        logits, labels = eval_prediction
        if task_name == "level1":
            return compute_level1_metrics(labels, logits)
        return compute_level2_metrics(labels, logits)

    return compute


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


def train(args: argparse.Namespace) -> Path:
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
    if args.validate_only:
        print(json.dumps(data_manifest, indent=2, sort_keys=True))
        return data_dir / args.task

    stack = _load_training_stack()
    torch = stack["torch"]
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "No accelerator detected; pass --allow-cpu only for a smoke run"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.expected_world_size and world_size != args.expected_world_size:
        raise RuntimeError(
            f"Expected WORLD_SIZE={args.expected_world_size}, found {world_size}"
        )

    training = contract["training"]
    per_device_batch, gradient_accumulation = distributed_batch_parameters(
        contract, world_size
    )
    if args.per_device_train_batch_size is not None:
        per_device_batch = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        gradient_accumulation = args.gradient_accumulation_steps

    seed = int(training["seed"])
    stack["set_seed"](seed)
    output_root = Path(args.output_dir).resolve()
    if (
        output_root.exists()
        and any(output_root.iterdir())
        and not (args.overwrite_output_dir or args.resume_from_checkpoint)
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Use --overwrite-output-dir or --resume-from-checkpoint."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    base = contract["base_model"]
    labels = task["label2id"]
    inverse_labels = id2label(task)
    model_config = contract["model"]
    use_bf16 = bool(torch.cuda.is_available() and not args.disable_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float32

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

    datasets = _tokenize_datasets(
        stack, tokenizer, data_dir / args.task, int(model_config["max_length"])
    )
    trainer_output = output_root / "checkpoints"
    epochs = args.num_train_epochs or float(training["epochs"])
    learning_rate = args.learning_rate or float(training["learning_rate"])
    training_args = stack["TrainingArguments"](
        output_dir=str(trainer_output),
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=gradient_accumulation,
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
        bf16=use_bf16,
        fp16=False,
        seed=seed,
        data_seed=int(training["data_seed"]),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to=[],
        run_name=f"{contract['reconstruction_name']}-{args.task}",
        save_safetensors=True,
    )
    trainer = stack["Trainer"](
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=stack["DataCollatorWithPadding"](
            tokenizer=tokenizer, pad_to_multiple_of=8
        ),
        compute_metrics=_trainer_metrics(args.task),
        callbacks=[
            stack["EarlyStoppingCallback"](
                early_stopping_patience=int(training["early_stopping_patience"])
            )
        ],
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    validation_metrics = trainer.evaluate(
        datasets["validation"], metric_key_prefix="validation"
    )
    test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")

    adapter_dir = output_root / "adapter"
    if trainer.is_world_process_zero():
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)
        label_mapping = {
            "label2id": labels,
            "id2label": {str(key): value for key, value in inverse_labels.items()},
            "taxonomy_version": contract["taxonomy_version"],
        }
        (adapter_dir / "label_mapping.json").write_text(
            json.dumps(label_mapping, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = {
            "train": train_result.metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        (output_root / "metrics.json").write_text(
            json.dumps(metrics, default=_json_default, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_manifest_path = data_dir / args.task / "data_manifest.json"
        (output_root / "data_manifest.json").write_text(
            json.dumps(data_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        release_eligible = (
            args.max_steps < 0
            and args.num_train_epochs is None
            and args.learning_rate is None
            and args.per_device_train_batch_size is None
            and args.gradient_accumulation_steps is None
            and world_size == 8
            and use_bf16
        )
        manifest = {
            "schema_version": 1,
            "task": args.task,
            "contract_sha256": contract_sha256(contract),
            "data_manifest_sha256": _file_sha256(source_manifest_path),
            "source_commit": _source_commit(),
            "base_model": base,
            "taxonomy_version": contract["taxonomy_version"],
            "world_size": world_size,
            "global_train_batch_size": (
                per_device_batch * world_size * gradient_accumulation
            ),
            "per_device_train_batch_size": per_device_batch,
            "gradient_accumulation_steps": gradient_accumulation,
            "precision": "bf16" if use_bf16 else "fp32",
            "release_eligible": release_eligible,
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
        (output_root / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_root / "reconstruction_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trainer.save_state()
    return adapter_dir


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
