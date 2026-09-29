"""Freeze a selected adapter or full model as a standard HF sequence checkpoint."""

# ruff: noqa: PLC0415

import argparse
import hashlib
import json
from pathlib import Path

from .data import load_contract
from .model import configuration_overrides, load_model, load_task_config
from .optimization import validate_method, verify_full_checkpoint
from .runtime_mapping import RUNTIME_TASKS, write_runtime_mappings
from .task_head import (
    assert_task_head_preserved,
    task_head_scope,
    verify_saved_task_head,
)


def validate_provenance(args):
    validate_method(args.method, args.adapter, False)
    run = json.loads(args.run_manifest.read_text())
    if run.get("test_used") is not False:
        raise ValueError("Candidate provenance must explicitly exclude test selection")
    if run.get("base_revision") != args.base_revision:
        raise ValueError("Export base revision differs from the training receipt")
    if run.get("base_model", args.base_id) != args.base_id:
        raise ValueError("Export base model differs from the training receipt")
    if run.get("method", "lora") != args.method:
        raise ValueError("Export method differs from the training receipt")
    if args.method == "full":
        origin = json.loads((Path(args.base) / "training-origin.json").read_text())
        expected = {
            "method": "full",
            "base_model": args.base_id,
            "base_revision": args.base_revision,
        }
        if origin != expected:
            raise ValueError("Full checkpoint origin differs from the training receipt")
        config = json.loads((Path(args.base) / "config.json").read_text())
        labels, ids = load_contract(args.contract)
        contract = configuration_overrides(args.contract)
        contract.update(label2id=labels, id2label={str(k): v for k, v in ids.items()})
        if any(config.get(key) != value for key, value in contract.items()):
            raise ValueError("Export contract differs from the trained full checkpoint")


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--runtime-task", choices=RUNTIME_TASKS, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a frozen candidate")
    validate_provenance(args)
    torch.set_num_threads(8)
    model, tokenizer, labels, _ = load_model(args.base, args.contract, args.adapter)
    model.eval()
    if args.method == "full":
        verify_full_checkpoint(model, args.base)
        head = task_head_scope(model)
    else:
        head = verify_saved_task_head(model, args.adapter / "adapter_model.safetensors")
    tokens = tokenizer(
        ["Numerical checkpoint equivalence probe.", "用于核对模型数值的输入。"],
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        before = model(**tokens).logits
        exported = (
            model.merge_and_unload(safe_merge=True) if args.method == "lora" else model
        )
        after = exported(**tokens).logits
    assert_task_head_preserved(head, exported)
    torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)
    if type(exported).__name__ != "ModernBertForSequenceClassification":
        raise ValueError(
            "Export must preserve the standard supported HF sequence architecture"
        )
    if exported.config.label2id != labels:
        raise ValueError("Export changed label IDs")
    exported.config._name_or_path = args.base_id
    args.output.mkdir(parents=True, exist_ok=True)
    exported.save_pretrained(args.output, safe_serialization=True)
    verify_full_checkpoint(exported, args.output)
    tokenizer.save_pretrained(args.output)
    restored = (
        type(exported)
        .from_pretrained(
            args.output,
            config=load_task_config(args.output),
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        .eval()
    )
    verify_full_checkpoint(restored, args.output)
    assert_task_head_preserved(head, restored)
    with torch.inference_mode():
        reloaded_logits = restored(**tokens).logits
    torch.testing.assert_close(after, reloaded_logits, atol=0, rtol=0)
    mapping_files = write_runtime_mappings(args.output, labels, args.runtime_task)
    receipt = {
        "method": args.method,
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "run_manifest_sha256": hashlib.sha256(
            args.run_manifest.read_bytes()
        ).hexdigest(),
        "parameter_count": sum(
            parameter.numel() for parameter in exported.parameters()
        ),
        "encoder_parameter_count": sum(
            parameter.numel() for parameter in exported.model.parameters()
        ),
        "label2id": labels,
        "runtime_task": args.runtime_task,
        "runtime_mapping_files": mapping_files,
        "task": exported.config.problem_type,
        "classifier_pooling": exported.config.classifier_pooling,
        "max_position_embeddings": exported.config.max_position_embeddings,
        "position_capacity_is_quality_evidence": False,
        "weight_dtype": "float32",
        "task_head": head,
        "task_head_preserved_after_reload": True,
        "all_tensors_preserved_after_reload": True,
        "test_used_for_selection": False,
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output.iterdir())
            if path.is_file()
        },
    }
    if args.method == "lora":
        receipt.update(
            adapter_sha256=hashlib.sha256(
                (args.adapter / "adapter_model.safetensors").read_bytes()
            ).hexdigest(),
            task_head_preserved_after_merge_and_reload=True,
            merge_probe_max_absolute_difference=float((before - after).abs().max()),
        )
    (args.output / "candidate-lock.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "method",
                    "parameter_count",
                    "all_tensors_preserved_after_reload",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
