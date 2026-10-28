"""Export a selected PII adapter or full checkpoint with identity checks."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate import load_model
from full_training import file_sha256, tensor_receipt, verify_full_checkpoint


def main():
    import torch  # noqa: PLC0415 - optional dependency for data-only tooling

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--method", choices=["lora", "full"], default="lora")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.method == "lora" and (not args.adapter or args.checkpoint):
        raise ValueError("LoRA export requires only an adapter")
    if args.method == "full" and (args.adapter or not args.checkpoint):
        raise ValueError("Full export requires only a native checkpoint")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a merged checkpoint")
    run = json.loads(args.run_manifest.read_text())
    if run.get("test_used") is not False:
        raise ValueError("Candidate provenance must explicitly exclude test selection")
    if run.get("base_revision") != args.base_revision:
        raise ValueError("Export base revision differs from the training receipt")
    if run.get("method", "lora") != args.method:
        raise ValueError("Export method differs from the training receipt")
    if run.get("base_model") != args.base_id:
        raise ValueError("Export base identity differs from the training receipt")
    torch.set_num_threads(8)
    source = args.checkpoint if args.method == "full" else args.base
    model, tokenizer, label_to_id, _ = load_model(source, args.config, args.adapter)
    if args.method == "full":
        receipt = json.loads((args.checkpoint / "full-checkpoint.json").read_text())
        if (
            receipt.get("method") != "full"
            or receipt.get("task") != "token-classification"
            or receipt.get("origin") != run
            or receipt.get("label2id") != label_to_id
            or receipt.get("config_sha256")
            != file_sha256(args.checkpoint / "config.json")
            or receipt.get("tensors") != tensor_receipt(model)
            or receipt.get("tokenizer_files")
            != {
                path.name: file_sha256(path)
                for path in sorted(args.checkpoint.iterdir())
                if path.name
                in {
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                }
            }
        ):
            raise ValueError(
                "Full checkpoint identity differs from its training receipt"
            )
        actual = verify_full_checkpoint(model, args.checkpoint)
        if receipt.get("files") != actual["files"]:
            raise ValueError("Full checkpoint weight files differ from their receipt")
    model.eval()
    # These are numerical probes, not quality evaluation or training examples.
    inputs = tokenizer(
        ["A short numerical equivalence probe.", "用于检查合并前后数值一致性。"],
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        before = model(**inputs).logits
        merged = (
            model.merge_and_unload(safe_merge=True) if args.method == "lora" else model
        )
        after = merged(**inputs).logits
    if not bool(torch.isfinite(before).all() and torch.isfinite(after).all()):
        raise ValueError("Adapter or merged logits are non-finite")
    torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)
    if merged.config.label2id != label_to_id:
        raise ValueError("Merge changed the label contract")
    if type(merged).__name__ != "ModernBertForTokenClassification":
        raise ValueError("Export must preserve the supported HF token architecture")
    merged.config._name_or_path = args.base_id
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    verify_full_checkpoint(merged, args.output)
    reloaded, _, _, _ = load_model(args.output, args.config)
    reloaded.eval()
    with torch.inference_mode():
        reloaded_logits = reloaded(**inputs).logits
    torch.testing.assert_close(after, reloaded_logits, atol=0, rtol=0)
    labels = {
        "label_to_idx": label_to_id,
        "idx_to_label": {str(index): label for label, index in label_to_id.items()},
    }
    for name in ("label_mapping.json", "pii_mapping.json"):
        (args.output / name).write_text(json.dumps(labels, indent=2) + "\n")
    evidence = {
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "method": args.method,
        "checkpoint_receipt_sha256": (
            file_sha256(args.checkpoint / "full-checkpoint.json")
            if args.method == "full"
            else None
        ),
        "adapter_sha256": (
            hashlib.sha256(
                (args.adapter / "adapter_model.safetensors").read_bytes()
            ).hexdigest()
            if args.adapter
            else None
        ),
        "label2id": label_to_id,
        "runtime_mapping_files": ["label_mapping.json", "pii_mapping.json"],
        "weight_dtype": "float32",
        "task": "token-classification",
        "parameter_count": sum(parameter.numel() for parameter in merged.parameters()),
        "encoder_parameter_count": sum(
            parameter.numel() for parameter in merged.model.parameters()
        ),
        "architectures": merged.config.architectures,
        "max_position_embeddings": merged.config.max_position_embeddings,
        "position_capacity_is_quality_evidence": False,
        "test_used_for_selection": False,
        "run_manifest_sha256": hashlib.sha256(
            args.run_manifest.read_bytes()
        ).hexdigest(),
        "merge_probe": {
            "rows": len(inputs["input_ids"]),
            "max_absolute_logit_difference": float((before - after).abs().max()),
            "atol": 2e-4,
            "rtol": 2e-4,
            "quality_evaluation": False,
            "reload_logits_bitwise_equal": True,
        },
    }
    (args.output / "merge_provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    evidence["files"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    (args.output / "candidate-lock.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(json.dumps(evidence["merge_probe"]), flush=True)


if __name__ == "__main__":
    main()
