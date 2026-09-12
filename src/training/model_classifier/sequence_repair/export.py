"""Freeze a selected adapter as a standard HF merged sequence checkpoint."""

# ruff: noqa: PLC0415

import argparse
import hashlib
import json
from pathlib import Path

from .model import load_model


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-id", default="llm-semantic-router/mmbert-32k-yarn")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a frozen candidate")
    run = json.loads(args.run_manifest.read_text())
    if run.get("test_used") is not False:
        raise ValueError("Candidate provenance must explicitly exclude test selection")
    if run.get("base_revision") != args.base_revision:
        raise ValueError("Export base revision differs from the training receipt")
    torch.set_num_threads(8)
    model, tokenizer, labels, _ = load_model(args.base, args.contract, args.adapter)
    model.eval()
    tokens = tokenizer(
        ["Numerical merge equivalence probe.", "用于核对合并数值的输入。"],
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        before = model(**tokens).logits
        merged = model.merge_and_unload(safe_merge=True)
        after = merged(**tokens).logits
    torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)
    if type(merged).__name__ != "ModernBertForSequenceClassification":
        raise ValueError(
            "Export must preserve the standard supported HF sequence architecture"
        )
    if merged.config.label2id != labels:
        raise ValueError("Merge changed label IDs")
    merged.config._name_or_path = args.base_id
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    (args.output / "label_mapping.json").write_text(
        json.dumps({"label2id": labels, "id2label": merged.config.id2label}, indent=2)
        + "\n"
    )
    receipt = {
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "adapter_sha256": hashlib.sha256(
            (args.adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(
            args.run_manifest.read_bytes()
        ).hexdigest(),
        "parameter_count": sum(parameter.numel() for parameter in merged.parameters()),
        "encoder_parameter_count": sum(
            parameter.numel() for parameter in merged.model.parameters()
        ),
        "label2id": labels,
        "task": merged.config.problem_type,
        "classifier_pooling": merged.config.classifier_pooling,
        "max_position_embeddings": merged.config.max_position_embeddings,
        "position_capacity_is_quality_evidence": False,
        "weight_dtype": "float32",
        "test_used_for_selection": False,
        "merge_probe_max_absolute_difference": float((before - after).abs().max()),
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output.iterdir())
            if path.is_file()
        },
    }
    (args.output / "candidate-lock.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in [
                    "parameter_count",
                    "merge_probe_max_absolute_difference",
                    "adapter_sha256",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
