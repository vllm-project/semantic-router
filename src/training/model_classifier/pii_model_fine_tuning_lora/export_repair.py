"""Merge the selected PII adapter and verify FP32 short-input equivalence."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate import load_model


def main():
    import torch  # noqa: PLC0415 - optional dependency for data-only tooling

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-id", default="llm-semantic-router/mmbert-32k-yarn")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a merged checkpoint")
    run = json.loads(args.run_manifest.read_text())
    if run.get("test_used") is not False:
        raise ValueError("Candidate provenance must explicitly exclude test selection")
    if run.get("base_revision") != args.base_revision:
        raise ValueError("Export base revision differs from the training receipt")
    torch.set_num_threads(8)
    model, tokenizer, label_to_id, _ = load_model(args.base, args.config, args.adapter)
    model.eval()
    # These are numerical probes, not quality evaluation or training examples.
    inputs = tokenizer(
        ["A short numerical equivalence probe.", "用于检查合并前后数值一致性。"],
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        before = model(**inputs).logits
        merged = model.merge_and_unload(safe_merge=True)
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
    labels = {
        "label_to_idx": label_to_id,
        "idx_to_label": {str(index): label for label, index in label_to_id.items()},
    }
    for name in ("label_mapping.json", "pii_mapping.json"):
        (args.output / name).write_text(json.dumps(labels, indent=2) + "\n")
    evidence = {
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "adapter_sha256": hashlib.sha256(
            (args.adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
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
