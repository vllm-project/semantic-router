"""Initialize a fresh task adapter and trainable head from a fixed encoder base."""

# ruff: noqa: PLC0415
import argparse
import hashlib
import json
from pathlib import Path

from .model import load_model, save_adapter


def main():
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-id", default="llm-semantic-router/mmbert-32k-yarn")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an initialized adapter")
    if min(args.rank, args.alpha) <= 0 or not 0 <= args.dropout < 1:
        raise ValueError("Invalid LoRA dimensions or dropout")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    model, tokenizer, _labels, _ids = load_model(args.base, args.contract)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"],
            modules_to_save=["head", "classifier"],
            bias="none",
            revision=args.base_revision,
        ),
    )
    save_adapter(model, tokenizer, args.output, args.base_id, args.base_revision)
    (args.output / "contract.json").write_bytes(args.contract.read_bytes())
    receipt = {
        "base_model": args.base_id,
        "base_revision": args.base_revision,
        "seed": args.seed,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "classifier_pooling": model.config.classifier_pooling,
        "adapter_sha256": hashlib.sha256(
            (args.output / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    (args.output / "initialization.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
