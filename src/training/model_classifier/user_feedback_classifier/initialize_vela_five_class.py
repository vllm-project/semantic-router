"""Extend a four-label adapter head while preserving all previously trained rows."""

# Torch and safetensors are optional for data-only tests.
# ruff: noqa: PLC0415

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .data_contract import ID2LABEL, LABEL2ID
from .vela_contract import VELA_ID2LABEL, VELA_LABEL2ID

LINEAR_WEIGHT_DIMENSIONS = 2


def extend_state(state, seed):
    import torch

    weight_key = "base_model.model.classifier.weight"
    bias_key = "base_model.model.classifier.bias"
    weight, bias = state[weight_key], state[bias_key]
    if (
        weight.ndim != LINEAR_WEIGHT_DIMENSIONS
        or weight.shape[0] != len(ID2LABEL)
        or bias.shape != (len(ID2LABEL),)
    ):
        raise ValueError("Expected a standard four-class sequence classifier")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    new_weight = (
        torch.randn((1, weight.shape[1]), generator=generator, dtype=weight.dtype)
        * 0.02
    )
    result = dict(state)
    result[weight_key] = torch.cat((weight, new_weight), dim=0)
    result[bias_key] = torch.cat((bias, torch.zeros(1, dtype=bias.dtype)), dim=0)
    return result


def main():
    from safetensors.torch import load_file, save_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an adapter")
    contract = json.loads(args.contract.read_text())
    if {
        int(key): value for key, value in contract["id2label"].items()
    } != ID2LABEL or contract["label2id"] != LABEL2ID:
        raise ValueError("Input must preserve the exact legacy four-label order")
    source_weights = args.adapter / "adapter_model.safetensors"
    initial_hash = hashlib.sha256(source_weights.read_bytes()).hexdigest()
    state = load_file(source_weights, device="cpu")
    extended = extend_state(state, args.seed)
    shutil.copytree(
        args.adapter,
        args.output,
        ignore=shutil.ignore_patterns(
            "adapter_model.safetensors", "README.md", "contract.json"
        ),
    )
    save_file(extended, args.output / "adapter_model.safetensors")
    contract.update(
        id2label=VELA_ID2LABEL,
        label2id=VELA_LABEL2ID,
        problem_type="single_label_classification",
    )
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    if hashlib.sha256(source_weights.read_bytes()).hexdigest() != initial_hash:
        raise ValueError("Source adapter changed during initialization; discard output")
    receipt = {
        "operation": "preserve encoder/LoRA/head and first four classifier rows; initialize NO_FEEDBACK row",
        "source_adapter_sha256": initial_hash,
        "source_contract_sha256": hashlib.sha256(
            args.contract.read_bytes()
        ).hexdigest(),
        "adapter_sha256": hashlib.sha256(
            (args.output / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "id2label": VELA_ID2LABEL,
        "requires_joint_training": True,
        "probability_note": "A fifth softmax term changes the original probabilities; this is not an equivalent checkpoint.",
    }
    (args.output / "initialization.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
