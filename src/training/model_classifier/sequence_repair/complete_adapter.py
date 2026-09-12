"""Make a legacy sequence adapter self-contained with its original task head."""

# ruff: noqa: PLC0415
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .model import load_model
from .task_head import verify_saved_task_head


def main():
    import torch
    from safetensors.torch import load_file, save_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a completed adapter")
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    if "head" in (config.get("modules_to_save") or []):
        raise ValueError("Adapter already owns its task head")
    torch.set_num_threads(8)
    model, tokenizer, _labels, _ids = load_model(args.base, args.contract, args.adapter)
    model.eval()
    tokens = tokenizer(
        ["A task head parity probe.", "核对分类头参数是否保持不变。"],
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        before = model(**tokens).logits
    head = model.get_base_model().head.state_dict()
    weights = load_file(args.adapter / "adapter_model.safetensors", device="cpu")
    for name, tensor in head.items():
        key = "base_model.model.head." + name
        if key in weights:
            raise ValueError("Unexpected existing head tensor in legacy adapter")
        weights[key] = tensor.detach().cpu().contiguous()
    args.output.mkdir(parents=True)
    for path in args.adapter.iterdir():
        if path.is_file() and path.name not in {
            "adapter_config.json",
            "adapter_model.safetensors",
            "task-head.json",
        }:
            shutil.copy2(path, args.output / path.name)
    config["modules_to_save"] = [*(config.get("modules_to_save") or []), "head"]
    (args.output / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    save_file(
        weights, args.output / "adapter_model.safetensors", metadata={"format": "pt"}
    )
    restored, _tokenizer, _labels, _ids = load_model(
        args.base, args.contract, args.output
    )
    restored.eval()
    with torch.inference_mode():
        after = restored(**tokens).logits
    if not bool(torch.isfinite(before).all() and torch.isfinite(after).all()):
        raise ValueError("Task-head completion produced non-finite logits")
    torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-6)
    scope = verify_saved_task_head(restored, args.output / "adapter_model.safetensors")
    (args.output / "task-head.json").write_text(json.dumps(scope, indent=2) + "\n")
    receipt = {
        "operation": "Preserve the existing sequence task head in a legacy adapter; no training or encoder change",
        "source_adapter_sha256": hashlib.sha256(
            (args.adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "completed_adapter_sha256": hashlib.sha256(
            (args.output / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "base_model": config["base_model_name_or_path"],
        "base_revision": config.get("revision"),
        "head_tensors": list(head),
        "task_head": scope,
        "maximum_absolute_logit_difference": float((before - after).abs().max()),
        "quality_evaluation": False,
    }
    (args.output / "task-head-completion.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
