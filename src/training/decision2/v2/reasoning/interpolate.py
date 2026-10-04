"""Weight interpolation of a fine-tuned full checkpoint towards its start: theta = start + alpha (tuned - start).

Tensors are matched by name, not by shard (the trainer and the release soups shard the same 426 backbone tensors
differently); the output keeps the tuned checkpoint's layout, tokenizer and config. FP32 throughout.

usage: python3 -m v2.reasoning.interpolate --tuned DIR --start DIR --alpha A --output DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..dec.dec_model import dec_fingerprint


def tensors(root: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for shard in sorted((root / "backbone").glob("*.safetensors")):
        out.update(load_file(str(shard)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.alpha <= 1:
        parser.error("alpha must be in (0, 1]")
    if args.output.exists():
        raise FileExistsError(args.output)
    start = tensors(args.start)
    tuned_shards = sorted((args.tuned / "backbone").glob("*.safetensors"))
    names = {name for shard in tuned_shards for name in load_file(str(shard))}
    if names != set(start):
        raise ValueError("tuned and start checkpoints hold different backbone tensors")
    pending = args.output.with_name(args.output.name + ".pending")
    shutil.copytree(args.tuned, pending, ignore=shutil.ignore_patterns("*.safetensors"))
    for shard in tuned_shards:
        tuned = load_file(str(shard))
        mixed = {
            name: (
                start[name].float() + args.alpha * (t.float() - start[name].float())
            ).contiguous()
            for name, t in tuned.items()
        }
        save_file(
            mixed, str(pending / "backbone" / shard.name), metadata={"format": "pt"}
        )
    head_t = load_file(str(args.tuned / "decision_head.safetensors"))
    head_s = load_file(str(args.start / "decision_head.safetensors"))
    if set(head_t) != set(head_s):
        raise ValueError("decision heads differ in tensors")
    save_file(
        {
            k: (
                head_s[k].float() + args.alpha * (v.float() - head_s[k].float())
            ).contiguous()
            for k, v in head_t.items()
        },
        str(pending / "decision_head.safetensors"),
    )
    meta = json.loads((pending / "decision_config.json").read_text())
    meta["interpolation"] = {
        "alpha": args.alpha,
        "tuned_model_sha256": dec_fingerprint(args.tuned, None)["model_sha256"],
        "start_model_sha256": dec_fingerprint(args.start, None)["model_sha256"],
    }
    (pending / "decision_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    pending.rename(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "alpha": args.alpha,
                "model_sha256": dec_fingerprint(args.output, None)["model_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
