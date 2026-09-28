"""Uniform weight average ("soup") of same-recipe, same-init full checkpoints.

Every member must be a full decoder-track checkpoint with the same
architecture, prompt, head dimension and initialization source; tensors are
averaged in FP32 with equal weights. The soup keeps the first member's
tokenizer and backbone config and records every member's inference identity.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from training.model.data import file_sha256

from .dec_model import dec_fingerprint

SAME = (
    "architecture",
    "prompt_version",
    "head_dim",
    "checkpoint_format",
    "max_options",
)


def shard_tensors(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    return {
        shard.name: load_file(str(shard))
        for shard in sorted((path / "backbone").glob("*.safetensors"))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.member) < 2:
        parser.error("a soup needs at least two members")
    if args.output.exists():
        raise FileExistsError(args.output)
    metas = [json.loads((m / "decision_config.json").read_text()) for m in args.member]
    for key in SAME:
        if len({json.dumps(m.get(key)) for m in metas}) != 1:
            raise ValueError(f"members differ in {key}")
    if metas[0].get("checkpoint_format") != "full":
        raise ValueError("soups are defined for full checkpoints")
    sources = {json.dumps(m.get("full_training_source"), sort_keys=True) for m in metas}
    if len(sources) != 1:
        raise ValueError("members have different initialization sources")
    shards = [shard_tensors(m) for m in args.member]
    if len({json.dumps({s: sorted(t) for s, t in x.items()}) for x in shards}) != 1:
        raise ValueError("members have different backbone tensor layouts")
    heads = [load_file(str(m / "decision_head.safetensors")) for m in args.member]
    weight = 1.0 / len(args.member)
    pending = args.output.with_name(args.output.name + ".pending")
    shutil.copytree(
        args.member[0], pending, ignore=shutil.ignore_patterns("*.safetensors")
    )
    for name in shards[0]:
        averaged = {
            key: sum(
                member[name][key].float() * weight for member in shards
            ).contiguous()
            for key in shards[0][name]
        }
        save_file(averaged, str(pending / "backbone" / name), metadata={"format": "pt"})
    save_file(
        {k: sum(h[k].float() * weight for h in heads).contiguous() for k in heads[0]},
        str(pending / "decision_head.safetensors"),
    )
    identities = [dec_fingerprint(m, None)["model_sha256"] for m in args.member]
    meta = dict(metas[0])
    meta["soup"] = {
        "method": "uniform weight average (FP32)",
        "members": [
            {"path": str(m), "model_sha256": ident}
            for m, ident in zip(args.member, identities)
        ],
    }
    meta.pop("initialization", None)
    meta["initialization"] = "uniform-soup-of-full-checkpoints"
    (pending / "decision_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for stale in ("checkpoint.json",):
        if (pending / stale).exists():
            (pending / stale).unlink()
    pending.rename(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "members": identities,
                "model_sha256": dec_fingerprint(args.output, None)["model_sha256"],
                "head_sha256": file_sha256(args.output / "decision_head.safetensors"),
            }
        )
    )


if __name__ == "__main__":
    main()
