"""Weight soups of exported Vega checkpoints, averaged in FP32 (uniform or weighted).

Members must agree on codes, token ids, attention mode, pooling and base revision. Backbone
tensors are streamed one at a time (FP32 accumulate, BF16 store); the readout stays FP32. The
output is a normal "code-readout v1" export whose provenance lists the members and weights.

    python -m d25.vega.train.soup --ckpt A --ckpt B --ckpt C [--weights 0.5,0.25,0.25] --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from d25.vega.train import model as M

AGREE = (
    "format_id",
    "prompt",
    "base_model",
    "revision",
    "codes",
    "token_ids",
    "attention_mode",
    "pooling",
)


def build_soup(
    members: list[Path], weights: list[float] | None, out: Path, name: str | None = None
) -> dict:
    """Write the FP32 weighted average of ``members`` to ``out`` (atomic); returns a summary."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    began = time.time()
    members = [Path(p) for p in members]
    weights = list(weights) if weights else [1.0] * len(members)
    if len(weights) != len(members) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("need one non-negative weight per checkpoint")
    weights = [w / sum(weights) for w in weights]
    configs = [json.loads((m / "decision_config.json").read_text()) for m in members]
    for key in AGREE:
        if any(c.get(key) != configs[0].get(key) for c in configs[1:]):
            raise ValueError(f"members disagree on {key}")
    keymaps = []
    for member in members:
        keymap = {}
        for path in M._weight_files(member):
            handle = safe_open(str(path), framework="pt")
            for key in handle.keys():
                keymap[key] = handle
        keymaps.append(keymap)
    keys = sorted(keymaps[0])
    if any(sorted(k) != keys for k in keymaps[1:]):
        raise ValueError("members have different tensor names")
    text_prefix, visual_prefix = M._prefixes(keys)
    text, visual = {}, {}
    for key in keys:
        acc = None
        for weight, keymap in zip(weights, keymaps):
            tensor = keymap[key].get_tensor(key).float()
            acc = tensor * weight if acc is None else acc.add_(tensor, alpha=weight)
        if key.startswith(text_prefix):
            text[key[len(text_prefix) :]] = acc.to(torch.bfloat16)
        elif key.startswith(visual_prefix):
            visual[key[len(visual_prefix) :]] = acc.to(torch.bfloat16)
        else:
            raise ValueError(f"unexpected tensor {key}")
    readout = sum(
        w * load_file(str(m / "readout.safetensors"))["weight"].float()
        for w, m in zip(weights, members)
    )
    base = configs[0]
    provenance = {
        "run": name or Path(out).name,
        "soup": [
            {"path": str(m.resolve()), "weight": w, "provenance": c.get("provenance")}
            for m, w, c in zip(members, weights, configs)
        ],
        "method": "FP32 weighted average of backbone and readout tensors",
        "created_seconds": time.time() - began,
    }
    decision = M.decision_config(
        codes=base["codes"],
        token_ids=base["token_ids"],
        attention_mode=base["attention_mode"],
        max_length=base["max_length"],
        provenance=provenance,
    )
    decision.update(base_model=base["base_model"], revision=base["revision"])
    M.export_checkpoint(
        out,
        config=M.load_config(members[0]),
        text_state=text,
        visual_state=visual,
        readout=readout,
        tokenizer_source=members[0],
        decision_config=decision,
    )
    return {
        "out": str(out),
        "members": [str(m) for m in members],
        "weights": weights,
        "seconds": round(time.time() - began, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument(
        "--weights", help="Comma-separated, normalised to sum 1 (default uniform)"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    weights = [float(w) for w in args.weights.split(",")] if args.weights else None
    try:
        summary = build_soup([Path(p) for p in args.ckpt], weights, Path(args.out))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
