"""Loaded and active parameters of a LoRA checkpoint on an official MoE base (host side, stdlib only).

Loaded = the base's text-decoder tensors the Decision model instantiates
(``v2.release.build.base_text_parameters``) + the LoRA adapter + the head, all from safetensors
headers (``v2.27b.m4b.ckpt_format params``). Active per token = loaded minus the routed expert
tensors (``.experts.`` inside the text decoder) plus top-k / num-experts of them: every other
text tensor, the adapter (its targets are attention, gated-delta and dense or shared-expert MLP
projections, never experts) and the head run for every token.

    python3 -m v2.27b.moe.moe_params --checkpoint CKPT --base BASE [--output OUT.json]
"""

from __future__ import annotations

import argparse
import importlib
import json
import struct
from pathlib import Path
from typing import Any

ckpt_format = importlib.import_module("v2.27b.m4b.ckpt_format")
TEXT_PREFIXES = ("model.language_model.", "model.layers.")


def _header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(size))


def routed_expert_parameters(base: Path, files: dict[str, str]) -> int:
    total = 0
    for name in sorted(files):
        if not name.endswith(".safetensors"):
            continue
        for tensor, meta in _header(base / name).items():
            if (
                tensor != "__metadata__"
                and tensor.startswith(TEXT_PREFIXES)
                and ".experts." in tensor
            ):
                count = 1
                for dim in meta["shape"]:
                    count *= dim
                total += count
    return total


def routing(base: Path) -> tuple[int, int]:
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    experts = text.get("num_experts")
    top_k = text.get("top_k_experts", text.get("num_experts_per_tok"))
    if not experts or not top_k:
        raise ValueError(f"{base}: no MoE routing in config.json")
    return int(experts), int(top_k)


def counts(checkpoint: Path, base: Path) -> dict[str, Any]:
    loaded, source = ckpt_format.params(checkpoint, ckpt_format.LORA, base)
    lora = json.loads(
        (checkpoint / "decision_config.json").read_text(encoding="utf-8")
    )["lora"]
    routed = routed_expert_parameters(base, lora["source_fingerprint"]["files_sha256"])
    experts, top_k = routing(base)
    if routed == 0 or routed % experts:
        raise ValueError(
            f"routed expert tensors ({routed}) do not split into {experts} experts"
        )
    active = loaded - routed + routed * top_k // experts
    return {
        "loaded_parameters": loaded,
        "active_parameters": active,
        "routed_expert_parameters": routed,
        "num_experts": experts,
        "top_k": top_k,
        "loaded_source": source,
        "active_source": (
            f"loaded {loaded:,} minus routed experts {routed:,} plus top-{top_k} of {experts} "
            f"({routed * top_k // experts:,}); safetensors headers"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = counts(args.checkpoint, args.base)
    text = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
