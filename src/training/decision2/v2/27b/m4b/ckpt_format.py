"""Checkpoint format check and loaded-parameter count for the M4b readout / formal drivers (host side, stdlib only).

``full`` is a full DecisionModel checkpoint (``backbone/`` + head; M4b's and M5's full fine-tunes and FP32 soups);
``peft-lora/1`` a LoRA checkpoint (``adapter/`` + head on the pinned base; M5's L128 soup). The drivers name the
format they expect (``CHECKPOINT_FORMAT``, default ``full``) and refuse any other. Parameter counts come from the
safetensors headers; the source strings are the ones run_formal.sh (full) and run_finalist.sh (LoRA) report.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

FULL = "full"
LORA = "peft-lora/1"
FORMATS = (FULL, LORA)


def config_format(checkpoint: Path) -> str:
    config = json.loads(
        (checkpoint / "decision_config.json").read_text(encoding="utf-8")
    )
    return config.get("checkpoint_format", FULL)


def check(checkpoint: Path, expected: str) -> str:
    """Raise unless ``checkpoint`` is a DecisionModel checkpoint of format ``expected``."""
    if expected not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, not {expected!r}")
    found = config_format(checkpoint)
    body = (
        checkpoint / "backbone"
        if expected == FULL
        else checkpoint / "adapter/adapter_model.safetensors"
    )
    if found != expected or not body.exists():
        raise SystemExit(
            f"{checkpoint} is not a {expected} DecisionModel checkpoint (checkpoint_format {found})"
        )
    return found


def count(path: Path) -> int:
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(size))
    total = 0
    for name, meta in header.items():
        if name != "__metadata__":
            n = 1
            for dim in meta["shape"]:
                n *= dim
            total += n
    return total


def params(
    checkpoint: Path, expected: str, base: Path | None = None
) -> tuple[int, str]:
    """(loaded parameters, parameter source) of a checkpoint of format ``expected``."""
    check(checkpoint, expected)
    head = count(checkpoint / "decision_head.safetensors")
    if expected == FULL:
        index = json.loads(
            (checkpoint / "backbone/model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        backbone = sum(
            count(checkpoint / "backbone" / f)
            for f in sorted(set(index["weight_map"].values()))
        )
        return backbone + head, (
            f"full FP32 checkpoint: text backbone {backbone:,} + head {head:,} (safetensors headers)"
        )
    if base is None:
        raise ValueError("a LoRA checkpoint's count needs the pinned base (--base)")
    from v2.release.build import base_text_parameters

    lora = json.loads(
        (checkpoint / "decision_config.json").read_text(encoding="utf-8")
    )["lora"]
    backbone = base_text_parameters(
        base, lora["source_fingerprint"]["files_sha256"], lora["source_kind"]
    )
    adapter = count(checkpoint / "adapter/adapter_model.safetensors")
    return backbone + adapter + head, (
        f"pinned base text backbone {backbone:,} + LoRA rank {lora['rank']} {adapter:,} + head {head:,} "
        "(safetensors headers)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "params"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--format", choices=FORMATS, required=True)
        if name == "params":
            p.add_argument("--base", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        check(args.checkpoint, args.format)
    else:
        loaded, source = params(args.checkpoint, args.format, args.base)
        print(f"{loaded}\t{source}")


if __name__ == "__main__":
    main()
