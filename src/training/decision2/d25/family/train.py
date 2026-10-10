"""Vega's FSDP2 trainer for the family sizes (Qwen3.5-9B/4B/2B/0.8B).

    python -m d25.vega.train.launch --nproc 8 -- d25.family.train \
        --base-model Qwen/Qwen3.5-4B --base-revision <sha> [d25.vega.train.train arguments]

Two things differ from the 27B and are set here before the Vega trainer runs:

- Qwen3.5-0.8B, -2B and -4B tie the language-model head to the input embeddings, so their
  checkpoints have no ``lm_head.weight``; the readout starts from the embedding rows of the answer
  codes instead (the same vectors the tied head would use).
- The exported ``decision_config.json`` names the family base model and revision.

The vision tower is not trained here: exports carry the base's ``visual.*`` tensors unchanged, so
an export is evaluated on images by the Omni engine as it is.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from d25.vega.train import model as M

_read_checkpoint = M.read_checkpoint


def _tied(directory: Path) -> bool:
    config = json.loads((directory / "config.json").read_text())
    text = config.get("text_config") or {}
    return bool(config.get("tie_word_embeddings") or text.get("tie_word_embeddings"))


def read_checkpoint(directory, token_ids, parts=("text", "visual", "readout")):
    directory = Path(directory)
    no_head = (
        "readout" in parts
        and not (directory / "readout.safetensors").exists()
        and _tied(directory)
    )
    if not no_head:
        return _read_checkpoint(directory, token_ids, parts)
    loaded = _read_checkpoint(
        directory, token_ids, tuple(p for p in parts if p != "readout") + ("text",)
    )
    embed = loaded["text"]["embed_tokens.weight"]
    loaded["readout"] = embed[token_ids].float().clone()
    loaded["readout_source"] = "embed_tokens rows of the answer codes (tied lm_head)"
    if "text" not in parts:
        loaded["text"] = {}
    return loaded


def pop_option(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


def main() -> int:
    argv = list(sys.argv[1:])
    base_model = pop_option(argv, "--base-model")
    base_revision = pop_option(argv, "--base-revision")
    if not base_model or not base_revision:
        raise SystemExit("--base-model and --base-revision are required")
    M.BASE_MODEL, M.BASE_REVISION = base_model, base_revision
    M.read_checkpoint = read_checkpoint
    from d25.vega.train import train as T

    return T.main(argv)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
