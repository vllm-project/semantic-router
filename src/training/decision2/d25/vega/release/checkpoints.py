"""Small code-readout v1 checkpoints for release-pipeline tests (staging repositories only).

``zero-shot`` turns a ``Qwen3_5ForConditionalGeneration`` checkpoint (e.g. Qwen3.5-0.8B-Base) into the
layout ws-train exports: ``Qwen3_5Model`` backbone (BF16, vision tower included), ``readout.safetensors`` =
the ``lm_head`` rows of the 255 answer codes (FP32), ``decision_config.json`` with the d25-vega prompt, and
the tokenizer. The probabilities are the base model's own zero-shot answers, so they are peaked enough for
argmax parity checks, and the package exercises exactly the code path of a trained 27B export.

    python -m d25.vega.release.checkpoints zero-shot --base <Qwen3.5 dir> --out <dir> \
        [--attention-mode causal|noncausal_full_attention] [--chat-template-from <dir with chat_template.jinja>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from d25.vega.common import decision_format as fmt

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def zero_shot(
    base: Path,
    out: Path,
    attention_mode: str,
    max_length: int,
    chat_template_from: Path | None,
    base_repo: str,
    base_revision: str | None,
) -> dict:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    copied = []
    for name in TOKENIZER_FILES:
        if (base / name).is_file():
            shutil.copyfile(base / name, partial / name)
            copied.append(name)
    tokenizer = AutoTokenizer.from_pretrained(str(partial))
    if not tokenizer.chat_template:
        if (
            chat_template_from is None
            or not (chat_template_from / "chat_template.jinja").is_file()
        ):
            raise ValueError(
                "the base has no chat template; pass --chat-template-from <dir with chat_template.jinja>"
            )
        shutil.copyfile(
            chat_template_from / "chat_template.jinja", partial / "chat_template.jinja"
        )
        copied.append("chat_template.jinja (from --chat-template-from)")
        tokenizer = AutoTokenizer.from_pretrained(str(partial))
    codes, token_ids = fmt.answer_codes(tokenizer)
    full = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(base), dtype=torch.bfloat16
    )
    readout = full.lm_head.weight.detach()[token_ids].float().clone()
    backbone = full.model
    backbone.config.architectures = ["Qwen3_5Model"]
    backbone.save_pretrained(
        str(partial), max_shard_size="5GB", safe_serialization=True
    )
    save_file({"weight": readout.contiguous()}, str(partial / "readout.safetensors"))
    decision = {
        "format_version": 1,
        "format_id": fmt.FORMAT_ID,
        "prompt": "d25-vega",
        "base_model": base_repo,
        "revision": base_revision,
        "codes": codes,
        "token_ids": token_ids,
        "temperature": 1.0,
        "attention_mode": attention_mode,
        "pooling": "last",
        "max_length": int(max_length),
        "readout_dtype": "float32",
        "provenance": {
            "kind": "release-pipeline test checkpoint",
            "readout": "lm_head rows of the answer codes",
            "base": base_repo,
            "trained": False,
        },
    }
    (partial / "decision_config.json").write_text(json.dumps(decision, indent=2) + "\n")
    partial.rename(out)
    return {
        "out": str(out),
        "codes": len(codes),
        "tokenizer_files": copied,
        "parameters": sum(p.numel() for p in backbone.parameters()) + readout.numel(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zero-shot")
    z.add_argument("--base", required=True, type=Path)
    z.add_argument("--out", required=True, type=Path)
    z.add_argument(
        "--attention-mode",
        default="causal",
        choices=("causal", "noncausal_full_attention"),
    )
    z.add_argument("--max-length", type=int, default=8192)
    z.add_argument("--chat-template-from", type=Path)
    z.add_argument("--base-repo", default="Qwen/Qwen3.5-0.8B-Base")
    z.add_argument("--base-revision")
    args = ap.parse_args(argv)
    print(
        json.dumps(
            zero_shot(
                args.base,
                args.out,
                args.attention_mode,
                args.max_length,
                args.chat_template_from,
                args.base_repo,
                args.base_revision,
            ),
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
