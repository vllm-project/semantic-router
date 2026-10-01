"""Decoder M10: materialize one LoRA seed checkpoint (head or label-token readout) as a full FP32 checkpoint.

The adapter is merged into its pinned source (``merge_and_unload(safe_merge=True)`` in FP32), so the soup of merged
seeds equals the base plus the mean adapter update. Before and after the merge the first ``--check-rows`` SELECT700
rows are scored on the GPU; their argmax agreement and maximum probability drift are written to
``<output>/merge_check.json`` (reported; BF16-autocast rounding differs between the two forms).

usage: m10_merge.py --checkpoint CK --source-path SRC --select SELECT --output OUT [--check-rows 128]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.model.data import load_partition
from training.model.decision_model import encode
from v2.dec.dec_model import dec_fingerprint, load_dec_checkpoint
from v2.dec.label_token import LabelTokenModel, encode_label
from v2.dec.preflight_dec import compare, select_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-rows", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = dec_fingerprint(args.checkpoint, args.source_path)
    model, tokenizer = load_dec_checkpoint(args.checkpoint, args.source_path)
    if model.metadata.get("checkpoint_format") != "peft-lora/1":
        raise ValueError("only LoRA checkpoints are merged")
    label = isinstance(model, LabelTokenModel)
    encode_fn = encode_label if label else encode
    rows = load_partition(args.select, "select")[: args.check_rows]
    model = model.float().to(torch.device("cuda:0")).eval()
    before = select_probabilities(model, tokenizer, rows, 2, 8448, encode_fn)
    source_kind = model.metadata["lora"]["source_kind"]
    if label:
        model.merge_lora()
    else:
        model.merge_lora(identity)
        model.metadata["full_training_source"] = {
            "kind": f"merged-lora-{source_kind}",
            "base_revision": model.metadata["lora_origin"]["configuration"].get(
                "base_revision"
            ),
            "source_fingerprint": model.metadata["lora_origin"]["source"],
        }
    model.metadata["merged_from"] = {
        "checkpoint": str(args.checkpoint),
        "model_sha256": identity["model_sha256"],
    }
    after = select_probabilities(model, tokenizer, rows, 2, 8448, encode_fn)
    check = compare(before, after)
    model = model.to(torch.device("cpu"))
    model.save(args.output, tokenizer)
    merged = dec_fingerprint(args.output, None)["model_sha256"]
    report = {
        "checkpoint": str(args.checkpoint),
        "adapter_identity": identity["model_sha256"],
        "merged_identity": merged,
        "readout": "label_token" if label else "head",
        "select_rows_checked": check,
    }
    (args.output / "merge_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
