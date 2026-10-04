"""Dev readout of a decoder-track checkpoint on select-split rows: per-row probabilities and per-family accuracy.

usage (in the training image): python3 -m v2.reasoning.devread --checkpoint DIR [--source-path BASE]
         --rows A.jsonl [B.jsonl ...] --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from training.model.data import load_partition
from training.model.decision_model import collate, encode

from ..dec.dec_model import dec_fingerprint, load_dec_checkpoint

TOKEN_BUDGET = 24_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    rows, files = [], []
    for path in args.rows:
        part = load_partition(path, "select")
        rows += part
        files += [path.name] * len(part)
    model, tokenizer = load_dec_checkpoint(args.checkpoint, args.source_path)
    device = torch.device("cuda:0")
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    items = [encode(row, tokenizer, args.max_length) for row in rows]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    batches, current = [], []
    for index in order:
        width = max(
            [len(items[i]["ids"]) for i in current] + [len(items[index]["ids"])]
        )
        if current and width * (len(current) + 1) > TOKEN_BUDGET:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    probs: dict[int, list[float]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_ids in batches:
            batch = collate([items[i] for i in batch_ids], pad_id)
            tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**tensors)
            p = torch.softmax(logits.float(), dim=-1).cpu()
            for j, i in enumerate(batch_ids):
                probs[i] = p[j, : len(items[i]["keys"])].tolist()
    per_family: dict[str, list[int]] = defaultdict(list)
    records = []
    per_file: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        p = probs[i]
        correct = int(max(range(len(p)), key=p.__getitem__) == row["label"])
        per_family[row["family"]].append(correct)
        per_file[files[i]][row["family"]].append(correct)
        records.append(
            {
                "id": row["id"],
                "file": files[i],
                "family": row["family"],
                "correct": correct,
                "p_gold": p[row["label"]],
            }
        )
    summary = {
        f: {"n": len(v), "acc": sum(v) / len(v)} for f, v in sorted(per_family.items())
    }
    macro = sum(s["acc"] for s in summary.values()) / len(summary)
    file_macro = {
        name: sum(sum(v) / len(v) for v in fams.values()) / len(fams)
        for name, fams in per_file.items()
    }
    out = {
        "checkpoint": str(args.checkpoint),
        "identity": dec_fingerprint(args.checkpoint, args.source_path)["model_sha256"],
        "rows": len(rows),
        "family_macro_accuracy": macro,
        "file_macro_accuracy": file_macro,
        "families": summary,
        "seconds": time.perf_counter() - started,
        "records": records,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=1))


if __name__ == "__main__":
    main()
