"""Are padded-batch gradients right? Compare against unpadded one-row passes.

For one official source with a fresh head, compute the summed CE gradient of a
few TRAIN rows (a) as one padded BF16 batch with the model's normal mask path,
(b) with a contiguous bool mask, (c) with a contiguous float mask, and (d) one
row at a time without padding. Reports cosine and relative norm of each
backbone gradient against (d). Gold enters only this local loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import encoder as enc
from . import kai8k
from .common import import_bundle, load_rights_clean, native_records, write_json
from .train import target_vector


def gradient(model: Any, packer: Any, records: list[dict[str, Any]], mode: str) -> Any:
    import torch

    model.zero_grad(set_to_none=True)
    encoded = [packer.encode(r) for r in records]
    batch = packer.collate(encoded, "cuda:0")
    if mode in ("bool", "float"):
        length = batch["attention_mask"].shape[1]
        keys = (
            batch["attention_mask"]
            .bool()[:, None, None, :]
            .expand(-1, 1, length, length)
            .contiguous()
        )
        if mode == "float":
            keys = torch.zeros(keys.shape, device=keys.device).masked_fill(
                ~keys, torch.finfo(torch.float32).min
            )
        original = model.bidirectional
        model.bidirectional = False
        batch = {**batch, "attention_mask": keys}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(batch)
    if mode in ("bool", "float"):
        model.bidirectional = original
    targets = torch.zeros(logits.shape, device=logits.device)
    for i, (record, e) in enumerate(zip(records, encoded)):
        targets[i, : len(e["candidate_ids"])] = torch.tensor(
            target_vector(record, e["candidate_ids"])
        )
    loss = -(
        targets * torch.log_softmax(logits.float(), -1) * batch["valid_candidates"]
    ).sum()
    loss.backward()
    return torch.cat(
        [
            p.grad.flatten().float()
            for p in model.backbone.parameters()
            if p.grad is not None
        ]
    )


def run(model: Any, packer: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    model.train()
    reference = None
    for record in records:
        g = gradient(model, packer, [record], "native")
        reference = g if reference is None else reference + g
    out = {}
    for mode in ("native", "bool", "float") if model.bidirectional else ("native",):
        g = gradient(model, packer, records, mode)
        out[mode] = {
            "cosine_vs_unpadded": float(
                torch.nn.functional.cosine_similarity(g, reference, dim=0)
            ),
            "norm_ratio_vs_unpadded": float(g.norm() / reference.norm()),
        }
    out["lengths"] = [packer.encode(r)["input_tokens"] for r in records]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--official", action="append", required=True, help="label=source=path"
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    kai8k.runtime_flags(torch)
    import_bundle(args.bundle)
    rows = [
        r
        for r in load_rights_clean(args.data_parent)["train"]
        if r["task_type"] == "choice"
    ]
    rows = sorted(rows, key=lambda r: r["id"])[:: max(1, len(rows) // args.rows)][
        : args.rows
    ]
    records = native_records(rows, args.bundle)
    results = {}
    for item in args.official:
        label, source, path = item.split("=", 2)
        model, packer, _ = enc.from_official(path, source)
        results[label] = run(model.to("cuda:0"), packer, records)
        del model
        torch.cuda.empty_cache()
    write_json(args.output, results)
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
