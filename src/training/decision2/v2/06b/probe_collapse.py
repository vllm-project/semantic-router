"""Do a model's candidate vectors collapse to one direction within a request?

For fixed SELECT Choice rows, compare the head's normalized candidate inputs
(marker state and span mean) across the candidates of each row. If every
candidate maps to the same vector, the head gradient sum_k (p_k - y_k) f(c_k)
vanishes and training stalls at chance. Gold is never read.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from . import encoder as enc
from . import kai8k
from .common import import_bundle, load_rights_clean, native_records, write_json


def cosine_stats(vectors: Any) -> float:
    import torch

    normed = torch.nn.functional.normalize(vectors.float(), dim=-1)
    pairs = list(itertools.combinations(range(len(normed)), 2))
    return float(sum(float(normed[i] @ normed[j]) for i, j in pairs) / len(pairs))


def probe(model: Any, packer: Any, records: list[dict[str, Any]]) -> dict[str, float]:
    import torch

    marker, span, spread = [], [], []
    model.eval()
    with torch.inference_mode():
        for record in records:
            batch = packer.collate([packer.encode(record)], "cuda:0")
            mask = batch["attention_mask"]
            if model.bidirectional:
                length = mask.shape[1]
                mask = mask.bool()[:, None, None, :].expand(-1, 1, length, length)
            hidden = model.backbone(
                input_ids=batch["input_ids"], attention_mask=mask, return_dict=True
            ).last_hidden_state
            m = model.head.candidate_norm(
                enc.pool_candidates(hidden, batch, "marker")[0].float()
            )
            s = model.head.candidate_norm(
                enc.pool_candidates(hidden, batch, "span-mean")[0].float()
            )
            marker.append(cosine_stats(m))
            span.append(cosine_stats(s))
            spread.append(float(torch.softmax(model(batch)[0], -1).std()))
    return {
        "rows": len(records),
        "marker_mean_pairwise_cosine": sum(marker) / len(marker),
        "span_mean_pairwise_cosine": sum(span) / len(span),
        "output_probability_std_mean": sum(spread) / len(spread),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--model", action="append", default=[], help="label=export_dir=manifest_sha256"
    )
    parser.add_argument(
        "--official", action="append", default=[], help="label=source=path (zero-step)"
    )
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    kai8k.runtime_flags(torch)
    import_bundle(args.bundle)
    rows = [
        r
        for r in load_rights_clean(args.data_parent)["select"]
        if r["task_type"] == "choice"
    ][: args.rows]
    records = native_records(rows, args.bundle)
    results = {}
    for item in args.official:
        label, source, path = item.split("=", 2)
        model, packer, _ = enc.from_official(path, source)
        results[label] = probe(model.to("cuda:0"), packer, records)
        del model
        torch.cuda.empty_cache()
    for item in args.model:
        label, path, manifest = item.split("=", 2)
        model, packer, _ = enc.load(path, manifest)
        results[label] = probe(model, packer, records)
        del model
        torch.cuda.empty_cache()
    write_json(args.output, results)
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
