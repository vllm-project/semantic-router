"""Experts-kernel numerical check against an FP32 reference (GPU, container; never trains).

The probe's random-head parity cannot separate a wrong kernel from BF16 rounding (untrained
logits are near-tied). This check compares the backbone's last hidden states at the readout
positions (every option endpoint and the query) of SELECT rows:

- reference: FP32 parameters, autocast off, ``eager`` experts;
- ``eager`` and ``grouped_mm`` under BF16 autocast (the training / inference precision).

Per row it records the relative L2 error ||h - h_ref|| / ||h_ref|| over the gathered vectors
and the cosine; the summary gives median / p95 / max per kernel and the grouped_mm / eager
ratio. Writes one JSON receipt (aggregates only).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from training.model.decision_model import DecisionModel, collate, encoder_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--source-stage", choices=("base", "posttrained"), required=True
    )
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.manual_seed(20260926)
    model, tokenizer = DecisionModel.from_base(
        args.model_path,
        args.revision,
        256,
        source_stage=args.source_stage,
        experts_implementation="eager",
    )
    model = model.float().to(device).eval()
    encode = encoder_for(model.metadata)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    with args.select.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()][: args.rows]
    items = [encode(row, tokenizer, args.max_length) for row in rows]

    def gathered(implementation: str, autocast: bool) -> list[torch.Tensor]:
        model.backbone.set_experts_implementation(implementation)
        out = []
        with torch.inference_mode():
            for item in items:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in collate([item], pad_id).items()
                }
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=autocast
                ):
                    hidden = model.backbone(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    ).last_hidden_state
                positions = [*item["candidate_positions"], item["query_position"]]
                out.append(hidden[0, positions].float().cpu())
        return out

    reference = gathered("eager", autocast=False)
    summary = {}
    for implementation in ("eager", "grouped_mm"):
        values = gathered(implementation, autocast=True)
        errors = [float((v - r).norm() / r.norm()) for v, r in zip(values, reference)]
        cosines = [
            float(
                torch.nn.functional.cosine_similarity(v.flatten(), r.flatten(), dim=0)
            )
            for v, r in zip(values, reference)
        ]
        ordered = sorted(errors)
        summary[implementation] = {
            "rows": len(errors),
            "relative_l2_median": statistics.median(errors),
            "relative_l2_p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "relative_l2_max": ordered[-1],
            "cosine_min": min(cosines),
        }
    summary["grouped_mm_over_eager_median"] = (
        summary["grouped_mm"]["relative_l2_median"]
        / summary["eager"]["relative_l2_median"]
    )
    summary["grouped_mm_over_eager_max"] = (
        summary["grouped_mm"]["relative_l2_max"] / summary["eager"]["relative_l2_max"]
    )
    result = {
        "schema_version": "decision2-27b-moe-experts-check/1",
        "model": args.model_path.name,
        "revision": args.revision,
        "architecture": model.metadata["architecture"],
        "reference": "FP32 parameters, autocast off, eager experts",
        "summary": summary,
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
