"""Cache complete FP32 teacher outputs for explicitly marked replay TRAIN rows."""

# ruff: noqa: PLC0415

import argparse
import hashlib
import json
from pathlib import Path

from .data import read_records
from .model import load_model
from .optimization import initial_artifact_receipt, verify_full_checkpoint
from .retention import row_identity, tokenizer_file_receipt, write_retention_cache


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite retention targets")
    if args.max_length <= 0:
        raise ValueError("The teacher input budget must be positive")
    initialization = initial_artifact_receipt(args.base, None)
    model, tokenizer, labels, _ = load_model(args.base, args.contract)
    if model.config.problem_type != "single_label_classification":
        raise ValueError("Retention targets require single-label classification")
    verify_full_checkpoint(model, args.base)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("Teacher input budget exceeds checkpoint capacity")
    rows = read_records(args.train, labels)
    if any(row.get("split", "train") != "train" for row in rows):
        raise ValueError("Teacher targets must not use development or test rows")
    if any(not isinstance(row.get("retention_replay", False), bool) for row in rows):
        raise ValueError("retention_replay must be a boolean TRAIN annotation")
    replay = [row for row in rows if row.get("retention_replay", False)]
    if not replay:
        raise ValueError("Mark the old TRAIN rows with retention_replay: true")
    encoded = [
        tokenizer(row["text"], truncation=False, padding=False) for row in replay
    ]
    if any(len(value["input_ids"]) > args.max_length for value in encoded):
        raise ValueError(
            "A replay row exceeds the teacher budget; truncation is forbidden"
        )
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.to(args.device).eval()
    records = []
    with torch.inference_mode():
        for row, value in zip(replay, encoded, strict=True):
            batch = {
                name: torch.tensor([ids], device=args.device)
                for name, ids in value.items()
            }
            logits = model(**batch).logits
            if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
                raise ValueError("Teacher outputs must be finite native FP32 logits")
            records.append(
                {**row_identity(row, value), "logits": logits[0].cpu().tolist()}
            )
    manifest = write_retention_cache(
        args.output,
        initialization=initialization,
        contract_sha256=hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
        tokenizer_files=tokenizer_file_receipt(args.base),
        label_to_id=labels,
        records=records,
    )
    print(
        json.dumps(
            {"manifest": str(manifest), "rows": len(records), "dtype": "float32"}
        )
    )


if __name__ == "__main__":
    main()
