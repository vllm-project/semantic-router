"""Gold-free token and optimizer-step identity audit for a matched TRAIN arm.

Run this identical script once with the historical control module root and
once with the treatment module root on the same private data/tokenizer. It
prints aggregate hashes only; neither benchmark answers nor row text.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from training.model.data import canonical, file_sha256, load_partition
from training.model.decision_model import encode
from training.model.plan import epoch_batches, planned_updates


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--select", required=True)
    parser.add_argument("--cal", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--accumulation", type=int, default=16)
    args = parser.parse_args()
    if args.max_length != 8192 or args.accumulation != 16:
        raise ValueError("Matched 2B audit requires frozen context and accumulation")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    train = load_partition(args.train, "train")
    select = load_partition(args.select, "select")
    cal = load_partition(args.cal, "cal")
    train_items = [encode(row, tokenizer, args.max_length) for row in train]
    select_items = [encode(row, tokenizer, args.max_length) for row in select]
    batches = epoch_batches(
        [len(item["ids"]) for item in train_items],
        [],
        epoch=0,
        seed=args.seed,
        microbatch=1,
        replay_fraction=0.0,
    )
    windows = [
        batches[i : i + args.accumulation]
        for i in range(0, len(batches), args.accumulation)
    ]
    schedule = [
        {
            "input_sha256": sha(
                [train[index]["input_sha256"] for batch in window for _, index in batch]
            ),
            "tokens": sum(
                len(train_items[index]["ids"]) for batch in window for _, index in batch
            ),
        }
        for window in windows
    ]
    result = {
        "schema_version": "decision2-matched-inline-schedule-audit/1",
        "data_sha256": {
            "train": file_sha256(args.train),
            "select": file_sha256(args.select),
            "cal": file_sha256(args.cal),
        },
        "counts": {"train": len(train), "select": len(select), "cal": len(cal)},
        "train_native_sha256": sha(
            [
                (item["prompt_sha256"], item["token_ids_sha256"], len(item["ids"]))
                for item in train_items
            ]
        ),
        "select_native_sha256": sha(
            [
                (item["prompt_sha256"], item["token_ids_sha256"], len(item["ids"]))
                for item in select_items
            ]
        ),
        "schedule_sha256": sha(schedule),
        "planned_updates": planned_updates(
            len(train), 0, 0.0, 1, args.accumulation, 1, None
        ),
        "total_train_tokens": sum(len(item["ids"]) for item in train_items),
        "maximum_train_tokens": max(len(item["ids"]) for item in train_items),
        "maximum_select_tokens": max(len(item["ids"]) for item in select_items),
    }
    if len(windows) != result["planned_updates"]:
        raise ValueError("Step scheduler disagrees with planned updates")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
