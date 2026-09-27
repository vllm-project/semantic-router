"""Compare a frozen own-Lux source with its fresh zero-update LoRA head.

This reads SELECT inputs but never their gold answers. It refuses an existing
output and writes the answer distributions only to a private runtime location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from training.model.data import file_sha256, load_partition
from training.model.decision_model import DecisionModel
from training.model.inline_teacher import (
    native_probabilities_batch,
    source_pairs_for_selection,
)
from training.model.lora import attach_lora
from training.model.source import source_fingerprint

SELECT_SHA = "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38"
SOURCE_REVISION = "bd45a30aee8c84032791c245c70f86dee5389cc8"
SCHEMA = "decision20-lux9b-short-replay-zero-lora/1"


def _key(probabilities: dict[str, float]) -> str | None:
    if not probabilities or any(not math.isfinite(p) for p in probabilities.values()):
        return None
    maximum = max(probabilities.values())
    winners = [key for key, value in probabilities.items() if value == maximum]
    return winners[0] if len(winners) == 1 else None


def compare(bare: list[dict], fresh: list[dict]) -> dict[str, float | int | str]:
    left = {row["id"]: row for row in bare}
    right = {row["id"]: row for row in fresh}
    if len(left) != 32 or left.keys() != right.keys():
        raise ValueError("Both zero-step outputs need the same 32 unique IDs")
    maximum = 0.0
    changes = 0
    for identifier, a in left.items():
        b = right[identifier]
        if (
            a["prompt_sha256"] != b["prompt_sha256"]
            or a["token_ids_sha256"] != b["token_ids_sha256"]
            or a["probabilities"].keys() != b["probabilities"].keys()
        ):
            raise ValueError("Source and fresh-LoRA native inputs differ")
        if _key(a["probabilities"]) is None or _key(b["probabilities"]) is None:
            raise ValueError("A zero-step answer is invalid")
        maximum = max(
            maximum,
            *(
                abs(a["probabilities"][key] - b["probabilities"][key])
                for key in a["probabilities"]
            ),
        )
        changes += _key(a["probabilities"]) != _key(b["probabilities"])
    return {
        "checked": 32,
        "categorical_changes": changes,
        "max_absolute_probability_drift": maximum,
        "status": "PASS" if changes == 0 and maximum <= 0.02 else "FAIL",
    }


def _collect(model: DecisionModel, tokenizer: object, rows: list[dict]) -> list[dict]:
    chosen = sorted(
        rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()
    )[:32]
    chosen_ids = {row["id"] for row in chosen}
    predictions = []
    with torch.inference_mode():
        for pair in source_pairs_for_selection(rows, chosen_ids):
            for row, (item, probabilities) in zip(
                pair, native_probabilities_batch(model, tokenizer, pair, 8192)
            ):
                if row["id"] in chosen_ids:
                    predictions.append(
                        {
                            "id": row["id"],
                            "prompt_sha256": item["prompt_sha256"],
                            "token_ids_sha256": item["token_ids_sha256"],
                            "probabilities": probabilities,
                        }
                    )
    if len(predictions) != 32:
        raise ValueError("Incomplete zero-step roster")
    return predictions


def run(source: Path, select: Path, output: Path) -> dict:
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("Output exists or its parent does not")
    if file_sha256(select) != SELECT_SHA:
        raise ValueError("Frozen SELECT600 changed")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16 GPU must be visible")
    torch.manual_seed(20260926)
    torch.cuda.manual_seed_all(20260926)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    rows = load_partition(select, "select")
    if len(rows) != 600:
        raise ValueError("Frozen SELECT count changed")
    identity = source_fingerprint(source)
    model, tokenizer = DecisionModel.from_decision1(source, 256)
    model = model.float().to(torch.device("cuda:0"))
    model.eval()
    model.backbone.config.use_cache = False
    bare = _collect(model, tokenizer, rows)
    attach_lora(
        model,
        rank=16,
        alpha=32,
        dropout=0.05,
        source_kind="decision1",
        source_fingerprint=identity,
    )
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.eval()
    fresh = _collect(model, tokenizer, rows)
    outcome = compare(bare, fresh)
    report = {
        "schema_version": SCHEMA,
        "source_repo_revision": SOURCE_REVISION,
        "source_files_sha256": identity["files_sha256"],
        "select_sha256": SELECT_SHA,
        "runner_sha256": file_sha256(Path(__file__)),
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
        },
        "elapsed_seconds": time.monotonic() - started,
        "outcome": outcome,
        "bare": bare,
        "fresh_lora": fresh,
    }
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source, args.select, args.output)
    print(json.dumps(report["outcome"], sort_keys=True))


if __name__ == "__main__":
    main()
