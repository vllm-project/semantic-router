"""Request sizes of a gold-free Index panel under a frozen MoE package's own prompt encoder (CPU).

    python3 -m v2.27b.moe.index_scan --checkpoint PKG/checkpoint --panel P/panel-N/panel.json \
        --out P/scan/NAME.json [--workers 32]

Each request is encoded exactly as ``infer.run_prompts`` does (``question_to_row`` and the
checkpoint's ``encoder_for`` at the package's limit, no truncation), and ``collate``'s batch shape
is recorded: questions x the longest prompt rounded up to 8 tokens ("padded tokens"), which drives
the native path's activation memory (one batch per request). Refused questions are counted, not
encoded further. The output (private: run IDs) lists every request with its padded tokens, sorted
largest first, and a histogram; nothing is answered.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing
from pathlib import Path
from typing import Any

from training.model import infer

_STATE: dict[str, Any] = {}


def _init(checkpoint: str, max_length: int) -> None:
    from transformers import AutoTokenizer

    from training.model.decision_model import encoder_for

    metadata = json.loads(
        (Path(checkpoint) / "decision_config.json").read_text(encoding="utf-8")
    )
    _STATE.update(
        tokenizer=AutoTokenizer.from_pretrained(checkpoint, local_files_only=True),
        encode=encoder_for(metadata),
        max_length=max_length,
    )


def measure(row: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": row["_evaluation"]["run_id"],
        "state": row["state"],
        "questions": row["questions"],
    }
    lengths, refused = [], 0
    for question_id, question in item["questions"].items():
        try:
            encoded = _STATE["encode"](
                infer.question_to_row(item, question_id, question),
                _STATE["tokenizer"],
                _STATE["max_length"],
            )
            lengths.append(len(encoded["ids"]))
        except ValueError:
            refused += 1
    longest = max(lengths) if lengths else 0
    return {
        "run_id": item["id"],
        "catalog_id": row["_evaluation"]["catalog_id"],
        "questions": len(lengths),
        "refused": refused,
        "max_tokens": longest,
        "tokens": sum(lengths),
        "padded_tokens": len(lengths) * math.ceil(longest / 8) * 8,
    }


def rows(panel: Path) -> list[dict[str, Any]]:
    manifest = json.loads(panel.read_text())
    out = []
    for shard in manifest["shards"]:
        with gzip.open(panel.parent / shard["file"], "rt", encoding="utf-8") as stream:
            out.extend(json.loads(line) for line in stream if line.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    panel_rows = rows(args.panel)
    with multiprocessing.Pool(
        args.workers, initializer=_init, initargs=(args.checkpoint, args.max_length)
    ) as pool:
        sizes = pool.map(measure, panel_rows, chunksize=64)
    sizes.sort(key=lambda s: -s["padded_tokens"])
    edges = [0, 8192, 32768, 65536, 131072, 196608, 262144, 327680, 393216, 524288]
    histogram = {
        f">={low}": sum(1 for s in sizes if s["padded_tokens"] >= low) for low in edges
    }
    report = {
        "schema": "decision2-27b-moe-index-scan/1",
        "panel_run_ids_sha256": json.loads(args.panel.read_text())["run_ids_sha256"],
        "rows": len(sizes),
        "max_length": args.max_length,
        "padded_tokens_at_least": histogram,
        "requests": sizes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    print(json.dumps({"rows": len(sizes), "padded_tokens_at_least": histogram}))


if __name__ == "__main__":
    main()
