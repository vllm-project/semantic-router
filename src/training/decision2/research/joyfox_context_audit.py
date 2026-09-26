"""Count native Joyfox row lengths on gold-free decision prompts.

The model is not loaded. This reports only aggregate token lengths and the
number of requests exceeding each context budget; no prompt text leaves the
experiment host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from inference.run import load_prompts


def audit(model_path: Path, prompts: Path) -> dict:
    from jev_inference.model import encode
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
    rows = load_prompts(prompts)
    lengths = []
    for row in rows:
        encoded = encode(
            tokenizer, {"state": row["state"], "questions": row["questions"]}, 1_000_000
        )
        segments = Counter(encoded["segments"])
        if 0 not in segments or len(segments) < 2:
            raise ValueError("Native row encoding lacks state or question segment")
        lengths.append(
            max(segments[0] + count for key, count in segments.items() if key)
        )
    ordered = sorted(lengths)
    return {
        "model_tokenizer": model_path.name,
        "prompt_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "items": len(rows),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "maximum": ordered[-1],
        "over_budget": {
            str(cap): sum(length > cap for length in lengths)
            for cap in (1024, 2048, 4096, 8192)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.model_path, args.prompts), sort_keys=True))


if __name__ == "__main__":
    main()
