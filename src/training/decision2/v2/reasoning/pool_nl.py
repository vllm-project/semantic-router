"""Natural-language problem pool for teacher graphs: reasoning-heavy rows of a released model's TRAIN files.

The rows were licence-audited and decontaminated when the released model's data was locked; they are reused
unchanged (same ids and input hashes), deduplicated by id across the given TRAIN files, filtered to the listed
sources, capped per source with a fixed seed, and written in a stable order.

usage: python3 -m v2.reasoning.pool_nl --train A.jsonl B.jsonl ... --out POOL.jsonl [--cap source=N ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCES = (
    "gsm8k_train",
    "mathqa_train",
    "decision2_verifiable_v2_a2",
    "hover_train",
    "hover_train_v1.1",
    "winogrande_xl_train",
    "dec10:multinli_nonfiction_train",
    "legacy:snli",
    "dec10:snli_train",
    "scitail_train",
    "csqa_train",
    "quartz_train",
    "ropes_train",
    "musique_full_v1.0_train",
    "twowiki_train",
    "hotpotqa_distractor_train",
    "balanced_copa_train",
)
DEFAULT_CAPS = {"mathqa_train": 5000}


def order_key(row_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", action="append", default=[], help="source=N")
    parser.add_argument("--seed", default="reasoning-pool-v1")
    args = parser.parse_args()
    caps = dict(DEFAULT_CAPS)
    for item in args.cap:
        source, _, value = item.partition("=")
        caps[source] = int(value)
    rows: dict[str, dict] = {}
    for path in args.train:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row["source"] in SOURCES and row["language"] == "en":
                rows.setdefault(row["id"], row)
    by_source: dict[str, list[dict]] = {}
    for row in rows.values():
        by_source.setdefault(row["source"], []).append(row)
    chosen = []
    for source in SOURCES:
        group = sorted(
            by_source.get(source, []), key=lambda r: order_key(r["id"], args.seed)
        )
        chosen.extend(group[: caps.get(source, len(group))])
    chosen.sort(key=lambda r: order_key(r["id"], args.seed))
    with args.out.open("w", encoding="utf-8") as sink:
        for row in chosen:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"rows": len(chosen), "by_source": Counter(r["source"] for r in chosen)},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
