"""9B M9 retention probes: the 4B M10 probe panel minus every item that overlaps the 9B TRAIN (x60).

The M10 panel (MMLU validation + dev, ARC validation, a GSM8K train hold-out; native Choice questions) is already free
of every Decision Index suite row and of the 4B TRAIN by word 13-grams (``v2/dec/ops/m10/m10_probes.py``). M9 keeps
that construction and drops, in addition, every item whose stem has a word 13-gram (or, for a stem shorter than 13
words, the whole stem) in a state / instructions / options string of the 9B TRAIN file, with the M10 ``overlap``
routine unchanged. Each probe's stem is its prompt state (true for every M10 probe). Gold never leaves node A.

usage: m9_probes.py --prompts M10_PROMPTS --gold M10_GOLD --train X60 --work DIR
                    --out-prompts P --out-gold G --report R
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_m10_probes() -> Any:
    path = Path(__file__).resolve().parents[2] / "dec" / "ops" / "m10" / "m10_probes.py"
    spec = importlib.util.spec_from_file_location("m10_probes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out-prompts", type=Path, required=True)
    parser.add_argument("--out-gold", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    m10 = load_m10_probes()
    prompts = read_jsonl(args.prompts)
    gold = {row["id"]: row for row in read_jsonl(args.gold)}
    if [p["id"] for p in prompts] != list(gold):
        raise ValueError("M10 probe prompts and gold differ in ids or order")
    args.work.mkdir(parents=True, exist_ok=True)
    candidates = args.work / "m9-probe-candidates.jsonl"
    with candidates.open("x", encoding="utf-8") as stream:
        for p in prompts:
            row = {"id": p["id"], "probe": gold[p["id"]]["probe"], "stem": p["state"]}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    overlap = args.work / "overlap-x60.json"
    m10.overlap(
        argparse.Namespace(
            candidates=str(candidates),
            corpus=[str(args.train)],
            kind="train",
            output=str(overlap),
        )
    )
    hits = set(json.loads(overlap.read_text())["hit_ids"])
    kept = [p for p in prompts if p["id"] not in hits]
    with args.out_prompts.open("x", encoding="utf-8") as ps, args.out_gold.open(
        "x", encoding="utf-8"
    ) as gs:
        for p in kept:
            ps.write(json.dumps(p, ensure_ascii=False) + "\n")
            gs.write(json.dumps(gold[p["id"]]) + "\n")
    report = {
        "schema": "lux9b-m9-probes/1",
        "source_prompts_sha256": sha256(args.prompts),
        "source_gold_sha256": sha256(args.gold),
        "train_sha256": sha256(args.train),
        "overlap_sha256": sha256(overlap),
        "source": dict(Counter(gold[p["id"]]["probe"] for p in prompts)),
        "excluded_x60": dict(Counter(gold[i]["probe"] for i in hits)),
        "final": dict(Counter(gold[p["id"]]["probe"] for p in kept)),
        "prompts_sha256": sha256(args.out_prompts),
        "gold_sha256": sha256(args.out_gold),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
