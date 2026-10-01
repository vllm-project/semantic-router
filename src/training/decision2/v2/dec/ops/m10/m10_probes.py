"""Decoder M10 knowledge / maths retention probes (diagnostic panels; never training data, never Index rows).

Sources are public non-test splits at pinned revisions: MMLU ``validation`` + ``dev`` (cais/mmlu ``all``), ARC
``validation`` (allenai/ai2_arc, Challenge and Easy) and a GSM8K ``train`` hold-out (openai/gsm8k ``main``). Every item
is a native Choice question (keys A, B, ...; gold = the correct option):

* MMLU / ARC: state = the question, instructions name the subject (MMLU) or the science question (ARC);
* GSM8K: state = the problem; four numeric options = the final answer plus three distractors taken, in seeded
  order, from the solution's intermediate results, then the problem's own numbers, then fixed perturbations.

Exclusion (word 13-grams after NFKC + casefold + ``\\w+``; a stem shorter than 13 words must appear whole):
``overlap --kind train`` against the M10 TRAIN file (any string in state / instructions / options) and
``overlap --kind suite`` against the Decision Index suite rows (run where the suite lives; only per-candidate hit
counts leave that node). ``finalize`` drops every candidate with a hit in either check and caps GSM8K by a seeded
sample. Subcommands:

    candidates --mmlu-dir D --arc-dir D --gsm8k-dir D --output candidates.jsonl     (needs pyarrow)
    overlap --candidates C --corpus F [--corpus F ...] --kind train|suite --output O (standard library only)
    finalize --candidates C --exclude O [--exclude O ...] --gsm8k-max N --prompts P --gold G --report R
    score --gold G --predictions P [--reference P] --output O [--draws N]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

NGRAM = 13
WORD = re.compile(r"\w+")
SEED = "dec-m10-probes-v1"
GSM8K_INSTRUCTIONS = "Which option is the correct final answer to the math problem?"
ARC_INSTRUCTIONS = "Which option correctly answers the science question?"
LETTERS = "ABCDEFGH"


def tokens(text: str) -> list[str]:
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def gram_hash(words: list[str]) -> int:
    return int.from_bytes(
        hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest(), "big"
    )


def strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4g}"


def parse_number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def gsm8k_options(
    question: str, solution: str, rng: random.Random
) -> tuple[list[str], int]:
    gold_text = solution.rsplit("####", 1)[1].strip()
    gold = parse_number(gold_text)
    if gold is None:
        raise ValueError("non-numeric GSM8K answer")
    pool: list[float] = []
    for _, value in re.findall(r"<<([^=<>]*)=([^<>]*)>>", solution):
        number = parse_number(value)
        if number is not None:
            pool.append(number)
    rng.shuffle(pool)
    asked = [parse_number(m) for m in re.findall(r"\d[\d,]*\.?\d*", question)]
    asked = [n for n in asked if n is not None]
    rng.shuffle(asked)
    perturb = [
        gold + 1,
        gold - 1,
        gold * 2,
        gold / 2,
        gold + 10,
        gold - 10,
        gold * 10,
        gold + 2,
    ]
    chosen: list[float] = []
    for number in [*pool, *asked, *perturb]:
        if number == gold or number in chosen or number < 0:
            continue
        if not float(number).is_integer() and float(gold).is_integer():
            continue
        chosen.append(number)
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        raise ValueError("too few GSM8K distractors")
    values = [gold, *chosen]
    order = list(range(4))
    rng.shuffle(order)
    return [number_text(values[i]) for i in order], order.index(0)


def candidates(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    out: list[dict[str, Any]] = []
    for split in ("validation", "dev"):
        table = pq.read_table(
            next(Path(args.mmlu_dir).glob(f"{split}-*.parquet"))
        ).to_pylist()
        for index, row in enumerate(table):
            subject = row["subject"].replace("_", " ")
            out.append(
                {
                    "id": f"mmlu-{split}-{index:05d}",
                    "probe": "mmlu",
                    "stem": row["question"],
                    "state": row["question"],
                    "instructions": f"This is a question about {subject}. Which option correctly answers it?",
                    "options": list(row["choices"]),
                    "gold": int(row["answer"]),
                }
            )
    for name in ("ARC-Challenge", "ARC-Easy"):
        table = pq.read_table(
            next((Path(args.arc_dir) / name).glob("validation-*.parquet"))
        ).to_pylist()
        for row in table:
            labels = list(row["choices"]["label"])
            out.append(
                {
                    "id": f"arc-{name.split('-')[1].lower()}-{row['id']}",
                    "probe": "arc-" + name.split("-")[1].lower(),
                    "stem": row["question"],
                    "state": row["question"],
                    "instructions": ARC_INSTRUCTIONS,
                    "options": list(row["choices"]["text"]),
                    "gold": labels.index(row["answerKey"]),
                }
            )
    table = pq.read_table(
        next(Path(args.gsm8k_dir).glob("train-*.parquet"))
    ).to_pylist()
    for index, row in enumerate(table):
        rng = random.Random(f"{SEED}/gsm8k/{index}")
        try:
            options, gold = gsm8k_options(row["question"], row["answer"], rng)
        except ValueError:
            continue
        out.append(
            {
                "id": f"gsm8k-train-{index:05d}",
                "probe": "gsm8k",
                "stem": row["question"],
                "state": row["question"],
                "instructions": GSM8K_INSTRUCTIONS,
                "options": options,
                "gold": gold,
            }
        )
    with Path(args.output).open("x", encoding="utf-8") as stream:
        for row in out:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(dict(Counter(row["probe"] for row in out))))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def overlap(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(Path(args.candidates)))
    grams: dict[int, set[str]] = defaultdict(set)
    short: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        words = tokens(row["stem"])
        if len(words) >= NGRAM:
            for i in range(len(words) - NGRAM + 1):
                grams[gram_hash(words[i : i + NGRAM])].add(row["id"])
        elif words:
            short[len(words)][gram_hash(words)].add(row["id"])
    hits: Counter[str] = Counter()
    scanned = 0
    for corpus in args.corpus:
        for record in read_jsonl(Path(corpus)):
            scanned += 1
            if args.kind == "train":
                payload = [
                    record.get("state"),
                    record.get("instructions"),
                    record.get("options"),
                ]
            else:
                payload = [
                    v for k, v in record.items() if k not in ("id", "_evaluation")
                ]
            matched: set[str] = set()
            for text in strings(payload):
                words = tokens(text)
                for i in range(len(words) - NGRAM + 1):
                    found = grams.get(gram_hash(words[i : i + NGRAM]))
                    if found:
                        matched |= found
                for length, table in short.items():
                    for i in range(len(words) - length + 1):
                        found = table.get(gram_hash(words[i : i + length]))
                        if found:
                            matched |= found
            for item in matched:
                hits[item] += 1
    report = {
        "kind": args.kind,
        "corpus_sha256": {
            Path(c).name: hashlib.sha256(Path(c).read_bytes()).hexdigest()
            for c in args.corpus
        },
        "candidates": len(rows),
        "corpus_records": scanned,
        "candidates_hit": len(hits),
        "hits_by_probe": dict(
            Counter(r["probe"] for r in rows if r["id"] in hits).most_common()
        ),
        "hit_ids": sorted(hits),
    }
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "hit_ids"}))


def finalize(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(Path(args.candidates)))
    excluded: set[str] = set()
    reports = []
    for path in args.exclude:
        report = json.loads(Path(path).read_text())
        excluded |= set(report["hit_ids"])
        reports.append(
            {
                "kind": report["kind"],
                "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "candidates_hit": report["candidates_hit"],
            }
        )
    kept = [row for row in rows if row["id"] not in excluded]
    gsm = [row for row in kept if row["probe"] == "gsm8k"]
    random.Random(f"{SEED}/gsm8k-sample").shuffle(gsm)
    keep_gsm = {row["id"] for row in gsm[: args.gsm8k_max]}
    final = [row for row in kept if row["probe"] != "gsm8k" or row["id"] in keep_gsm]
    with Path(args.prompts).open("x", encoding="utf-8") as prompts, Path(
        args.gold
    ).open("x", encoding="utf-8") as gold:
        for row in final:
            criteria = {LETTERS[i]: text for i, text in enumerate(row["options"])}
            prompts.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "state": row["state"],
                        "questions": {
                            "q": {
                                "type": "choice",
                                "instructions": row["instructions"],
                                "criteria": criteria,
                            }
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            gold.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "probe": row["probe"],
                        "gold": LETTERS[row["gold"]],
                        "options": len(row["options"]),
                    }
                )
                + "\n"
            )
    summary = {
        "candidates": dict(Counter(row["probe"] for row in rows)),
        "excluded": dict(
            Counter(row["probe"] for row in rows if row["id"] in excluded)
        ),
        "final": dict(Counter(row["probe"] for row in final)),
        "gsm8k_max": args.gsm8k_max,
        "exclusion_reports": reports,
        "prompts_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(Path(args.gold).read_bytes()).hexdigest(),
    }
    Path(args.report).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def correctness(gold: dict[str, dict], predictions: Path) -> dict[str, int]:
    result = {}
    for record in read_jsonl(predictions):
        answer = record["answers"].get("q", {})
        result[record["id"]] = int(answer.get("choice") == gold[record["id"]]["gold"])
    return result


def score(args: argparse.Namespace) -> None:
    gold = {row["id"]: row for row in read_jsonl(Path(args.gold))}
    left = correctness(gold, Path(args.predictions))
    if set(left) != set(gold):
        raise ValueError("predictions do not cover the probe gold exactly")
    right = correctness(gold, Path(args.reference)) if args.reference else None
    rng = random.Random(f"{SEED}/bootstrap")
    out: dict[str, Any] = {}
    probes = sorted({row["probe"] for row in gold.values()})
    groups = {p: [i for i in gold if gold[i]["probe"] == p] for p in probes}
    groups["arc"] = groups.get("arc-challenge", []) + groups.get("arc-easy", [])
    for name, ids in groups.items():
        if not ids:
            continue
        n = len(ids)
        acc = sum(left[i] for i in ids) / n
        chance = sum(1 / gold[i]["options"] for i in ids) / n
        entry: dict[str, Any] = {"n": n, "accuracy": acc, "chance": chance}
        if right is not None:
            diffs = [left[i] - right[i] for i in ids]
            draws = sorted(
                sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(args.draws)
            )
            entry["reference_accuracy"] = sum(right[i] for i in ids) / n
            entry["delta"] = sum(diffs) / n
            entry["delta_ci95"] = [
                draws[int(0.025 * args.draws)],
                draws[int(0.975 * args.draws) - 1],
            ]
        out[name] = entry
    parts = [p for p in ("mmlu", "arc", "gsm8k") if p in out]
    out["macro_mmlu_arc_gsm8k"] = (
        sum(out[p]["accuracy"] for p in parts) / len(parts) if parts else math.nan
    )
    if right is not None and parts:
        # Each probe resampled within itself; the macro is the mean of the probe deltas.
        diffs = {p: [left[i] - right[i] for i in groups[p]] for p in parts}
        draws = sorted(
            sum(
                sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d)
                for d in diffs.values()
            )
            / len(parts)
            for _ in range(args.draws)
        )
        out["macro_delta"] = sum(out[p]["delta"] for p in parts) / len(parts)
        out["macro_delta_ci95"] = [
            draws[int(0.025 * args.draws)],
            draws[int(0.975 * args.draws) - 1],
        ]
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--mmlu-dir", required=True)
    c.add_argument("--arc-dir", required=True)
    c.add_argument("--gsm8k-dir", required=True)
    c.add_argument("--output", required=True)
    o = sub.add_parser("overlap")
    o.add_argument("--candidates", required=True)
    o.add_argument("--corpus", action="append", required=True)
    o.add_argument("--kind", choices=("train", "suite"), required=True)
    o.add_argument("--output", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--candidates", required=True)
    f.add_argument("--exclude", action="append", required=True)
    f.add_argument("--gsm8k-max", type=int, default=1000)
    f.add_argument("--prompts", required=True)
    f.add_argument("--gold", required=True)
    f.add_argument("--report", required=True)
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--reference")
    s.add_argument("--output", required=True)
    s.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    {
        "candidates": candidates,
        "overlap": overlap,
        "finalize": finalize,
        "score": score,
    }[args.command](args)


if __name__ == "__main__":
    main()
