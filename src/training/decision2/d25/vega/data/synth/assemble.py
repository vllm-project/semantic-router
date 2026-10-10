"""Assemble SYN1 from finished generation shards.

1. Collect accepted rows of every sealed shard (``DONE``) of the given work directories.
2. Drop exact duplicates (normalised state + question + option texts) and near-duplicate states
   across different seeds (MinHash over 5-token shingles, estimated Jaccard >= 0.8). Variants of one
   seed are near-duplicates by design (minimal pairs) and are kept.
3. Decontaminate with ws-data's ``d25.vega.data.decontam check`` against every given index (public
   suite, proxy-protected items); any flagged row is dropped.
4. Shuffle (fixed seed), write ``train-000NN-of-000MM.jsonl.gz`` + ``manifest.json``.

    python -m d25.vega.data.synth.assemble --work /data/d25/vega/synth/syn1 --index IDX1 IDX2 \
        --out /data/d25/shared/data/synth/SYN1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from d25.vega.common import decision_format as df
from d25.vega.data import util

SHINGLE = 5
PERMS = 64
BANDS = 16
MASK61 = np.uint64((1 << 61) - 1)


def collect(
    work_dirs: list[Path], first: int = 0, last: int | None = None
) -> tuple[list[dict], list[str]]:
    """Accepted rows of the sealed shards whose index is in [first, last) (a fixed seed count per release)."""
    rows, shards = [], []
    for work in work_dirs:
        for shard in sorted((work / "shards").glob("*")):
            index = int(shard.name)
            if index < first or (last is not None and index >= last):
                continue
            if (shard / "DONE").exists():
                shards.append(str(shard))
                rows += list(util.read_jsonl(shard / "rows.jsonl.gz"))
    return rows, shards


def row_tokens(row: dict) -> list[str]:
    _, texts = df.options(row["question"])
    text = " ".join(
        [
            util.state_text(row["state"]),
            df.describe(row["question"].get("instructions") or ""),
            *texts,
        ]
    )
    return util.normalize_tokens(text)


def token_ids(tokens: list[str]) -> np.ndarray:
    return np.array(
        [
            int.from_bytes(
                hashlib.blake2b(t.encode(), digest_size=8).digest(), "little"
            )
            >> 3
            for t in tokens
        ],
        dtype=np.uint64,
    )


def minhash(tokens: list[str], a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    if len(tokens) < SHINGLE:
        return None
    ids = token_ids(tokens)
    shingles = ids[: len(ids) - SHINGLE + 1].copy()
    for k in range(1, SHINGLE):
        shingles = (
            shingles * np.uint64(1000003) + ids[k : len(ids) - SHINGLE + 1 + k]
        ) & MASK61
    values = (np.outer(shingles, a) + b) & MASK61
    return values.min(axis=0)


def near_duplicates(rows: list[dict]) -> set[str]:
    rng = np.random.default_rng(20261010)
    a = rng.integers(1, (1 << 61) - 1, size=PERMS, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, size=PERMS, dtype=np.uint64)
    signatures: dict[str, np.ndarray] = {}
    seed_of: dict[str, str] = {}
    seen_states: dict[str, str] = {}
    for row in rows:
        state = util.state_text(row["state"])
        key = util.sha(state, 24)
        if (
            key in seen_states
        ):  # identical states (NOTA pairs, multi-question rows) share a signature
            signatures[row["id"]] = signatures.get(seen_states[key])
        else:
            signatures[row["id"]] = minhash(util.normalize_tokens(state), a, b)
            seen_states[key] = row["id"]
        seed_of[row["id"]] = row["meta"]["seed_id"]
    buckets: dict[tuple, list[str]] = defaultdict(list)
    rows_per_band = PERMS // BANDS
    for row_id, signature in signatures.items():
        if signature is None:
            continue
        for band in range(BANDS):
            buckets[
                (
                    band,
                    signature[
                        band * rows_per_band : (band + 1) * rows_per_band
                    ].tobytes(),
                )
            ].append(row_id)
    drop: set[str] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        for i, first in enumerate(members):
            if first in drop:
                continue
            for other in members[i + 1 :]:
                if other in drop or seed_of[other] == seed_of[first]:
                    continue
                similarity = float(np.mean(signatures[first] == signatures[other]))
                if similarity >= 0.8:
                    drop.add(other)
    return drop


def decontaminate(
    rows: list[dict], indexes: list[str], workers: int, log: dict
) -> set[str]:
    flagged: set[str] = set()
    if not indexes:
        return flagged
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        rows_path = Path(tmp) / "rows.jsonl.gz"
        util.write_jsonl(rows_path, rows)
        for index in indexes:
            out = Path(tmp) / f"flags-{util.sha(index, 8)}.jsonl.gz"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "d25.vega.data.decontam",
                    "check",
                    "--index",
                    index,
                    "--rows",
                    str(rows_path),
                    "--out",
                    str(out),
                    "--workers",
                    str(workers),
                ],
                check=True,
            )
            reasons: Counter = Counter()
            benches: Counter = Counter()
            for flag in util.read_jsonl(out):
                if flag.get("drop"):
                    flagged.add(flag["id"])
                    reasons.update(flag.get("reasons") or [])
                    benches[str(flag.get("bench"))] += 1
            log[index] = {
                "dropped": sum(reasons.values()) and len([1 for _ in reasons]) and None,
                "reasons": dict(reasons),
                "benchmarks": dict(benches.most_common(30)),
            }
            log[index]["dropped"] = int(sum(benches.values()))
    return flagged


def stats(rows: list[dict]) -> dict:
    def count(field):
        return dict(Counter(str(r["meta"].get(field)) for r in rows).most_common())

    noul_true = [r["target"][1] for r in rows if r["question"]["type"] == "noul"]
    yes_choice = [r for r in rows if r["meta"].get("question_form") == "choice_yes_no"]
    yes_rate_choice = sum(
        1
        for r in yes_choice
        if list(r["question"]["criteria"])[r["label"]].lower() == "yes"
    ) / max(len(yes_choice), 1)
    teacher_agree = defaultdict(list)
    for r in rows:
        probs = next(iter(r["meta"].get("teachers", {}).values()), None)
        if probs:
            teacher_agree[r["meta"]["archetype"]].append(
                int(max(range(len(probs)), key=probs.__getitem__) == r["label"])
            )
    positions = Counter()
    for r in rows:
        if r["question"]["type"] == "choice":
            n = len(r["target"])
            positions[f"{min(r['label'] * 4 // max(n, 1), 3)}/4"] += 1
    options = Counter(
        len(r["target"]) for r in rows if r["question"]["type"] == "choice"
    )
    buckets = Counter()
    for n, c in options.items():
        buckets[
            (
                "2"
                if n == 2
                else (
                    "3-4"
                    if n <= 4
                    else (
                        "5-8"
                        if n <= 8
                        else (
                            "9-16"
                            if n <= 16
                            else "17-30" if n <= 30 else "31-60" if n <= 60 else "61+"
                        )
                    )
                )
            )
        ] += c
    return {
        "rows": len(rows),
        "seeds": len({r["meta"]["seed_id"] for r in rows}),
        "domains": len({r["meta"]["domain"] for r in rows}),
        "by_archetype": count("archetype"),
        "by_condition": count("condition"),
        "by_sector": count("sector"),
        "by_question_form": count("question_form"),
        "by_state_format": count("state_format"),
        "by_state_language": count("state_language"),
        "by_nota": count("nota"),
        "by_type": dict(Counter(r["question"]["type"] for r in rows)),
        "noul_p_true_mean": round(float(np.mean(noul_true)), 4) if noul_true else None,
        "yes_no_choice_yes_rate": round(yes_rate_choice, 4),
        "choice_option_buckets": dict(buckets),
        "choice_answer_position_quartile": dict(positions),
        "teacher_argmax_agrees_with_gold": {
            k: round(float(np.mean(v)), 4) for k, v in sorted(teacher_agree.items())
        },
        "words_median": (
            float(np.median([r["meta"]["words"] for r in rows])) if rows else None
        ),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", nargs="+", type=Path, required=True)
    ap.add_argument(
        "--shards",
        default=None,
        help="first:last shard indices to include (default: every sealed shard)",
    )
    ap.add_argument("--index", nargs="*", default=[], help="decontam index directories")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows-per-file", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--code-tag", default="")
    a = ap.parse_args(argv)
    first, last = (int(x) for x in a.shards.split(":")) if a.shards else (0, None)
    rows, shards = collect(a.work, first, last)
    log: dict = {"collected": len(rows), "shards": len(shards)}
    seen, unique = set(), []
    for row in sorted(rows, key=lambda r: r["id"]):
        key = util.sha(" ".join(row_tokens(row)), 24)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    log["exact_duplicates"] = len(rows) - len(unique)
    near = near_duplicates(unique)
    unique = [r for r in unique if r["id"] not in near]
    log["near_duplicates"] = len(near)
    decontam_log: dict = {}
    flagged = decontaminate(unique, a.index, a.workers, decontam_log)
    clean = [r for r in unique if r["id"] not in flagged]
    log["decontam"] = decontam_log
    log["decontam_dropped"] = len(flagged)
    random.Random(a.seed).shuffle(clean)
    a.out.mkdir(parents=True, exist_ok=True)
    files = max(1, -(-len(clean) // a.rows_per_file))
    digests = {}
    for i in range(files):
        name = f"train-{i:05d}-of-{files:05d}.jsonl.gz"
        util.write_jsonl(
            a.out / name, clean[i * a.rows_per_file : (i + 1) * a.rows_per_file]
        )
        digests[name] = util.sha256_file(a.out / name)
    sample = clean[0] if clean else {}
    manifest = {
        "corpus": "SYN1",
        "format": df.FORMAT_ID,
        "licence": "synthetic (generated by Apache-2.0 open model)",
        "generator": sample.get("meta", {}).get("generator"),
        "verifier": sample.get("meta", {}).get("verifier"),
        "prompts": sample.get("meta", {}).get("prompts"),
        "code_tag": a.code_tag,
        "shuffle_seed": a.seed,
        "pipeline": log,
        "files": digests,
        **stats(clean),
    }
    util.write_json(a.out / "manifest.json", manifest)
    print(
        json.dumps({k: manifest[k] for k in ("rows", "seeds", "domains")}),
        json.dumps(log)[:2000],
    )


if __name__ == "__main__":
    main()
