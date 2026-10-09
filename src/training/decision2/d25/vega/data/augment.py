"""Augmentation rows for M2 (each tagged in meta.aug; the original rows stay in the mixture too).

    python -m d25.vega.data.augment --index DIR --holdouts holdouts.json --rows <base row files> --out rows/aug/aug.jsonl.gz

Types (counts are targets; base rows are decontamination- and holdout-clean, hard-label where needed):
    shuffle          option order permuted, target permuted with it (positional keys stay positional)
    nota_replace /   none-of-the-above minimal pair: the gold option replaced by "None of the other options"
    nota_keep        (target = NOTA) and the same question with NOTA added but gold kept (target = gold)
    verify_shifted   label-verification noul ("Proposed answer: X. Is it correct?") with a per-family yes-rate
                     of 0.2, 0.5 or 0.8, so the prior differs across families
    verify_negated   the same with the question negated ("Is it incorrect?"), label flipped
    long_pad         string state padded with unrelated documents (marked as unrelated), question unchanged
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as df
from d25.vega.data.holdouts import Holdouts
from d25.vega.data.tokens import load_lengths
from d25.vega.data.util import (
    make_row,
    rank,
    read_jsonl,
    rng_for,
    write_json,
    write_jsonl,
)

PLAN = {
    "shuffle": 40000,
    "nota": 12000,
    "verify_shifted": 30000,
    "verify_negated": 10000,
    "long_pad": 15000,
}
LETTER = re.compile(r"^[A-Z]$")
OPTION_I = re.compile(r"^option_\d+$")
NOTA_LIKE = re.compile(r"\b(none|other|all of|neither|both)\b", re.I)
REFERENTIAL = re.compile(
    r"\b(above|below|previous|following option|options? [a-e]\b|[a-e] and [a-e])\b",
    re.I,
)
VERIFY = [
    "Is the proposed answer correct?",
    "Is this the right answer?",
    "Does the proposed option correctly answer the question?",
]
NEGATED = [
    "Is the proposed answer incorrect?",
    "Is this the wrong answer?",
    "Does the proposed option fail to answer the question correctly?",
]
NOTA_TEXT = [
    "None of the other options",
    "None of the other options is correct.",
    "None of these",
]


def key_style(criteria: dict[str, Any]) -> str:
    keys = list(criteria)
    if all(LETTER.match(k) for k in keys) and keys == [
        chr(65 + i) for i in range(len(keys))
    ]:
        return "letters"
    if all(OPTION_I.match(k) for k in keys) and keys == [
        f"option_{i}" for i in range(len(keys))
    ]:
        return "option_i"
    if all(v is None for v in criteria.values()):
        return "text"
    return "other"


def rebuild(style: str, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Criteria from (key, description) pairs; positional key styles are re-keyed by position."""
    if style == "letters":
        return {
            chr(65 + i): (d if d is not None else k) for i, (k, d) in enumerate(pairs)
        }
    if style == "option_i":
        return {
            f"option_{i}": (d if d is not None else k) for i, (k, d) in enumerate(pairs)
        }
    return dict(pairs)


def option_text(key: str, desc: Any) -> str:
    if desc is None:
        return key
    text = df.describe(desc)
    return text if (LETTER.match(key) or OPTION_I.match(key)) else f"{key}: {text}"


def eligible_choice(row: dict[str, Any], hard: bool, max_options: int = 26) -> bool:
    q = row["question"]
    if q["type"] != "choice" or not 2 <= len(q["criteria"]) <= max_options:
        return False
    texts = [option_text(k, v) for k, v in q["criteria"].items()]
    if any(REFERENTIAL.search(t) for t in texts):
        return False
    return not hard or max(row["target"]) >= 0.99


def derived(
    base: dict[str, Any],
    aug: str,
    suffix: str,
    question: dict[str, Any],
    target: list[float],
    state: Any = None,
) -> dict[str, Any]:
    meta = dict(base["meta"])
    meta.pop("n_tokens", None)
    meta.update(
        {"aug": aug, "aug_of": base["id"], "soft": any(0 < v < 1 for v in target)}
    )
    return make_row(
        row_id=f"{base['id']}:aug-{suffix}",
        source=base["source"],
        family=base["family"],
        state=base["state"] if state is None else state,
        question=question,
        target=target,
        meta=meta,
    )


def build(
    rows: list[dict[str, Any]], lengths: dict[str, int], seed: int
) -> tuple[list[dict[str, Any]], Counter]:
    out: list[dict[str, Any]] = []
    stats: Counter = Counter()
    order = sorted(rows, key=lambda r: rank(r["id"], f"{seed}:aug"))
    used: set[str] = set()

    def take(kind: str, n: int, pred):
        chosen = []
        for row in order:
            if len(chosen) >= n:
                break
            if row["id"] in used or not pred(row):
                continue
            chosen.append(row)
            used.add(row["id"])
        stats[f"{kind}:base"] = len(chosen)
        return chosen

    for row in take(
        "shuffle",
        PLAN["shuffle"],
        lambda r: eligible_choice(r, hard=False)
        and len(r["question"]["criteria"]) >= 3,
    ):
        rng = rng_for(row["id"], f"{seed}:shuffle")
        q = row["question"]
        style = key_style(q["criteria"])
        pairs = list(q["criteria"].items())
        perm = list(range(len(pairs)))
        while perm == sorted(perm):
            rng.shuffle(perm)
        new_pairs = [pairs[i] for i in perm]
        target = [row["target"][i] for i in perm]
        if style in ("letters", "option_i"):
            new_pairs = [(pairs[i][0], pairs[i][1]) for i in perm]
        out.append(
            derived(
                row,
                "shuffle",
                "shuffle",
                {**q, "criteria": rebuild(style, new_pairs)},
                target,
            )
        )
        stats["shuffle"] += 1

    def nota_ok(r):
        """Answer-like options only: short label sets (classification) can have several applicable labels, where
        "none of the other options" would be wrong after removing the gold one."""
        if (
            not eligible_choice(r, hard=True, max_options=12)
            or len(r["question"]["criteria"]) < 3
        ):
            return False
        texts = [option_text(k, v) for k, v in r["question"]["criteria"].items()]
        if any(NOTA_LIKE.search(t) for t in texts):
            return False
        return sum(len(t.split()) for t in texts) / len(texts) >= 3 or r[
            "source"
        ].startswith("knowledge:")

    for row in take("nota", PLAN["nota"], nota_ok):
        rng = rng_for(row["id"], f"{seed}:nota")
        q = row["question"]
        style = key_style(q["criteria"])
        pairs = list(q["criteria"].items())
        gold = max(range(len(pairs)), key=row["target"].__getitem__)
        nota_text = rng.choice(NOTA_TEXT)
        nota_pair = (
            (nota_text, None)
            if style == "text"
            else (
                ("none_of_the_others", nota_text)
                if style == "other"
                else (nota_text, nota_text)
            )
        )
        removed = [p for i, p in enumerate(pairs) if i != gold]
        pos = rng.randrange(len(removed) + 1)
        replaced = removed[:pos] + [nota_pair] + removed[pos:]
        t_rep = [0.0] * len(replaced)
        t_rep[pos] = 1.0
        kept = list(pairs)
        pos_k = rng.randrange(len(kept) + 1)
        kept = kept[:pos_k] + [nota_pair] + kept[pos_k:]
        t_keep = [0.0] * len(kept)
        t_keep[gold + (1 if pos_k <= gold else 0)] = 1.0
        pair_id = f"{row['id']}:nota"
        a = derived(
            row,
            "nota_replace",
            "nota-replace",
            {**q, "criteria": rebuild(style, replaced)},
            t_rep,
        )
        b = derived(
            row,
            "nota_keep",
            "nota-keep",
            {**q, "criteria": rebuild(style, kept)},
            t_keep,
        )
        a["meta"]["pair"] = b["meta"]["pair"] = pair_id
        out += [a, b]
        stats["nota_pairs"] += 1

    family_rate: dict[str, float] = {}

    def rate(family: str) -> float:
        if family not in family_rate:
            family_rate[family] = [0.2, 0.5, 0.8][
                int(rank(family, f"{seed}:rate")[:8], 16) % 3
            ]
        return family_rate[family]

    for kind in ("verify_shifted", "verify_negated"):
        for row in take(
            kind, PLAN[kind], lambda r: eligible_choice(r, hard=True, max_options=30)
        ):
            rng = rng_for(row["id"], f"{seed}:{kind}")
            q = row["question"]
            pairs = list(q["criteria"].items())
            gold = max(range(len(pairs)), key=row["target"].__getitem__)
            yes = rng.random() < (
                rate(row["family"]) if kind == "verify_shifted" else 0.5
            )
            pick = (
                gold if yes else rng.choice([i for i in range(len(pairs)) if i != gold])
            )
            proposal = option_text(*pairs[pick])
            ask = rng.choice(VERIFY if kind == "verify_shifted" else NEGATED)
            instructions = f"{df.describe(q.get('instructions') or 'Choose the best matching option.')}\nProposed answer: {proposal}\n{ask}"
            correct = 1.0 if pick == gold else 0.0
            p_true = correct if kind == "verify_shifted" else 1.0 - correct
            noul = {"type": "noul", "instructions": instructions}
            out.append(
                derived(row, kind, kind.replace("_", "-"), noul, [1.0 - p_true, p_true])
            )
            stats[f"{kind}:yes" if p_true >= 0.5 else f"{kind}:no"] += 1

    distractors = [
        r["state"]
        for r in order
        if isinstance(r["state"], str) and 200 <= len(r["state"]) <= 4000
    ][:200000]
    for row in take(
        "long_pad",
        PLAN["long_pad"],
        lambda r: isinstance(r["state"], str)
        and len(r["state"]) >= 20
        and 0 < lengths.get(r["id"], 10**9) <= 1500,
    ):
        rng = rng_for(row["id"], f"{seed}:pad")
        docs, budget = [], rng.randrange(2000, 9000)
        while budget > 0 and distractors:
            doc = distractors[rng.randrange(len(distractors))]
            if doc == row["state"]:
                continue
            docs.append(doc)
            budget -= len(doc)
        block = "\n\n".join(docs)
        if rng.random() < 0.5:
            state = f"{row['state']}\n\n---\nUnrelated material (not relevant to the question):\n{block}"
        else:
            state = f"Unrelated material (not relevant to the question):\n{block}\n---\n\n{row['state']}"
        out.append(
            derived(
                row,
                "long_pad",
                "long-pad",
                row["question"],
                list(row["target"]),
                state=state,
            )
        )
        stats["long_pad"] += 1
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--holdouts", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    dropped = set()
    for path in sorted(args.index.glob("flags-*.jsonl.gz")):
        dropped |= {f["id"] for f in read_jsonl(path) if f["drop"]}
    holdouts = Holdouts(args.holdouts)
    files = [p for p in args.rows if not p.endswith(".ntok.jsonl.gz")]
    lengths = load_lengths(files)
    rows = []
    skipped = Counter()
    for path in files:
        for row in read_jsonl(path):
            if row["id"] in dropped:
                skipped["decontam"] += 1
            elif holdouts.check(row) or row["source"].startswith("d20:"):
                skipped["holdouts_or_d20"] += 1
            else:
                rows.append(row)
    out, stats = build(rows, lengths, args.seed)
    write_jsonl(args.out, out)
    report = {
        "plan": PLAN,
        "base_rows": len(rows),
        "skipped": dict(skipped),
        "stats": dict(stats),
        "rows": len(out),
        "files": files,
        "seed": args.seed,
    }
    write_json(args.out.with_name("report.json"), report)
    print(json.dumps(report, indent=1)[:3000])


if __name__ == "__main__":
    main()
