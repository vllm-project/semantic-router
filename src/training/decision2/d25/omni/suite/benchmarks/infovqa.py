"""InfographicVQA: 244 validation infographics, one short-answer question each, as 4-way MC.

Distractors are the answers to the other questions about the same infographic, of the same kind as
the gold answer (with digits / without digits), excluding every accepted gold answer. 248 of the 500
validation infographics have a question with at least three such distractors (the board has 244;
nearby normalisations give 242 to 248). One eligible question per infographic and three distractors
are drawn with a seeded hash, options are shuffled, and 244 infographics are kept by hash order.
Every eligible question is kept in the ``infovqa-val-eligible`` variant. The test split hides its
answers, so the board's rows can only come from validation.
"""

from __future__ import annotations

import ast
import glob
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from d25.omni.suite import rows as R

BENCHMARK = "InfographicVQA"
SOURCES = ("infovqa",)
BOARD_ROWS = 244


def _list(value):
    return value if isinstance(value, list) else ast.literal_eval(value)


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def kind(text) -> bool:
    return bool(re.search(r"\d", str(text)))


def _questions(ctx):
    root = ctx.source("infovqa")
    for path in sorted(
        glob.glob(str(root / "InfographicVQA" / "validation-*.parquet"))
    ):
        name = str(Path(path).relative_to(root))
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for r in batch.to_pylist():
                yield name, r


def build(ctx) -> dict:
    by_image = defaultdict(list)
    payloads = {}
    for name, r in _questions(ctx):
        r["_file"] = name
        r["answers"] = _list(r["answers"])
        by_image[r["image_url"]].append(r)
        payloads.setdefault(r["image_url"], r["image"]["bytes"])
        r.pop("image", None)
        r.pop("ocr", None)
    eligible_rows, per_image = [], {}
    for url in sorted(by_image):
        qs = sorted(by_image[url], key=lambda q: str(q["questionId"]))
        candidates = []
        for q in qs:
            gold = q["answers"][0]
            accepted = {norm(a) for a in q["answers"]}
            pool = {}
            for o in qs:
                if o is q:
                    continue
                a = o["answers"][0]
                if (
                    norm(a) not in accepted
                    and kind(a) == kind(gold)
                    and norm(a) not in pool
                ):
                    pool[norm(a)] = a
            if len(pool) >= 3:
                candidates.append((q, sorted(pool.values(), key=norm)))
        rows_here = []
        for q, pool in candidates:
            ref = ctx.store(payloads[url])
            rng = R.stable_rng("infovqa", q["questionId"])
            distractors = rng.sample(pool, 3)
            options = [q["answers"][0]] + distractors
            rng.shuffle(options)
            gold = R.LETTERS[options.index(q["answers"][0])]
            rows_here.append(
                R.make_row(
                    benchmark=BENCHMARK,
                    split="validation",
                    source_id=str(q["questionId"]),
                    images=[ref],
                    instructions=q["question"],
                    criteria=R.letter_criteria(options),
                    gold=gold,
                    provenance=ctx.provenance(
                        "infovqa", q["_file"], q["questionId"], image_url=url
                    ),
                    tags=[f"answer_type:{t}" for t in _list(q["answer_type"])]
                    + [f"kind:{'number' if kind(q['answers'][0]) else 'text'}"],
                    extra={"accepted_answers": q["answers"], "image_url": url},
                )
            )
        eligible_rows += rows_here
        if rows_here:
            pick = R.stable_rng("infovqa-image", url).randrange(len(rows_here))
            per_image[url] = rows_here[pick]
    order = sorted(per_image, key=lambda u: R.stable_rng("infovqa-keep", u).random())
    keep = set(order[:BOARD_ROWS])
    board = [per_image[u] for u in sorted(per_image) if u in keep]
    return {
        "rows": board,
        "variants": {"infovqa-val-eligible": eligible_rows},
        "notes": f"validation; {len(per_image)} eligible infographics, {len(board)} kept; sibling-answer distractors",
    }
