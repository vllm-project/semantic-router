"""CharXiv: 532 validation reasoning questions as 4-way multiple choice with distractors from the same figure.

The test split hides its reasoning answers, so the board's rows can only come from validation (1,000
figures, one reasoning question each). Every row is 4-way, so the board's chance sum (133.0) fixes the row
count but not the distractor rule. The only per-figure annotations besides the reasoning pair are the three
descriptive answers (the fourth is always "Not Applicable" in validation): axis labels, tick values, legend
labels (comma-separated lists, split into items), counts, layouts, trends. A question's candidate
distractors are those items, without "Not Applicable" and without anything equal to the gold answer
(numbers compared by value, text case- and space-insensitively); candidates of the gold's kind (number or
text) come first, then the others, each group in seeded order, and the first three are taken. 896 questions
have at least three candidates; 532 are kept by seeded hash order. All eligible questions are kept in the
``charxiv-val-eligible`` variant and those whose three distractors share the gold's kind in
``charxiv-val-same-kind`` (same row ids and options as the board rows).
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq

from d25.omni.suite import rows as R

BENCHMARK = "CharXiv"
SOURCES = ("charxiv",)
FILE = "val.parquet"
BOARD_ROWS = 532
ANSWER_TYPES = {
    1: "text-in-chart",
    2: "text-in-general",
    3: "number-in-chart",
    4: "number-in-general",
}
NOT_APPLICABLE = {"", "not applicable", "n/a", "none"}
SPLIT = re.compile(r",\s+")
NUMBER = re.compile(r"[-+−]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.I)
POWER = re.compile(r"(?:([-+−]?\d+(?:\.\d+)?)\s*[x×*]\s*)?10\^\{?([-+−]?\d+)\}?")


def number(text: str) -> float | None:
    """Numeric value of a plain number, percentage, money amount or power of ten; else None."""
    s = text.strip().replace(" ", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s).lstrip("$").rstrip("%")
    if NUMBER.fullmatch(s):
        return float(s.replace("−", "-"))
    m = POWER.fullmatch(s)
    if m:
        mantissa = float(m.group(1).replace("−", "-")) if m.group(1) else 1.0
        return mantissa * 10.0 ** int(m.group(2).replace("−", "-"))
    return None


def key(text: str) -> str:
    value = number(text)
    if value is not None:
        return f"#{value:.12g}"
    return re.sub(r"\s+", " ", text.strip().strip(".").lower())


def candidates(row: dict, gold: str) -> list[str]:
    """Same-figure distractor candidates in preference order (gold's kind first, seeded within kind)."""
    gold_key, gold_numeric = key(gold), number(gold) is not None
    pool: dict[str, str] = {}
    for i in range(1, 5):
        answer = row.get(f"descriptive_a{i}")
        if answer is None:
            continue
        for item in SPLIT.split(str(answer)):
            item = item.strip()
            k = key(item)
            if item.lower() in NOT_APPLICABLE or k == gold_key or k in pool:
                continue
            pool[k] = item
    same = [v for v in pool.values() if (number(v) is not None) == gold_numeric]
    other = [v for v in pool.values() if (number(v) is not None) != gold_numeric]
    fid = Path(row["figure_path"]).stem
    R.stable_rng("charxiv-same", fid).shuffle(same)
    R.stable_rng("charxiv-other", fid).shuffle(other)
    return same + other


def build(ctx) -> dict:
    eligible, same_kind = [], []
    by_type: dict[str, int] = {}
    root = ctx.source("charxiv")
    for batch in pq.ParquetFile(root / FILE).iter_batches(batch_size=32):
        for r in batch.to_pylist():
            gold = str(r["reasoning_a"]).strip()
            pool = candidates(r, gold)
            if len(pool) < 3:
                continue
            fid = Path(r["figure_path"]).stem
            distractors = pool[:3]
            gold_numeric = number(gold) is not None
            n_same = sum((number(d) is not None) == gold_numeric for d in distractors)
            options = [gold] + distractors
            R.stable_rng("charxiv-options", fid).shuffle(options)
            answer_type = ANSWER_TYPES.get(r["reasoning_a_type"], "unknown")
            row = R.make_row(
                benchmark=BENCHMARK,
                split="validation",
                subtask=answer_type,
                source_id=fid,
                images=[ctx.store(r["image"]["bytes"])],
                instructions=r["reasoning_q"].strip(),
                criteria=R.letter_criteria(options),
                gold=R.LETTERS[options.index(gold)],
                provenance=ctx.provenance(
                    "charxiv",
                    FILE,
                    fid,
                    figure_path=r["figure_path"],
                    original_id=r["original_id"],
                ),
                tags=[
                    f"answer_type:{answer_type}",
                    f"q_source:{r['reasoning_q_source']}",
                    f"same_kind_distractors:{n_same}",
                ],
                extra={
                    "category": r["category"],
                    "num_subplots": r["num_subplots"],
                    "gold_text": gold,
                },
            )
            eligible.append(row)
            by_type[answer_type] = by_type.get(answer_type, 0) + 1
            if n_same == 3:
                same_kind.append(row)
    order = sorted(
        eligible,
        key=lambda row: R.stable_rng(
            "charxiv-keep", row["metadata"]["provenance"]["source_id"]
        ).random(),
    )
    keep = {row["id"] for row in order[:BOARD_ROWS]}
    board = [row for row in eligible if row["id"] in keep]
    return {
        "rows": board,
        "variants": {
            "charxiv-val-eligible": eligible,
            "charxiv-val-same-kind": same_kind,
        },
        "notes": (
            f"validation reasoning questions; {len(eligible)} with >= 3 same-figure distractor candidates "
            f"{dict(sorted(by_type.items()))}, {len(board)} kept by seeded hash; {len(same_kind)} with three "
            "distractors of the gold's kind"
        ),
    }
