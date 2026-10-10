"""RealWorldQA: 589 rows parsed from the 765 test prompts (426 multiple choice + 163 yes/no).

Multiple choice: option lines matching ``^\\s*[A-F][.)]\\s`` whose letters run A, B, ... in order and
include the answer letter. Yes/no: prompts without option lines whose answer is Yes or No. The strict
parse finds 591 rows; rows 448 (``C.Red``) and 693 (``B Only``) parse as malformed two-option
questions and are dropped, which gives the board's 589 rows and its chance sum 228.583 exactly.
They are kept, re-parsed leniently, in the ``realworldqa-dropped`` variant. The remaining 174
open-answer prompts are not built.
"""

from __future__ import annotations

import re

import pyarrow.parquet as pq

from d25.omni.suite import rows as R

BENCHMARK = "RealWorldQA"
SOURCES = ("realworldqa",)
FILES = ("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet")
OPTION = re.compile(r"^\s*([A-F])[.)]\s+(.*\S)\s*$")
LENIENT = re.compile(r"^\s*([A-F])(?:[.)]\s*|\s+)(.*\S)\s*$")
TRAILER = re.compile(r"\n?Please answer directly with[^\n]*$")
YES_NO = {"Yes": "yes", "No": "no"}


def parse(question: str, answer: str, pattern=OPTION):
    """``("mc", stem, options)``, ``("yn", stem, None)``, ``("bad", ...)`` or ``("open", ...)``."""
    body = TRAILER.sub("", question.rstrip())
    lines = body.split("\n")
    options = [(m.group(1), m.group(2)) for line in lines if (m := pattern.match(line))]
    stem = "\n".join(line for line in lines if not pattern.match(line)).strip()
    if options:
        letters = [k for k, _ in options]
        ok = letters == list(R.LETTERS[: len(letters)]) and answer in letters
        return ("mc" if ok else "bad"), stem, options
    if answer in YES_NO:
        return "yn", stem, None
    return "open", stem, None


def build(ctx) -> dict:
    board, dropped = [], []
    index = 0
    for name in FILES:
        for r in pq.ParquetFile(ctx.source("realworldqa") / name).read().to_pylist():
            kind, stem, options = parse(r["question"], r["answer"])
            source_id = str(index)
            index += 1
            if kind == "open":
                continue
            target = board
            if kind == "bad":
                kind, stem, options = parse(r["question"], r["answer"], LENIENT)
                target = dropped
            if kind == "mc":
                criteria, gold = {k: v for k, v in options}, r["answer"]
            else:
                criteria, gold = {"yes": "Yes", "no": "No"}, YES_NO[r["answer"]]
            target.append(
                R.make_row(
                    benchmark=BENCHMARK,
                    split="test",
                    subtask="multiple-choice" if kind == "mc" else "yes-no",
                    source_id=source_id,
                    images=[ctx.store(r["image"]["bytes"])],
                    instructions=stem,
                    criteria=criteria,
                    gold=gold,
                    provenance=ctx.provenance("realworldqa", name, source_id),
                    tags=[f"format:{kind}"]
                    + (["malformed-options"] if target is dropped else []),
                )
            )
    return {
        "rows": board,
        "variants": {"realworldqa-dropped": dropped},
        "notes": "strict option parse; malformed option rows dropped; Please-answer trailer removed",
    }
