"""BLINK: seven val subtasks, 961 rows, 2 to 4 options over 1 to 2 images (set E).

All 14 val subtasks are built and tagged ``subtask:<name>``; the board rows are the set chosen in
``blink_subset`` (default E: Counting, Functional_Correspondence, Multi-view_Reasoning,
Relative_Depth, Semantic_Correspondence, Spatial_Relation, Visual_Correspondence), and every val row
is also written to the ``blink-val-all`` variant so another set can be selected without a rebuild.
Instructions are the source prompt without its trailing option list; images keep source order.
"""

from __future__ import annotations

import re

import pyarrow.parquet as pq

from d25.omni.suite import blink_subset
from d25.omni.suite import rows as R

BENCHMARK = "BLINK"
SOURCES = ("blink",)
SELECT = re.compile(
    r"\s*Select (from|between) the following (choices|options)\.?\s*(\n|$).*", re.S
)


def instructions(prompt: str) -> str:
    text = SELECT.sub("", prompt).strip()
    if not text:
        raise ValueError(f"empty BLINK instructions from {prompt!r}")
    return text


def build(ctx, subset: str | None = None) -> dict:
    chosen = set(blink_subset.NAMED[subset or blink_subset.DEFAULT])
    board, everything = [], []
    for sub in sorted(blink_subset.SUBTASKS):
        name = f"{sub}/val-00000-of-00001.parquet"
        table = pq.ParquetFile(ctx.source("blink") / name).read()
        for r in table.to_pylist():
            images = [
                ctx.store(r[f"image_{i}"]["bytes"])
                for i in range(1, 5)
                if r.get(f"image_{i}")
            ]
            row = R.make_row(
                benchmark=BENCHMARK,
                split="val",
                subtask=sub,
                source_id=r["idx"],
                images=images,
                instructions=instructions(r["prompt"]),
                criteria=R.letter_criteria(r["choices"]),
                gold=R.letter_of(r["answer"]),
                provenance=ctx.provenance("blink", name, r["idx"]),
                tags=[
                    f"subtask:{sub}",
                    "board-set:" + ("yes" if sub in chosen else "no"),
                ]
                + [
                    f"in:{k}"
                    for k, members in blink_subset.NAMED.items()
                    if sub in members
                ],
            )
            everything.append(row)
            if sub in chosen:
                board.append(row)
    return {
        "rows": board,
        "variants": {"blink-val-all": everything},
        "notes": f"val, set {subset or blink_subset.DEFAULT} = {sorted(chosen)}",
    }
