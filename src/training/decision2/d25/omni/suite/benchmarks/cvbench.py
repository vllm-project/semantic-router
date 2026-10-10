"""CV-Bench: test Count (788) + Relation (650) + Distance (600) = 2,038 rows, 2 to 6 options.

The board lists "counting, 2D relation and distance items", so Depth (600 rows) is left out; the
score lattice confirms the chance sum of exactly these rows (802.62). Depth is kept as a variant.
"""

from __future__ import annotations

import pyarrow.parquet as pq

from d25.omni.suite import rows as R

BENCHMARK = "CV-Bench"
SOURCES = ("cvbench",)
BOARD_TASKS = ("Count", "Relation", "Distance")
FILES = ("test_2d.parquet", "test_3d.parquet")


def _iter(ctx):
    for name in FILES:
        pf = pq.ParquetFile(ctx.source("cvbench") / name)
        index = 0
        for batch in pf.iter_batches(batch_size=128):
            for r in batch.to_pylist():
                yield name, index, r
                index += 1


def build(ctx) -> dict:
    board, depth = [], []
    for name, index, r in _iter(ctx):
        source_id = f"{r['type']}-{index}"
        row = R.make_row(
            benchmark=BENCHMARK,
            split="test",
            subtask=r["task"],
            source_id=source_id,
            images=[ctx.store(r["image"]["bytes"])],
            instructions=r["question"],
            criteria=R.letter_criteria(r["choices"]),
            gold=R.letter_of(r["answer"]),
            provenance=ctx.provenance(
                "cvbench", name, source_id, filename=r["filename"]
            ),
            tags=[f"source:{r['source_dataset']}"],
            extra={
                "source_dataset": r["source_dataset"],
                "source_filename": r["source_filename"],
            },
        )
        (board if r["task"] in BOARD_TASKS else depth).append(row)
    return {
        "rows": board,
        "variants": {"cvbench-depth": depth},
        "notes": "test_2d + test_3d, tasks Count/Relation/Distance; question text as instructions, choices in source order",
    }
