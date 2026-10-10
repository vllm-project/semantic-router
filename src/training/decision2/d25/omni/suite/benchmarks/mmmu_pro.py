"""MMMU-Pro vision: 1,730 test rows; the question and its options are shown inside the image.

Each row offers the letters of the source option list (2 to 10, mostly 10) with no option text,
since the options are part of the screenshot. The two rows changed by the 2026-10-04 upstream fix
are tagged ``revision-sensitive``.
"""

from __future__ import annotations

import ast
import glob
from pathlib import Path

import pyarrow.parquet as pq

from d25.omni.suite import rows as R

BENCHMARK = "MMMU-Pro vision"
SOURCES = ("mmmu_pro",)
INSTRUCTIONS = (
    "The image shows a multiple-choice question together with its lettered options. "
    "Which option is correct?"
)
REVISION_SENSITIVE = {"test_Chemistry_240", "validation_Finance_5"}


def options(text: str) -> list:
    value = ast.literal_eval(text)
    if not isinstance(value, list) or not value:
        raise ValueError(f"bad options {text[:80]!r}")
    return value


def build(ctx) -> dict:
    out = []
    root = ctx.source("mmmu_pro")
    for path in sorted(glob.glob(str(root / "vision" / "test-*.parquet"))):
        name = str(Path(path).relative_to(root))
        for r in pq.ParquetFile(path).read().to_pylist():
            n = len(options(r["options"]))
            out.append(
                R.make_row(
                    benchmark=BENCHMARK,
                    split="test",
                    subtask=r["subject"],
                    source_id=r["id"],
                    images=[ctx.store(r["image"]["bytes"])],
                    instructions=INSTRUCTIONS,
                    criteria={R.LETTERS[i]: None for i in range(n)},
                    gold=r["answer"].strip(),
                    provenance=ctx.provenance("mmmu_pro", name, r["id"]),
                    tags=(
                        ["revision-sensitive"] if r["id"] in REVISION_SENSITIVE else []
                    ),
                )
            )
    return {
        "rows": out,
        "variants": {},
        "notes": "vision config; letters only, n options = len(source options)",
    }
