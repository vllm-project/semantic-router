"""R-Bench-M (en): 665 test rows, one image, options A to F.

Empty options are dropped (row 295 lists only A to D), which reproduces the board's chance sum
110.917; duplicated option texts are kept as published.
"""

from __future__ import annotations

import pyarrow.parquet as pq

from d25.omni.suite import rows as R

BENCHMARK = "R-Bench-M"
SOURCES = ("rbench",)
FILE = "rbench-m_en/test-00000-of-00001.parquet"


def build(ctx) -> dict:
    out = []
    for r in pq.ParquetFile(ctx.source("rbench") / FILE).read().to_pylist():
        criteria = {
            k: r[k] for k in "ABCDEF" if r.get(k) is not None and str(r[k]).strip()
        }
        out.append(
            R.make_row(
                benchmark=BENCHMARK,
                split="test",
                source_id=str(r["index"]),
                images=[ctx.store(r["image"]["bytes"])],
                instructions=r["question"],
                criteria=criteria,
                gold=r["answer"].strip(),
                provenance=ctx.provenance("rbench", FILE, r["index"]),
            )
        )
    return {
        "rows": out,
        "variants": {},
        "notes": "rbench-m_en test; empty options dropped",
    }
