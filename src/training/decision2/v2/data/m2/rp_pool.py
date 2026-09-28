"""RP-v2 teacher prompt pool: every non-A0s row of the given recipe manifests.

    python3 -m v2.data.m2.rp_pool --recipe mx-v2-full-L.ids.jsonl \\
        --recipe mx-v2-short-M.ids.jsonl --rows H1.train.jsonl ... \\
        --out-rows rp-v2.rows.jsonl --out-prompts rp-v2.prompts.jsonl

A0s rows are skipped (the canonical own-Lux A0 TRAIN file covers them). Because
recipes are nested (S ⊂ M ⊂ L), targets on this pool cover every recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from training.model.data import canonical
from v2.data.build_a0_variants import native_prompt
from v2.data.m2.common import read_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", type=Path, action="append", required=True)
    parser.add_argument("--rows", type=Path, action="append", required=True)
    parser.add_argument("--out-rows", type=Path, required=True)
    parser.add_argument("--out-prompts", type=Path, required=True)
    args = parser.parse_args(argv)
    wanted = {
        r["id"] for path in args.recipe for r in read_jsonl(path) if r["pool"] != "A0s"
    }
    rows = {}
    for path in args.rows:
        for row in read_jsonl(path):
            if row["id"] in wanted:
                rows[row["id"]] = row
    missing = wanted - set(rows)
    if missing:
        raise ValueError(
            f"{len(missing)} recipe ids not found, e.g. {sorted(missing)[:3]}"
        )
    ordered = [rows[i] for i in sorted(rows)]
    for path, lines in (
        (args.out_rows, [canonical(r) for r in ordered]),
        (
            args.out_prompts,
            [json.dumps(native_prompt(r), ensure_ascii=False) for r in ordered],
        ),
    ):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.writelines(line + "\n" for line in lines)
    print(json.dumps({"rows": len(ordered)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
