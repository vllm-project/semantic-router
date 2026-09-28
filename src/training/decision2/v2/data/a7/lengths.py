"""Per-row token lengths for A7 admission and inventory (pinned runtime image).

`native_decoder` tokenizers count exactly the ids that
`training.model.decision_model.encode` builds (prefix, each option, suffix of
`segments`) without its Score-key cast, so raw 1.0 rows with opaque keys can
be counted too. `raw` tokenizers count `v2.data.freeze.raw_text`, as
`v2.data.freeze` does. Output: one {"id", <tokenizer name>: n, ...} per row,
sorted by id, plus a summary on stdout.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v2.data.freeze import TOKENIZER_KINDS, load_tokenizer, raw_text

_WORK: dict[str, Any] = {}


def native_length(row: Mapping[str, Any], tokenizer: Any) -> int:
    from training.model.decision_model import segments

    prefix, options, suffix = segments(dict(row))
    total = len(tokenizer.encode(prefix, add_special_tokens=False))
    for option in options:
        total += len(tokenizer.encode(option, add_special_tokens=False))
    return total + len(tokenizer.encode(suffix, add_special_tokens=False))


def row_length(row: Mapping[str, Any], spec: Mapping[str, Any], tokenizer: Any) -> int:
    if spec["kind"] == "native_decoder":
        return native_length(row, tokenizer)
    return len(tokenizer.encode(raw_text(row), add_special_tokens=False))


def _init(specs: Sequence[Mapping[str, Any]]) -> None:
    _WORK["specs"] = list(specs)
    _WORK["tokenizers"] = [load_tokenizer(spec) for spec in specs]


def _count(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item: dict[str, Any] = {"id": row["id"]}
        for spec, tokenizer in zip(_WORK["specs"], _WORK["tokenizers"]):
            item[spec["name"]] = row_length(row, spec, tokenizer)
        out.append(item)
    return out


def count_rows(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    workers: int = 1,
    chunk: int = 512,
) -> list[dict[str, Any]]:
    for spec in specs:
        if spec.get("kind") not in TOKENIZER_KINDS:
            raise ValueError(f"tokenizer kind must be one of {TOKENIZER_KINDS}")
    batches = [rows[start : start + chunk] for start in range(0, len(rows), chunk)]
    if workers <= 1:
        _init(specs)
        results = [_count(batch) for batch in batches]
    else:
        with multiprocessing.get_context("fork").Pool(
            workers, initializer=_init, initargs=(specs,)
        ) as pool:
            results = pool.map(_count, batches)
    return sorted(
        (item for batch in results for item in batch), key=lambda item: item["id"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rows",
        action="append",
        required=True,
        help="PATH, or NAME=PATH to prefix ids with 'NAME:'",
    )
    parser.add_argument(
        "--tokenizers",
        type=Path,
        required=True,
        help="JSON list of {name, path, revision, kind: native_decoder|raw}",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    specs = json.loads(args.tokenizers.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for value in args.rows:
        name, separator, path_text = value.partition("=")
        path = Path(path_text if separator else value)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if separator:
                        row = dict(row, id=f"{name}:{row['id']}")
                    rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("row ids must be unique across --rows files")
    counted = count_rows(rows, specs, workers=args.workers)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for item in counted:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    summary = {
        spec["name"]: {
            "total": sum(item[spec["name"]] for item in counted),
            "max": max((item[spec["name"]] for item in counted), default=0),
        }
        for spec in specs
    }
    print(json.dumps({"rows": len(counted), "tokens": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
