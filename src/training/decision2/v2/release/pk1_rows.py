"""Row-for-row check of a mixture's base rows against a published data file (stdlib).

A size track that renumbered option keys inside its own mixture builder (for
example the 0.6B track's A0s-r) uses "renumbered data" only if its base rows
equal the research & data track's published positional-key file minus the
excluded families, row for row (coordinator decision 2026-09-28 21:30 (2)).
This compares ids, order and the exact canonical JSON line of every row, and
optionally counts the rows that differ from the pre-renumbering original.
Receipts carry counts and digests only, never row text.

    python3 -m v2.release.pk1_rows --mixture M.jsonl --mixture-sha256 H \
        --published P.jsonl --published-sha256 H --exclude-family F [...] \
        [--original O.jsonl --original-sha256 H] --output receipt.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from v2.release.layout import canonical, sha_file, write_json

SCHEMA = "dev2-release-pk1-rows/1"


def read_lines(path: Path, expected_sha256: str) -> list[tuple[bytes, dict[str, Any]]]:
    if sha_file(path) != expected_sha256:
        raise ValueError(f"{path.name} differs from its pinned SHA-256")
    rows = []
    with Path(path).open("rb") as stream:
        for line in stream:
            text = line.rstrip(b"\n")
            if text:
                rows.append((text, json.loads(text)))
    return rows


def _digest(values: list[Any]) -> str:
    return hashlib.sha256(canonical(values).encode("utf-8")).hexdigest()


def _lines_digest(lines: list[bytes]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line + b"\n")
    return digest.hexdigest()


def compare(
    mixture: list[tuple[bytes, dict[str, Any]]],
    published: list[tuple[bytes, dict[str, Any]]],
    excluded_families: set[str],
    original: list[tuple[bytes, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    kept = [
        (line, row) for line, row in published if row["family"] not in excluded_families
    ]
    dropped = Counter(
        row["family"] for _, row in published if row["family"] in excluded_families
    )
    kept_ids = [row["id"] for _, row in kept]
    if len(set(kept_ids)) != len(kept_ids):
        raise ValueError("Published file repeats an id")
    excluded_ids = {row["id"] for _, row in published} - set(kept_ids)
    by_id = {}
    for index, (line, row) in enumerate(mixture):
        if row["id"] in by_id:
            raise ValueError("Mixture repeats an id")
        by_id[row["id"]] = (index, line, row)
    base = [by_id[i] for i in kept_ids if i in by_id]
    kept_inputs = {row["input_sha256"]: row["id"] for _, row in kept}
    line_identical = row_equal = 0
    field_differences: Counter[str] = Counter()
    for (line, row), (_, m_line, m_row) in zip(
        kept, (by_id.get(i, (None, None, None)) for i in kept_ids)
    ):
        if m_row is None:
            continue
        if m_line == line:
            line_identical += 1
        if m_row == row:
            row_equal += 1
        else:
            field_differences.update(
                k for k in set(row) | set(m_row) if row.get(k) != m_row.get(k)
            )
    repeats = sum(
        1
        for _, row in mixture
        if row["input_sha256"] in kept_inputs
        and kept_inputs[row["input_sha256"]] != row["id"]
    )
    result: dict[str, Any] = {
        "published_rows": len(published),
        "excluded_family_rows": dict(sorted(dropped.items())),
        "published_kept_rows": len(kept),
        "mixture_rows": len(mixture),
        "mixture_rows_with_kept_ids": len(base),
        "missing_from_mixture": len(kept) - len(base),
        "excluded_ids_in_mixture": len(excluded_ids & set(by_id)),
        "kept_rows_form_mixture_prefix_in_order": [index for index, _, _ in base]
        == list(range(len(base))),
        "line_identical_rows": line_identical,
        "row_equal_rows": row_equal,
        "differing_fields": dict(sorted(field_differences.items())),
        "other_mixture_rows_repeating_a_kept_input": repeats,
        "kept_ids_sha256": _digest(sorted(kept_ids)),
        "kept_lines_sha256": _lines_digest([line for line, _ in kept]),
        "mixture_prefix_lines_sha256": _lines_digest(
            [line for line, _ in mixture[: len(kept)]]
        ),
    }
    if original is not None:
        before = {row["id"]: line for line, row in original}
        changed = [line for line, row in kept if before.get(row["id"]) != line]
        still_old = sum(
            1
            for line, row in kept
            if before.get(row["id"]) != line
            and row["id"] in by_id
            and by_id[row["id"]][1] == before.get(row["id"])
        )
        result["original"] = {
            "rows": len(original),
            "kept_rows_changed_by_renumbering": len(changed),
            "mixture_rows_still_in_original_form": still_old,
            "ids_absent_from_original": sum(1 for i in kept_ids if i not in before),
        }
    result["passed"] = (
        len(base) == len(kept) > 0
        and result["excluded_ids_in_mixture"] == 0
        and result["kept_rows_form_mixture_prefix_in_order"]
        and line_identical == len(kept)
        and row_equal == len(kept)
        and repeats == 0
        and (
            original is None
            or result["original"]["mixture_rows_still_in_original_form"] == 0
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--mixture-sha256", required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--published-sha256", required=True)
    parser.add_argument(
        "--published-source", default=None, help="repository@revision:path"
    )
    parser.add_argument("--exclude-family", action="append", default=[])
    parser.add_argument("--original", type=Path)
    parser.add_argument("--original-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.original is None) != (args.original_sha256 is None):
        parser.error("--original and --original-sha256 go together")
    result = compare(
        read_lines(args.mixture, args.mixture_sha256),
        read_lines(args.published, args.published_sha256),
        set(args.exclude_family),
        read_lines(args.original, args.original_sha256) if args.original else None,
    )
    receipt = {
        "schema": SCHEMA,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mixture_sha256": args.mixture_sha256,
        "published_sha256": args.published_sha256,
        "published_source": args.published_source,
        "original_sha256": args.original_sha256,
        "excluded_families": sorted(args.exclude_family),
        "module_sha256": sha_file(Path(__file__)),
        **result,
    }
    write_json(args.output, receipt)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("published_kept_rows", "line_identical_rows", "passed")
            }
        )
    )
    sys.exit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
