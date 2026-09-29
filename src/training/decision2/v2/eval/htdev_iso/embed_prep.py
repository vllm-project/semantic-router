"""Inputs for the HT-DEV source-level embedding scan (`v2.data.embed_scan`).

    python3 -m v2.eval.htdev_iso.embed_prep --corpora <training-corpora.json> \
        --protected-rows <protected-all-splits.jsonl> --source KEY ... \
        --families <embed-families.json> --out <dir> --workers N

Training side: every jsonl/jsonl.gz row of the corpus manifest whose provenance fields
(`independence.PROVENANCE`) match one of the family patterns becomes a protected row
{id, state}; `state` is the row's `state` (else prompt/input/text/messages/conversations),
deduplicated by normalised text. Source side: one candidates file per key (<key>.jsonl)
with {id: "<task>|<source_item_id>", group_id: id, state: joined overlap_texts},
deduplicated by normalised text within the key. Writes <out>/pi/manifest.json (the
embed_scan protected-inventory format) and <out>/prep-receipt.json (counts per family
pattern, provenance value and key; no text).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from v2.data.embed_scan import windows
from v2.eval.sealed.independence import PROVENANCE, strings
from v2.eval.sealed.schema import normalized

STATE_KEYS = ("state", "prompt", "input", "text", "messages", "conversations")
_PATTERNS: list[tuple[str, re.Pattern[str]]] = []


def _init(patterns: dict[str, str]) -> None:
    _PATTERNS[:] = [
        (name, re.compile(p, re.IGNORECASE)) for name, p in patterns.items()
    ]


def _state(row: dict[str, Any]) -> str:
    for key in STATE_KEYS:
        if key in row and row[key]:
            value = row[key]
            return value if isinstance(value, str) else "\n".join(strings(value))
    return ""


def _scan_file(path: str) -> tuple[Counter, list[tuple[str, str]]]:
    counts: Counter = Counter()
    rows: list[tuple[str, str]] = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream):
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            fields = " ".join(
                str(v) for k in PROVENANCE if k in row for v in strings(row[k])
            )
            for name, pattern in _PATTERNS:
                match = pattern.search(fields)
                if match:
                    counts[(name, match.group(0).casefold())] += 1
                    state = _state(row)
                    if state:
                        rows.append((f"{path}#{number}", state))
                    break
    return counts, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpora", type=Path, required=True)
    parser.add_argument("--protected-rows", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)
    os.umask(0o077)
    args.out.mkdir(parents=True, mode=0o700)
    (args.out / "pi").mkdir(mode=0o700)
    families = json.loads(args.families.read_text(encoding="utf-8"))["patterns"]
    labels = json.loads(args.corpora.read_text(encoding="utf-8"))["labels"]
    files = sorted(
        {
            f["path"]
            for entry in labels.values()
            for f in entry["files"]
            if f["path"].endswith((".jsonl", ".jsonl.gz"))
        }
    )
    matched: Counter = Counter()
    seen: set[str] = set()
    protected_windows = 0
    protected = args.out / "pi" / "training-social.jsonl"
    with Pool(
        args.workers, initializer=_init, initargs=(families,)
    ) as pool, protected.open("x", encoding="utf-8") as stream:
        for counts, rows in pool.imap_unordered(_scan_file, files, chunksize=2):
            matched.update(counts)
            for identity, state in rows:
                key = hashlib.sha256(normalized(state).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                protected_windows += len(windows(state))
                stream.write(json.dumps({"id": identity, "state": state}) + "\n")
    digest = hashlib.sha256(protected.read_bytes()).hexdigest()
    (args.out / "pi" / "manifest.json").write_text(
        json.dumps(
            [{"role": "training-social", "path": str(protected), "sha256": digest}]
        )
    )
    wanted = set(args.source)
    per_key: Counter = Counter()
    key_windows: Counter = Counter()
    handles = {k: (args.out / f"{k}.jsonl").open("x", encoding="utf-8") for k in wanted}
    seen_key: set[tuple[str, str]] = set()
    with args.protected_rows.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["source"] not in wanted:
                continue
            state = "\n".join(row["overlap_texts"])
            text_key = (
                row["source"],
                hashlib.sha256(normalized(state).encode()).hexdigest(),
            )
            if text_key in seen_key:
                continue
            seen_key.add(text_key)
            identity = f"{row['task']}|{row['source_item_id']}"
            handles[row["source"]].write(
                json.dumps({"id": identity, "group_id": identity, "state": state})
                + "\n"
            )
            per_key[row["source"]] += 1
            key_windows[row["source"]] += len(windows(state))
    for handle in handles.values():
        handle.close()
    receipt = {
        "schema": "htdev-iso-embed-prep/1",
        "training_files_scanned": len(files),
        "families": families,
        "matched_rows": {f"{n}:{v}": c for (n, v), c in sorted(matched.items())},
        "protected_rows_dedup": len(seen),
        "protected_sha256": digest,
        "candidate_rows_dedup": dict(sorted(per_key.items())),
        "candidate_windows": dict(sorted(key_windows.items())),
        "protected_windows": protected_windows,
    }
    (args.out / "prep-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "matched_rows"}))
    print(json.dumps(receipt["matched_rows"])[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
