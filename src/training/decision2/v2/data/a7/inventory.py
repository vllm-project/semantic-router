"""Count-only census of Decision 1.0 training corpora for the A7 inventory.

Tolerates the raw 1.0 row formats (opaque option keys, source objects). For
each named file it reports SHA-256, rows, decision types, languages, Score
level counts, families by type, upstream lineage, soft-label fields and, when
a `v2.data.a7.lengths` file is given, token totals by type and language. It
also reports pairwise input-hash intersections, which expose replay lineage
between files. Output never contains text.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, digest
from v2.data.a7.build_a7 import Excluded, lineage

SCHEMA = "decision2.v2.a7.inventory.v1"
SOFT_FIELDS = (
    "target_probs",
    "teacher_probs",
    "soft_targets",
    "soft_label",
    "probabilities",
)


def _lineage(source: Any) -> str:
    try:
        return lineage(source)
    except (Excluded, ValueError, SyntaxError):
        return "unrecognized"


def native_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a Kai-native encoder row (state_text/question/target) onto the
    census fields; other rows pass through unchanged."""
    question = row.get("question")
    if "state_text" not in row or not isinstance(question, Mapping):
        return dict(row)
    candidates = (
        question.get("levels")
        or question.get("options")
        or question.get("candidates")
        or ([{}, {}] if str(question.get("type")).lower() == "noul" else [])
    )
    target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
    probabilities = target.get("probabilities")
    soft = (
        isinstance(probabilities, list)
        and bool(probabilities)
        and max(probabilities) < 1
    )
    if "probability" in target:
        soft = 0 < float(target["probability"]) < 1
    provenance = (
        row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
    )
    return {
        "id": row.get("id"),
        "task_type": str(question.get("type", "")).lower(),
        "language": row.get("language"),
        "family": row.get("source_family") or row.get("domain") or row.get("schema_id"),
        "source": row.get("source_id"),
        "group_id": row.get("component_id"),
        "options": [{"key": str(index)} for index in range(len(candidates))],
        "soft_target": soft,
        "label_origin": provenance.get("label_origin"),
    }


def census(
    rows: Sequence[Mapping[str, Any]], lengths: Mapping[str, Mapping[str, int]] | None
) -> tuple[dict[str, Any], set[str]]:
    types: collections.Counter[str] = collections.Counter()
    languages: collections.Counter[str] = collections.Counter()
    levels: collections.Counter[str] = collections.Counter()
    family_types: collections.Counter[str] = collections.Counter()
    lineages: collections.Counter[str] = collections.Counter()
    soft: collections.Counter[str] = collections.Counter()
    origins: collections.Counter[str] = collections.Counter()
    tokens: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    hashes: set[str] = set()
    duplicates = 0
    groups = set()
    for original in rows:
        row = native_view(original)
        if row.get("soft_target"):
            soft["soft_target"] += 1
        if row.get("label_origin"):
            origins[str(row["label_origin"])[:120]] += 1
        task_type = str(row.get("task_type"))
        types[task_type] += 1
        languages[str(row.get("language"))] += 1
        family_types[f"{row.get('family')}|{task_type}"] += 1
        lineages[_lineage(row.get("source"))] += 1
        groups.add(row.get("group_id"))
        if task_type == "score":
            levels[str(len(row.get("options") or []))] += 1
        for field in SOFT_FIELDS:
            if field in row:
                soft[field] += 1
        if all(field in row for field in INPUT_FIELDS):
            value = digest({field: row[field] for field in INPUT_FIELDS})
            duplicates += value in hashes
            hashes.add(value)
        if lengths is not None and row.get("id") in lengths:
            for name, count in lengths[row["id"]].items():
                if name != "id":
                    tokens[name][f"type:{task_type}"] += count
                    tokens[name][f"language:{row.get('language')}"] += count
                    tokens[name]["total"] += count
    report = {
        "rows": len(rows),
        "groups": len(groups),
        "duplicate_inputs": duplicates,
        "types": dict(sorted(types.items())),
        "languages": dict(sorted(languages.items())),
        "score_levels": dict(sorted(levels.items(), key=lambda item: int(item[0]))),
        "family_types": dict(sorted(family_types.items())),
        "lineages": dict(sorted(lineages.items())),
        "soft_label_fields": dict(sorted(soft.items())),
        "label_origins": dict(origins.most_common(20)),
    }
    if lengths is not None:
        report["tokens"] = {
            name: dict(sorted(counts.items())) for name, counts in tokens.items()
        }
    return report, hashes


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--lengths", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    lengths: dict[str, dict[str, int]] | None = None
    if args.lengths:
        lengths = {}
        for path in args.lengths:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    item = json.loads(line)
                    lengths[item["id"]] = item
    files: dict[str, Any] = {}
    hashes: dict[str, set[str]] = {}
    for value in args.file:
        name, path_text = value.split("=", 1)
        path = Path(path_text)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        scoped = None
        if lengths is not None:
            scoped = {
                row["id"]: lengths[f"{name}:{row['id']}"]
                for row in rows
                if f"{name}:{row['id']}" in lengths
            }
        report, file_hashes = census(rows, scoped)
        files[name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            **report,
        }
        hashes[name] = file_hashes
    names = list(files)
    intersections = {
        f"{left}&{right}": len(hashes[left] & hashes[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if hashes[left] & hashes[right]
    }
    payload = {"schema": SCHEMA, "files": files, "input_intersections": intersections}
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
        )
    print(json.dumps({name: files[name]["rows"] for name in names}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
