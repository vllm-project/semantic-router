"""Deterministic A0-derived arms: A0s (minus FLUTE), A0p (option order) and RP-v1.

All three read the frozen rights-clean v2 TRAIN and never its labels' sources
beyond what the rows carry. RP-v1 is written twice: as training rows (gold,
for hard replay) and as gold-free native System One prompts (for teachers).
Native prompts keep criteria insertion order; sorting keys would reorder
Choice options and change what a teacher sees.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row

A0_SHA256 = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
FLUTE_SOURCE = "css_flute_official_train"
A0P_VERSION = "a0p-v1"
RP_VERSION = "decision2-rp-v1"
RP_QUOTAS = {"choice": 640, "noul": 512, "score": 384}
POSITIONAL_WORDS = re.compile(
    r"\b(first|second|third|fourth|last|final|previous|next)\s+"
    r"(option|choice|candidate|answer)s?\b",
    re.IGNORECASE,
)
POSITIONAL_MARKS = re.compile(
    r"\b[Oo]ption\s+[A-H]\b|\b[Cc]hoice\s+[A-H]\b|\([A-H]\)|(?<!\w)[A-H]\)"
    r"|第[一二三四五六七八九十]+(?:个|项|选项)|最后一(?:个|项)|选项[A-H一二三四五六七八]"
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_a0(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if sha_bytes(data) != A0_SHA256:
        raise ValueError("A0 TRAIN SHA-256 changed")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    for row in rows:
        validate_row(row, "train")
    return rows


def canonical_lines(rows: list[dict[str, Any]]) -> bytes:
    ordered = sorted(rows, key=lambda row: row["id"])
    return "".join(canonical(row) + "\n" for row in ordered).encode("utf-8")


def with_hash(row: dict[str, Any]) -> dict[str, Any]:
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return row


def a0s(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["source"] != FLUTE_SOURCE]


def _text(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def derangement(size: int, seed: str) -> list[int]:
    """Uniform-ish fixed-point-free permutation from a SHA-256 stream."""
    counter = 0
    while True:
        stream = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        keys = [
            hashlib.sha256(stream + index.to_bytes(2, "big")).digest()
            for index in range(size)
        ]
        order = sorted(range(size), key=lambda index: keys[index])
        if all(order[position] != position for position in range(size)):
            return order
        counter += 1


def a0p(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out, skipped = [], collections.Counter()
    for row in rows:
        if row["task_type"] != "choice":
            continue
        if len(row["options"]) < 3:
            skipped["fewer_than_three_options"] += 1
            continue
        text = _text(row["instructions"]) + " " + _text(row["options"])
        if POSITIONAL_WORDS.search(text) or POSITIONAL_MARKS.search(text):
            skipped["positional_reference"] += 1
            continue
        order = derangement(len(row["options"]), A0P_VERSION + row["id"])
        new = copy.deepcopy(row)
        new["id"] = row["id"] + ":p1"
        new["options"] = [copy.deepcopy(row["options"][index]) for index in order]
        new["label"] = order.index(row["label"])
        meta = dict(new["audit_metadata"])
        meta["a0p"] = {"version": A0P_VERSION, "source_id": row["id"], "order": order}
        new["audit_metadata"] = meta
        out.append(with_hash(new))
    return out, dict(sorted(skipped.items()))


def replay_prompt_set(
    rows: list[dict[str, Any]], quotas: dict[str, int] = RP_QUOTAS
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    groups = {
        group: members
        for group, members in groups.items()
        if all(row["source"] != FLUTE_SOURCE for row in members)
    }
    ranked = sorted(
        groups, key=lambda group: sha_bytes(f"{RP_VERSION}:{group}".encode())
    )
    counts: collections.Counter[str] = collections.Counter()
    chosen: list[dict[str, Any]] = []
    for group in ranked:
        need = collections.Counter(row["task_type"] for row in groups[group])
        if any(counts[kind] + need[kind] > quotas.get(kind, 0) for kind in need):
            continue
        counts.update(need)
        chosen.extend(groups[group])
        if all(counts[kind] == quota for kind, quota in quotas.items()):
            break
    if any(counts[kind] != quota for kind, quota in quotas.items()):
        raise ValueError(f"RP-v1 quotas unmet: {dict(counts)}")
    return sorted(chosen, key=lambda row: row["id"])


def native_prompt(row: dict[str, Any]) -> dict[str, Any]:
    if row["task_type"] == "score":
        criteria: Any = [option["description"] for option in row["options"]]
    else:
        criteria = {option["key"]: option["description"] for option in row["options"]}
    return {
        "id": row["id"],
        "state": row["state"],
        "questions": {
            "decision": {
                "type": row["task_type"],
                "instructions": row["instructions"],
                "criteria": criteria,
            }
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def by(field: str) -> dict[str, int]:
        return dict(sorted(collections.Counter(row[field] for row in rows).items()))

    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "task_type": by("task_type"),
        "language": by("language"),
        "source": by("source"),
    }


def _write_new(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return sha_bytes(data)


def build(a0_path: Path, out_dir: Path) -> dict[str, Any]:
    rows = load_a0(a0_path)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest: dict[str, Any] = {
        "schema": "decision2-a0-variants/v1",
        "a0_sha256": A0_SHA256,
    }
    variant_s = a0s(rows)
    manifest["A0s"] = {
        **summarize(variant_s),
        "content_sha256": _write_new(
            out_dir / "a0s.train.jsonl", canonical_lines(variant_s)
        ),
        "removed_source": FLUTE_SOURCE,
    }
    variant_p, skipped = a0p(rows)
    for row in variant_p:
        validate_row(row, "train")
    manifest["A0p"] = {
        **summarize(variant_p),
        "content_sha256": _write_new(
            out_dir / "a0p.train.jsonl", canonical_lines(variant_p)
        ),
        "skipped_choice_rows": skipped,
        "gold_position": dict(
            sorted(collections.Counter(row["label"] for row in variant_p).items())
        ),
    }
    replay = replay_prompt_set(rows)
    prompts = "".join(
        json.dumps(native_prompt(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in replay
    ).encode("utf-8")
    manifest["RP-v1"] = {
        **summarize(replay),
        "content_sha256": _write_new(
            out_dir / "rp-v1.train.jsonl", canonical_lines(replay)
        ),
        "prompts_sha256": _write_new(out_dir / "rp-v1.prompts.jsonl", prompts),
        "quotas": RP_QUOTAS,
        "score_levels": dict(
            sorted(
                collections.Counter(
                    len(row["options"]) for row in replay if row["task_type"] == "score"
                ).items()
            )
        ),
    }
    _write_new(
        out_dir / "a0-variants.manifest.json",
        json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8"),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-train", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.a0_train, args.out_dir), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
