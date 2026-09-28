"""Shared row construction for human-source arms (A1, A3, A5, A6h).

Rows follow the strict training contract of ``training.model.data``. Option
order is a deterministic per-row rotation; held-out groups (AHO) are the
``sha256(group_id) % 10 == 0`` slice and carry ``split=select``.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row

NOUL_OPTIONS = {
    "en": ("No", "Yes"),
    "zh": ("否", "是"),
    "ko": ("아니요", "예"),
    "ja": ("いいえ", "はい"),
    "de": ("Nein", "Ja"),
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def noul_options(language: str) -> list[dict[str, str]]:
    no, yes = NOUL_OPTIONS[language]
    return [{"key": "false", "description": no}, {"key": "true", "description": yes}]


def score_options(descriptions: Sequence[str]) -> list[dict[str, str]]:
    if not 2 <= len(descriptions) <= 10:
        raise ValueError("Score needs 2..10 levels")
    return [
        {"key": str(index), "description": text}
        for index, text in enumerate(descriptions)
    ]


def choice_options(
    descriptions: Sequence[str], keys: Sequence[str] | None = None
) -> list[dict[str, str]]:
    keys = (
        list(keys)
        if keys is not None
        else [f"o{index + 1}" for index in range(len(descriptions))]
    )
    if len(keys) != len(descriptions) or len(set(keys)) != len(keys):
        raise ValueError("Choice keys must be unique and match descriptions")
    return [{"key": key, "description": text} for key, text in zip(keys, descriptions)]


def rotate(
    options: list[dict[str, Any]], label: int, seed: str
) -> tuple[list[dict[str, Any]], int]:
    """Rotate Choice options by a seed-derived offset; Score/Noul must not call this."""
    offset = int(sha(seed), 16) % len(options)
    rotated = options[offset:] + options[:offset]
    return rotated, (label - offset) % len(options)


def make_row(
    *,
    arm: str,
    source: str,
    family: str,
    task_type: str,
    language: str,
    group_key: str,
    local_id: str,
    state: Any,
    instructions: Any,
    options: list[dict[str, Any]],
    label: int,
    render_template: str,
    audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": f"{arm}-{family}-{sha(source + '|' + local_id)[:20]}",
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": label,
        "task_type": task_type,
        "family": family,
        "group_id": f"{arm}:{source}:{sha(group_key)[:20]}",
        "language": language,
        "split": "train",
        "source": source,
        "evaluation_role": "train",
        "render_template": render_template,
        "audit_metadata": {"source_local_id": local_id, **(audit or {})},
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return validate_row(row, "train")


def is_aho(group_id: str) -> bool:
    return int(sha(group_id), 16) % 10 == 0


def cap_groups(
    rows: Iterable[dict[str, Any]], cap_rows: int, seed: str
) -> list[dict[str, Any]]:
    """Keep whole groups in seed-hash order until adding one would exceed the cap."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    chosen: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda key: sha(f"{seed}:{key}")):
        if len(chosen) + len(groups[group]) > cap_rows:
            continue
        chosen.extend(groups[group])
        if len(chosen) == cap_rows:
            break
    return chosen


def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def by(field: str) -> dict[str, int]:
        return dict(
            sorted(collections.Counter(str(row[field]) for row in rows).items())
        )

    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "task_type": by("task_type"),
        "language": by("language"),
        "source": by("source"),
        "family": by("family"),
        "label_by_family": {
            family: dict(
                sorted(
                    collections.Counter(
                        row["label"] for row in rows if row["family"] == family
                    ).items()
                )
            )
            for family in sorted({row["family"] for row in rows})
        },
    }


def _write_new(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def write_arm(
    rows: Sequence[dict[str, Any]], out_dir: Path, arm: str, build: Mapping[str, Any]
) -> dict[str, Any]:
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row id in arm")
    train = [row for row in rows if not is_aho(row["group_id"])]
    aho = []
    for row in rows:
        if is_aho(row["group_id"]):
            held = dict(row, split="select", evaluation_role="select")
            aho.append(validate_row(held, "select"))
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {
        "arm": arm,
        "build": dict(build),
        "train": {
            **counts(train),
            "sha256": _write_new(
                out_dir / f"{arm}.train.jsonl",
                "".join(
                    canonical(r) + "\n" for r in sorted(train, key=lambda r: r["id"])
                ).encode(),
            ),
        },
        "aho": {
            **counts(aho),
            "sha256": _write_new(
                out_dir / f"{arm}.aho.jsonl",
                "".join(
                    canonical(r) + "\n" for r in sorted(aho, key=lambda r: r["id"])
                ).encode(),
            ),
        },
    }
    _write_new(
        out_dir / f"{arm}.build.json",
        (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode(),
    )
    return manifest
