"""Shared row construction, slices and writers for the v2 (Milestone 2) arms.

Group ids are ``m2:<namespace>:<hash>`` and never contain the arm, so one
upstream item (a question shared by TyDi QA and MIRACL, or a HotpotQA question
used by two families) always lands in the same slice. Slices: AHO when
``sha256(group_id) % 10 == 0`` (split ``select``, diagnostic), SHO when
``sha256("sho-v2:" + group_id) % 50 == 0`` among the rest (sealed, private),
TRAIN otherwise.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row
from v2.data.sources.common import choice_options, noul_options, score_options, sha

SLICES = ("train", "aho", "sho")
ABSTAIN = "The paragraphs do not give enough information to answer."

__all__ = [
    "ABSTAIN",
    "SLICES",
    "cap_groups",
    "choice_options",
    "counts",
    "group_id",
    "make_row",
    "noul_options",
    "read_jsonl",
    "rotate",
    "row_id",
    "score_options",
    "sha",
    "slice_of",
    "write_arm",
]


def group_id(namespace: str, key: str) -> str:
    return f"m2:{namespace}:{sha(key)[:24]}"


def row_id(family: str, source: str, local_id: str) -> str:
    return f"m2-{family}-{sha(source + '|' + local_id)[:24]}"


def slice_of(group: str) -> str:
    if int(sha(group), 16) % 10 == 0:
        return "aho"
    if int(sha("sho-v2:" + group), 16) % 50 == 0:
        return "sho"
    return "train"


def rotate(
    options: list[dict[str, Any]], label: int, seed: str
) -> tuple[list[dict[str, Any]], int]:
    """Rotate Choice options by a seed-derived offset (never Score or Noul)."""
    offset = int(sha(seed), 16) % len(options)
    return options[offset:] + options[:offset], (label - offset) % len(options)


def make_row(
    *,
    source: str,
    family: str,
    task_type: str,
    language: str,
    namespace: str,
    group_key: str,
    local_id: str,
    state: Any,
    instructions: Any,
    options: list[dict[str, Any]],
    label: int,
    template: str,
    audit: Mapping[str, Any] | None = None,
    rotate_choice: bool = False,
) -> dict[str, Any]:
    ident = row_id(family, source, local_id)
    if rotate_choice:
        if task_type != "choice":
            raise ValueError("only Choice options may be rotated")
        options, label = rotate(options, label, f"m2-v1:{ident}")
        if all(re.fullmatch(r"o\d+", option["key"]) for option in options):
            options = [
                dict(option, key=f"o{position}")
                for position, option in enumerate(options, 1)
            ]
    row = {
        "id": ident,
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": label,
        "task_type": task_type,
        "family": family,
        "group_id": group_id(namespace, group_key),
        "language": language,
        "split": "train",
        "source": source,
        "evaluation_role": "train",
        "render_template": template,
        "audit_metadata": {"source_local_id": local_id, **(audit or {})},
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return validate_row(row, "train")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                yield value


def cap_groups(
    rows: Iterable[dict[str, Any]], cap_rows: int, seed: str
) -> list[dict[str, Any]]:
    """Whole groups in seed-hash order until adding one would exceed the cap."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    chosen: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda group: sha(f"{seed}:{group}")):
        if len(chosen) + len(groups[key]) > cap_rows:
            continue
        chosen.extend(groups[key])
        if len(chosen) == cap_rows:
            break
    return chosen


def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def by(field: str) -> dict[str, int]:
        return dict(
            sorted(collections.Counter(str(row[field]) for row in rows).items())
        )

    score = [row for row in rows if row["task_type"] == "score"]
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
                        str(row["label"]) for row in rows if row["family"] == family
                    ).items()
                )
            )
            for family in sorted({row["family"] for row in rows})
        },
        "score_levels": dict(
            sorted(
                collections.Counter(str(len(row["options"])) for row in score).items()
            )
        ),
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
        raise ValueError(f"{arm}: duplicate row id")
    parts: dict[str, list[dict[str, Any]]] = {name: [] for name in SLICES}
    for row in rows:
        name = slice_of(row["group_id"])
        if name != "train":
            row = validate_row(
                dict(row, split="select", evaluation_role="select"), "select"
            )
        parts[name].append(row)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest: dict[str, Any] = {"arm": arm, "build": dict(build)}
    for name in SLICES:
        ordered = sorted(parts[name], key=lambda row: row["id"])
        data = "".join(canonical(row) + "\n" for row in ordered).encode("utf-8")
        manifest[name] = {
            **counts(ordered),
            "sha256": _write_new(out_dir / f"{arm}.{name}.jsonl", data),
        }
    _write_new(
        out_dir / f"{arm}.build.json",
        (
            json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    )
    return manifest
