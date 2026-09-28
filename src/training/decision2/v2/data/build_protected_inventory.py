"""Assemble a hash-pinned, input-only protected inventory (PI-v2).

The private spec lists ``{role, origin, sha256, project}`` entries. Each origin
file is verified, optionally projected to ``{id, state, questions|instructions,
options}`` (dropping answer-bearing and metadata fields), copied into a private
output directory and listed in ``manifest.json`` as ``{role, path, sha256}``.
Byte-identical origins are kept once under the first role that lists them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ANSWER_KEYS = frozenset(
    {
        "gold",
        "label",
        "labels",
        "answer",
        "answers",
        "target",
        "targets",
        "teacher_probs",
        "target_probs",
        "correct",
        "solution",
    }
)
INPUT_KEYS = ("id", "state", "questions", "instructions", "options")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_answers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_answers(item)
            for key, item in value.items()
            if key not in ANSWER_KEYS
        }
    if isinstance(value, list):
        return [_strip_answers(item) for item in value]
    return value


def project_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row.get("id"), str) or "state" not in row:
        raise ValueError("Protected row lacks id or state")
    return {key: _strip_answers(row[key]) for key in INPUT_KEYS if key in row}


def _contains_answer_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(ANSWER_KEYS & set(value)) or any(
            _contains_answer_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_answer_key(item) for item in value)
    return False


def build(spec_path: Path, out_dir: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to overwrite")
    out_dir.mkdir(parents=True, mode=0o700)
    entries: list[dict[str, Any]] = []
    public: list[dict[str, Any]] = []
    seen_origin: dict[str, str] = {}
    roles: set[str] = set()
    for item in spec:
        role, origin = item["role"], Path(item["origin"])
        if role in roles:
            raise ValueError(f"Duplicate role {role}")
        roles.add(role)
        origin_sha = sha_file(origin)
        if origin_sha != item["sha256"]:
            raise ValueError(f"{role}: origin SHA-256 changed")
        if origin_sha in seen_origin:
            public.append(
                {"role": role, "duplicate_of": seen_origin[origin_sha], "rows": 0}
            )
            continue
        seen_origin[origin_sha] = role
        rows = [
            json.loads(line) for line in origin.open(encoding="utf-8") if line.strip()
        ]
        if item.get("project"):
            rows = [project_row(row) for row in rows]
        if any(_contains_answer_key(row) for row in rows):
            raise ValueError(f"{role}: answer-bearing key present; set project")
        target = out_dir / f"{role}.jsonl"
        with target.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
        os.chmod(target, 0o600)
        sha = sha_file(target)
        entries.append({"role": role, "path": str(target), "sha256": sha})
        public.append(
            {
                "role": role,
                "rows": len(rows),
                "sha256": sha,
                "origin_sha256": origin_sha,
                "projected": bool(item.get("project")),
            }
        )
    manifest = out_dir / "manifest.json"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(entries, stream, indent=1, sort_keys=True)
    os.chmod(manifest, 0o600)
    receipt = {
        "schema": "decision2-protected-inventory/v2",
        "manifest_sha256": sha_file(manifest),
        "roles": public,
        "rows_total": sum(entry["rows"] for entry in public),
    }
    receipt_path = out_dir / "public-receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=1, sort_keys=True)
    os.chmod(receipt_path, 0o600)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.spec, args.out_dir)
    print(json.dumps(receipt, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
