"""Node-local copy of a house embedding manifest: same roles and SHA-256, local paths.

    python3 -m v2.data.dq.embed_manifest --source M.json --source-sha256 HEX \
        [--map OLD=NEW ...] [--extend ROLE=PATH ...] --out OUT.json --receipt R.json

An entry keeps its path when the file there has the pinned SHA-256; otherwise ``--map`` must
name a local file with that SHA-256. ``--extend`` appends roles with their hashes. Every role
of the source manifest is kept with its pinned hash, so the copy relaxes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2.data.dq.blind_review import file_sha256, write_json


def _pairs(values: list[str]) -> dict[str, str]:
    out = {}
    for value in values:
        left, sep, right = value.partition("=")
        if not sep or not left or not right:
            raise ValueError(f"expected X=Y, got {value!r}")
        if left in out:
            raise ValueError(f"repeated {left}")
        out[left] = right
    return out


def localize(
    entries: list[dict[str, Any]], mapping: dict[str, str], extend: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out, same, moved = [], 0, []
    unused = dict(mapping)
    for entry in entries:
        path = Path(entry["path"])
        if path.is_file() and file_sha256(path) == entry["sha256"]:
            out.append(dict(entry))
            same += 1
            continue
        target = unused.pop(entry["path"], None)
        if target is None:
            raise ValueError(
                f"{entry['role']}: {path} missing or changed and not mapped"
            )
        if file_sha256(Path(target)) != entry["sha256"]:
            raise ValueError(
                f"{entry['role']}: {target} does not have the pinned sha256"
            )
        out.append({**entry, "path": target})
        moved.append({"role": entry["role"], "from": entry["path"], "to": target})
    if unused:
        raise ValueError(f"unused --map entries: {sorted(unused)}")
    roles = {entry["role"] for entry in out}
    added = []
    for role, path in extend.items():
        if role in roles:
            raise ValueError(f"--extend role {role} already present")
        digest = file_sha256(Path(path))
        out.append({"role": role, "path": path, "sha256": digest})
        added.append({"role": role, "path": path, "sha256": digest})
    receipt = {
        "schema": "dev2-dq-embed-manifest/1",
        "source_roles": len(entries),
        "same_path": same,
        "remapped": moved,
        "extended": added,
        "roles": len(out),
    }
    return out, receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--map", action="append", default=[])
    parser.add_argument("--extend", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    actual = file_sha256(args.source)
    if actual != args.source_sha256:
        parser.error(f"source manifest sha256 {actual} != {args.source_sha256}")
    entries = json.loads(args.source.read_text(encoding="utf-8"))
    out, receipt = localize(entries, _pairs(args.map), _pairs(args.extend))
    receipt["source_sha256"] = actual
    receipt["out_sha256"] = write_json(args.out, out)
    write_json(args.receipt, receipt)
    print(json.dumps({k: receipt[k] for k in ("roles", "same_path", "out_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
