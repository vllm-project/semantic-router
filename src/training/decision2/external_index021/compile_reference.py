"""Rebuild compact, public-only fixtures from an exact Space snapshot.

Fetch the pinned Hugging Face Space JSON on an authorized SSH machine using
the HF CLI, then provide copies of its two public JSON files to this command.
No training/evaluation examples or credentials are written to the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .protocol import spec


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_files(
    index_path: Path, methodology_path: Path, *, check: bool = False
) -> None:
    target = Path(__file__).parent / "data"
    old = spec()
    if (
        _digest(index_path) != old["index_sha256"]
        or _digest(methodology_path) != old["methodology_sha256"]
    ):
        raise ValueError("snapshot differs from the pinned Space revision")
    index, method = json.loads(index_path.read_text()), json.loads(
        methodology_path.read_text()
    )
    panel = method["index"]
    areas = [
        {
            "id": a["id"],
            "weight": a["weight"],
            "benchmarks": index["suite"]["areas"][a["id"]],
        }
        for a in panel["areas"]
        if a["weight"] > 0
    ]
    protocol = {
        **old,
        "areas": areas,
        "gold_weight": panel["gold"]["weight"],
        "gold_ids": [b["id"] for b in panel["gold"]["benchmarks"]],
        "chance": {str(b["id"]): b["chance"] for b in index["suite"]["chance_levels"]},
        "not_in_index": [b["id"] for b in panel["not_in_index"]],
        "added_ids": index["suite"]["added"]["benchmarks"],
        "panel_id": panel["panel_id"],
    }
    selected_ids = {n for area in areas for n in area["benchmarks"]}
    rows = []
    for item in (index["jev"], *index["models"]):
        rows.append(
            {
                "name": item["name"],
                "scores": item["scores"],
                "areas": {
                    a["id"]: {k: a[k] for k in ("raw", "skill", "coverage")}
                    for a in item["categories"]
                    if a["id"] in {x["id"] for x in areas}
                },
                "benchmarks": {
                    bid: {
                        k: value[k]
                        for k in ("raw", "skill", "coverage", "random")
                        if k in value
                    }
                    for bid, value in item["benchmarks"].items()
                    if int(bid) in selected_ids
                },
            }
        )
    fixture = {"source_sha256": old["index_sha256"], "rows": rows}
    expected = {
        target / "protocol-021.json": json.dumps(protocol, indent=2, sort_keys=True)
        + "\n",
        target
        / "published-021-summary.json": json.dumps(fixture, separators=(",", ":"))
        + "\n",
    }
    for path, data in expected.items():
        if check:
            if path.read_text() != data:
                raise ValueError(f"compiled public fixture differs: {path.name}")
        else:
            path.write_text(data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("index", type=Path)
    p.add_argument("methodology", type=Path)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    compile_files(args.index, args.methodology, check=args.check)


if __name__ == "__main__":
    main()
