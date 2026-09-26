"""Freeze a 100-item, three-panel, gold-free competitor smoke packet.

Selection uses only DEV family/group, CSS task, and public tier metadata. It
never reads target values into the emitted model-visible packet.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from small_decision_competitors import json_bytes, read_rows, sha_file

PANEL_HASHES = {
    "dev_prompts": "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    "dev_gold": "c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc",
    "css_prompts": "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
    "css_gold": "9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391",
    "public_prompts": "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
    "public_targets": "abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f",
}
DEV_GROUP_QUOTA = {
    "attribute_gate": 3,
    "rule_precedence": 3,
    "set_reconciliation": 2,
    "transition_table": 2,
}


def read_meta(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def rank(value: str) -> str:
    return hashlib.sha256(("small-competitor-smoke-v1/" + value).encode()).hexdigest()


def select(inputs: dict[str, Path], output: Path) -> dict:
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(output)
    for key, path in inputs.items():
        if sha_file(path) != PANEL_HASHES[key]:
            raise ValueError(f"{key} differs from the frozen panel")
    dev, css, public = [
        read_rows(inputs[f"{name}_prompts"]) for name in ("dev", "css", "public")
    ]
    dev_meta = read_meta(inputs["dev_gold"])
    css_meta = read_meta(inputs["css_gold"])
    public_meta = read_meta(inputs["public_targets"])
    if (len(dev), len(css), len(public)) != (1600, 1430, 231):
        raise ValueError("panel cardinality differs")
    for prompts, meta in ((dev, dev_meta), (css, css_meta), (public, public_meta)):
        if [row["id"] for row in prompts] != [row["id"] for row in meta]:
            raise ValueError("prompt/metadata order differs")

    groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in dev_meta:
        groups[row["family"]].add(row["group_id"])
    selected_groups = {
        family: set(sorted(groups[family], key=rank)[:quota])
        for family, quota in DEV_GROUP_QUOTA.items()
    }
    dev_ids = {
        row["id"]
        for row in dev_meta
        if row["group_id"] in selected_groups[row["family"]]
    }
    if len(dev_ids) != 40:
        raise ValueError("DEV group selection was not 40 rows")
    css_tasks: dict[str, list[str]] = collections.defaultdict(list)
    for row in css_meta:
        css_tasks[row["task"]].append(row["id"])
    if len(css_tasks) != 3:
        raise ValueError("CSS pilot should have exactly three tasks")
    css_ids = {
        item_id for ids in css_tasks.values() for item_id in sorted(ids, key=rank)[:10]
    }
    tiers: dict[str, list[str]] = collections.defaultdict(list)
    for row in public_meta:
        tiers[row["tier"]].append(row["id"])
    if set(tiers) != {"easy", "standard", "hard"}:
        raise ValueError("public tier inventory differs")
    public_ids = {
        item_id for ids in tiers.values() for item_id in sorted(ids, key=rank)[:10]
    }
    if (len(css_ids), len(public_ids)) != (30, 30):
        raise ValueError("CSS or public stratum count differs")
    chosen = (
        [row for row in dev if row["id"] in dev_ids]
        + [row for row in css if row["id"] in css_ids]
        + [row for row in public if row["id"] in public_ids]
    )
    if len(chosen) != 100 or len({row["id"] for row in chosen}) != 100:
        raise ValueError("smoke IDs are missing or duplicate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        for row in chosen:
            target.write(json_bytes(row))
    manifest = {
        "selection": "small-competitor-smoke-v1: 10 DEV groups, 10 per CSS task, 10 per public tier",
        "panel_hashes": PANEL_HASHES,
        "prompt_sha256": sha_file(output),
        "ids_sha256": hashlib.sha256(
            json_bytes([row["id"] for row in chosen])
        ).hexdigest(),
        "counts": {"dev": 40, "css": 30, "public": 30},
    }
    output.with_suffix(".manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in PANEL_HASHES:
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {key: getattr(args, key) for key in PANEL_HASHES}
    print(json.dumps(select(inputs, args.output)))


if __name__ == "__main__":
    main()
