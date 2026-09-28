"""Equal-row substitution mixtures: A0 with a fixed share of whole groups replaced by an arm.

The control trains on A0; a data arm trains on A0 minus randomly chosen whole
source groups (at least ``fraction`` of A0 rows) plus randomly chosen whole
arm groups until the added rows reach the removed rows. Row counts and hence
optimizer updates match the control to within one group; token totals are
reported by the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MIX_VERSION = "decision2-9b-substitution-mix-v1"


def canonical_line(row: dict[str, Any]) -> str:
    return json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def content_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["id"]):
        digest.update((canonical_line(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def groups_of(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["group_id"]].append(index)
    return groups


def substitute(
    a0: list[dict[str, Any]], arm: list[dict[str, Any]], fraction: float, seed: int
):
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if set(r["group_id"] for r in a0) & set(r["group_id"] for r in arm):
        raise ValueError("A0 and the arm share a group id")
    if set(r["id"] for r in a0) & set(r["id"] for r in arm):
        raise ValueError("A0 and the arm share a row id")
    rng = random.Random(seed)
    target = round(fraction * len(a0))
    a0_groups = groups_of(a0)
    order = sorted(a0_groups)
    rng.shuffle(order)
    removed: set[int] = set()
    for group in order:
        if len(removed) >= target:
            break
        removed.update(a0_groups[group])
    arm_groups = groups_of(arm)
    arm_order = sorted(arm_groups)
    rng.shuffle(arm_order)
    added: list[int] = []
    for group in arm_order:
        if len(added) >= len(removed):
            break
        added.extend(arm_groups[group])
    if len(added) < len(removed):
        raise ValueError("Arm has fewer rows than the removed share")
    kept = [row for index, row in enumerate(a0) if index not in removed]
    new = sorted((arm[i] for i in added), key=lambda r: r["id"])
    return kept + new, {
        "removed_rows": len(removed),
        "removed_groups": sum(1 for g in a0_groups.values() if g[0] in removed),
        "removed_by_type": dict(Counter(a0[i]["task_type"] for i in removed)),
        "added_rows": len(added),
        "added_by_type": dict(Counter(arm[i]["task_type"] for i in added)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--a0-file-sha256", required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--arm-content-sha256", required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from training.model.data import file_sha256, load_partition

    if file_sha256(args.a0) != args.a0_file_sha256:
        raise SystemExit("A0 file digest differs from the pinned partition")
    a0 = load_partition(args.a0, "train")
    arm = load_partition(args.arm, "train")
    if content_sha256(arm) != args.arm_content_sha256:
        raise SystemExit(f"{args.arm_name} content digest differs from the registry")
    rows, stats = substitute(a0, arm, args.fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_line(row) + "\n")
    load_partition(args.output, "train")
    manifest = {
        "version": MIX_VERSION,
        "a0_file_sha256": args.a0_file_sha256,
        "arm": args.arm_name,
        "arm_file_sha256": file_sha256(args.arm),
        "arm_content_sha256": args.arm_content_sha256,
        "fraction": args.fraction,
        "seed": args.seed,
        "rows": len(rows),
        "by_type": dict(Counter(r["task_type"] for r in rows)),
        **stats,
        "output_file_sha256": file_sha256(args.output),
        "output_content_sha256": content_sha256(rows),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
