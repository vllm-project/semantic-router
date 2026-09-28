"""Third disjoint Kev TRAIN-only teacher screen; no student or release scoring.

The original v1 screen stopped on rounded output validation. The v2 screen
stopped before inference because its container hid the linked Git metadata.
This screen uses new TRAIN groups, the frozen v2 numeric rule and an explicit
CPU provenance/cache preflight before any GPU invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from inference.kev import (
    KEV_BASE_ID,
    KEV_BASE_REVISION,
    KEV_MODEL_REVISION,
    load_native,
    model_fingerprint,
    verify_provenance,
)
from inference.run import local_revision
from training.model.data import file_sha256, load_partition

from .eikos_teacher_train_pilot import KINDS, PER_KIND, roster_sha256, write_once
from .eikos_teacher_train_pilot import roster as v1_roster
from .kev_teacher_train_pilot_v2 import aggregate
from .kev_teacher_train_pilot_v2 import roster as v2_roster


def roster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select new independent groups, excluding both prior screen rosters."""
    excluded = {row["group_id"] for row in v1_roster(rows)}
    excluded.update(row["group_id"] for row in v2_roster(rows))
    seen = set(excluded)
    selected = []
    for kind in KINDS:
        eligible = sorted(
            (row for row in rows if row["task_type"] == kind),
            key=lambda row: hashlib.sha256(
                f"kev-v3\x00{kind}\x00{row['group_id']}\x00{row['id']}".encode()
            ).digest(),
        )
        count = 0
        for row in eligible:
            group = row["group_id"]
            if group in seen:
                continue
            selected.append(row)
            seen.add(group)
            count += 1
            if count == PER_KIND:
                break
        if count != PER_KIND:
            raise ValueError(f"Too few fresh independent TRAIN groups for {kind}")
    return selected


def preflight(model_path: Path, source_path: Path, revision: str) -> dict[str, Any]:
    """Prove linked source and the complete pinned base cache without GPU use."""
    if revision != KEV_MODEL_REVISION or not local_revision(model_path, revision):
        raise ValueError("Kev local download does not attest its pinned revision")
    provenance = verify_provenance(model_path, source_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download

    base = Path(
        snapshot_download(
            KEV_BASE_ID,
            revision=KEV_BASE_REVISION,
            local_files_only=True,
        )
    )
    index = json.loads((base / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards or any(not (base / shard).is_file() for shard in shards):
        raise ValueError("Pinned official base weight cache is incomplete")
    if not (base / "tokenizer.json").is_file():
        raise ValueError("Pinned official base tokenizer cache is incomplete")
    return {
        "revision_attested": True,
        "publisher_source_files_verified": len(provenance["source_hashes"]),
        "model_fingerprint": model_fingerprint(model_path),
        "official_base_revision": KEV_BASE_REVISION,
        "official_base_shards_cached": len(shards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--revision", default=KEV_MODEL_REVISION)
    parser.add_argument("--roster-sha256")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if file_sha256(args.train) != args.train_sha256:
        raise ValueError("Frozen TRAIN bytes differ")
    train = load_partition(args.train, "train")
    selected = roster(train)
    identity = roster_sha256(selected)
    old_groups = {
        row["group_id"]
        for previous in (v1_roster(train), v2_roster(train))
        for row in previous
    }
    overlap = len(old_groups & {row["group_id"] for row in selected})
    if overlap or len(selected) != len({row["group_id"] for row in selected}):
        raise ValueError("TRAIN roster is not independent and disjoint")
    common = {
        "rows": len(selected),
        "groups": len(selected),
        "per_type": PER_KIND,
        "prior_group_overlap": overlap,
        "roster_sha256": identity,
        "script_sha256": file_sha256(__file__),
    }
    if args.dry_run:
        print(json.dumps(common, sort_keys=True))
        return
    if (
        args.model_path is None
        or args.source_path is None
        or args.roster_sha256 != identity
    ):
        raise ValueError("Pinned model/source and locked roster are required")
    checked = preflight(args.model_path, args.source_path, args.revision)
    if args.preflight:
        print(json.dumps({**common, **checked}, sort_keys=True))
        return
    if args.output is None:
        raise ValueError("Private aggregate output is required")
    model, decide, runtime = load_native(args.model_path, args.source_path, "cuda:0")
    assert model is not None
    stats = aggregate(selected, decide)
    payload = {
        "schema": "decision2-kev-teacher-train-screen/3",
        "source": f"jaredpalmer/kev-4b@{KEV_MODEL_REVISION}",
        "train_sha256": args.train_sha256,
        "roster_sha256": identity,
        "script_sha256": common["script_sha256"],
        "sampled_rows": len(selected),
        "sampled_groups": len(selected),
        "model_fingerprint": checked["model_fingerprint"],
        "runtime": runtime,
        "by_type": stats,
    }
    write_once(args.output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
