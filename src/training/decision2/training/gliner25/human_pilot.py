"""Freeze a group-aware, human-annotated TRAIN-only GLiNER source screen.

The native schema and 512-position admission check are identical to the
published-source adapter. No SELECT/CAL label or benchmark target enters the
training export. Run this builder only on an authorized experiment host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inference import gliner25

from training.gliner25.pilot import native_record
from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)

CONTRACT = "gliner25-human-head512-step32-v1"
SEED = "decision2-gliner25-human-head-20260927-v1"
MAX_POSITIONS = 512
EXPECTED_HASHES = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    "rights": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
}
GOEMOTIONS = "google_goemotions_official_train"
QUOTAS = {
    GOEMOTIONS: 128,  # paired Choice and Noul records per group
    "legacy:cosmos_qa": 80,
    "legacy:snli": 80,
    "legacy:squad2_answerability": 48,
    "css_flute_official_train": 48,
}
EXPECTED_ELIGIBLE = {
    GOEMOTIONS: 2800,
    "legacy:cosmos_qa": 448,
    "legacy:snli": 272,
    "legacy:squad2_answerability": 332,
    "css_flute_official_train": 120,
}


def _rank(*parts: str) -> str:
    return hashlib.sha256(":".join((SEED, *parts)).encode()).hexdigest()


def choose_human_rows(
    eligible: list[dict[str, Any]], quotas: dict[str, int] = QUOTAS
) -> list[dict[str, Any]]:
    """Draw complete GoEmotions pairs and one record per other source group."""
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in eligible:
        if row["source"] not in quotas or row["language"] != "en":
            raise ValueError("Human TRAIN pool contains unexpected source/language")
        by_source[row["source"]][row["group_id"]].append(row)

    selected: list[dict[str, Any]] = []
    for source, count in quotas.items():
        grouped = by_source[source]
        candidates = []
        for group, rows in grouped.items():
            if source == GOEMOTIONS:
                if len(rows) != 2 or {r["task_type"] for r in rows} != {
                    "choice",
                    "noul",
                }:
                    raise ValueError(
                        "GoEmotions group is not an intact Choice/Noul pair"
                    )
                candidates.append((group, sorted(rows, key=lambda r: r["id"])))
            else:
                one = min(rows, key=lambda r: (_rank(source, group, r["id"]), r["id"]))
                candidates.append((group, [one]))
        if len(candidates) < count:
            raise ValueError(f"Only {len(candidates)} admissible {source} groups")
        chosen = sorted(candidates, key=lambda item: (_rank(source, item[0]), item[0]))[
            :count
        ]
        for _, rows in chosen:
            selected.extend(rows)
    expected_rows = sum(
        quota * (2 if source == GOEMOTIONS else 1) for source, quota in quotas.items()
    )
    if (
        len(selected) != expected_rows
        or len({r["id"] for r in selected}) != expected_rows
    ):
        raise ValueError("Human-only screen contains duplicate or missing rows")
    return sorted(selected, key=lambda row: row["id"])


def build(
    *,
    source: Path,
    train: Path,
    select: Path,
    cal: Path,
    rights: Path,
    output: Path,
    manifest: Path,
) -> dict[str, Any]:
    if output.exists() or manifest.exists():
        raise FileExistsError("Pilot sample or manifest already exists")
    actual_hashes = {
        name: file_sha256(path)
        for name, path in (
            ("train", train),
            ("select", select),
            ("cal", cal),
            ("rights", rights),
        )
    }
    if actual_hashes != EXPECTED_HASHES:
        raise ValueError("Rights-clean source bytes changed")
    partitions = {
        name: load_partition(path, name)
        for name, path in (("train", train), ("select", select), ("cal", cal))
    }
    if {name: len(rows) for name, rows in partitions.items()} != {
        "train": 7455,
        "select": 700,
        "cal": 700,
    }:
        raise ValueError("Rights-clean source counts changed")
    check_partition_isolation(partitions)
    gliner25.verify_release(source, gliner25.REVISION)
    native = gliner25.load_native(source, "cpu")
    human_rows = [row for row in partitions["train"] if row["source"] in QUOTAS]
    if len(human_rows) != 3974 or any(row["language"] != "en" for row in human_rows):
        raise ValueError("Human TRAIN source inventory changed")
    eligible = []
    records = {}
    admitted_by_source: Counter[str] = Counter()
    excluded_by_source: Counter[str] = Counter()
    for row in human_rows:
        record, length = native_record(native, row)
        if length > MAX_POSITIONS:
            excluded_by_source[row["source"]] += 1
            continue
        eligible.append(row)
        records[row["id"]] = record
        admitted_by_source[row["source"]] += 1
    if dict(admitted_by_source) != EXPECTED_ELIGIBLE or dict(excluded_by_source) != {
        "legacy:squad2_answerability": 2
    }:
        raise ValueError("Human native-512 eligibility changed")
    chosen = choose_human_rows(eligible)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(
                json.dumps(records[row["id"]], ensure_ascii=False, sort_keys=True)
            )
            stream.write("\n")
    receipt = {
        "contract": CONTRACT,
        "seed": SEED,
        "source_revision": gliner25.REVISION,
        "source_weights_sha256": gliner25.MODEL_FILES["model.safetensors"],
        "library_commit": gliner25.LIBRARY_COMMIT,
        "source_data_sha256": actual_hashes,
        "human_rows": len(human_rows),
        "eligible_by_source": dict(sorted(admitted_by_source.items())),
        "overflow_by_source": dict(sorted(excluded_by_source.items())),
        "selected_rows": len(chosen),
        "selected_groups": len({(r["source"], r["group_id"]) for r in chosen}),
        "selected_by_source": dict(
            sorted(Counter(r["source"] for r in chosen).items())
        ),
        "selected_by_type": dict(
            sorted(Counter(r["task_type"] for r in chosen).items())
        ),
        "native_train_sha256": file_sha256(output),
        "max_positions": MAX_POSITIONS,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "train", "select", "cal", "rights", "output", "manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
