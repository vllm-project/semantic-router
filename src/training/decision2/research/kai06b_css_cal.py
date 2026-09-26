"""Extract the exact private human CSS CAL rows as gold-free Kai prompts.

The parent data and noncommercial rights receipt are hash-pinned. Raw text and
gold remain on the research host; only aggregate calibration results may be
shared. No training, SELECT, pilot, or FINAL row is included in the output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from inference.run import digest
from training.data.build_kai06b_native_v1 import _read_parent, sha


def _encode(rows: list[dict]) -> bytes:
    return b"".join(
        (
            json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode()
        for row in rows
    )


def build(parent: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    partitions, _, rights = _read_parent(parent, "balanced-human5824")
    subset = [
        row for row in partitions["cal"] if row["source"].startswith("css_pilot:")
    ]
    source_counts = dict(sorted(Counter(row["source"] for row in subset).items()))
    if source_counts != {
        "css_pilot:discourse": 100,
        "css_pilot:implicit_hate": 100,
        "css_pilot:semeval_stance": 100,
    }:
        raise ValueError("Expected exact disjoint human CSS CAL buckets")
    prompts, gold = [], []
    for row in subset:
        if row["task_type"] != "choice":
            raise ValueError("CSS CAL contains a non-Choice row")
        labels = [option["key"] for option in row["options"]]
        if len(labels) != len(set(labels)) or not 0 <= row["label"] < len(labels):
            raise ValueError("Invalid CSS CAL option/label mapping")
        payload = {
            "state": row["state"],
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": row["instructions"],
                    "criteria": {
                        option["key"]: option["description"]
                        for option in row["options"]
                    },
                }
            },
        }
        prompts.append({"id": row["id"], **payload})
        gold.append(
            {
                "id": row["id"],
                "task": row["source"],
                "group_id": row["group_id"],
                "gold": labels[row["label"]],
                "labels": labels,
                "input_sha256": digest(payload),
            }
        )
    stage = output.with_name(output.name + ".pending")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir(parents=True, mode=0o700)
    (stage / "prompts.jsonl").write_bytes(_encode(prompts))
    (stage / "gold.jsonl").write_bytes(_encode(gold))
    reread = [json.loads(line) for line in (stage / "prompts.jsonl").open()]
    if any(
        digest({"state": prompt["state"], "questions": prompt["questions"]})
        != label["input_sha256"]
        for prompt, label in zip(reread, gold)
    ):
        raise ValueError("Serialized prompt hash differs from private CAL gold")
    manifest = {
        "schema_version": "kai06b-private-css-cal/1",
        "private_only": True,
        "raw_text_redistribution": False,
        "parent_cal_sha256": "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf",
        "parent_manifest_sha256": "869a94c0c74b9e80f2b60bf414eb7440cda17cbce1e61906621bbe206ea5aa9f",
        "parent_rights_evidence_sha256": rights["rights_evidence_sha256"],
        "holdout_source_rights": {
            source: rights["source_rights"][group]
            for source, group in rights["holdout_groups"]["cal"].items()
            if source in source_counts
        },
        "source_counts": source_counts,
        "parent_partition_isolation_checked": True,
        "outputs": {
            name: {"sha256": sha(stage / name), "rows": len(prompts)}
            for name in ("prompts.jsonl", "gold.jsonl")
        },
        "scope": "post-training source-matched calibration only; no model training or final selection",
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(stage, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.output)["outputs"], sort_keys=True))


if __name__ == "__main__":
    main()
