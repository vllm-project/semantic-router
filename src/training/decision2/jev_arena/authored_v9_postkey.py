"""Compare sealed v9 blind reviews with frozen private targets.

This operator runs only after the original and ablation reviews are sealed.
It writes aggregate counts and input hashes, never private prompt text or IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

FROZEN_SHA = {
    "prompts": "84514f8efa2dab34bf2a6f115c8e895969f7c5caf79c32120d3727e27b8c02ce",
    "ablations": "d66853149a4f6f85bf95cb1c0deacc6a58c7e800625d3e0d9a5cc762ed87ed49",
    "targets": "671f958dbd29365d859edfd3ccb1c22e31f5978656335cf9460514da673c9966",
    "proofs": "810db5678277f399f6c6dec574cce0f5e63ca1423492e849f6ac7269ca538425",
    "blind_original": "5c542aa48232ff579defc5f743f6eca5208b5e20509cb8cc36b5034192d4fb22",
    "blind_ablation": "2afa3e7851e26c05365c0d18cf0dd2598664a0a450b48ae89bc89ceb05e7df17",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def review_rows(path: Path, columns: int) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        fields = [cell.strip() for cell in line.strip("|").split("|")]
        if len(fields) == columns and fields[0].isdigit():
            rows.append(fields)
    return rows


def answer_value(kind: str, written: str) -> str | bool | int:
    if kind == "noul":
        mapping = {"Certified": True, "Not certified": False}
        if written not in mapping:
            raise ValueError("unrecognized Boolean blind answer")
        return mapping[written]
    if kind == "score":
        match = re.fullmatch(r"Grade ([0-9]+)", written)
        if not match:
            raise ValueError("unrecognized Score blind answer")
        return int(match.group(1))
    if kind == "choice":
        return written
    raise ValueError("unknown answer kind")


def compare(packet: Path) -> dict[str, Any]:
    paths = {
        "prompts": packet / "prompts.jsonl",
        "ablations": packet / "ablations.gold-free.jsonl",
        "targets": packet / "private/targets.jsonl",
        "proofs": packet / "private/proof_traces.jsonl",
        "blind_original": packet / "blind-original-review.md",
        "blind_ablation": packet / "blind-ablation-review.md",
    }
    for name, path in paths.items():
        if sha(path) != FROZEN_SHA[name]:
            raise ValueError(f"{name} SHA-256 differs from frozen review")
    prompts = load_jsonl(paths["prompts"])
    targets = load_jsonl(paths["targets"])
    proofs = load_jsonl(paths["proofs"])
    ablations = load_jsonl(paths["ablations"])
    original = review_rows(paths["blind_original"], 5)
    deletion = review_rows(paths["blind_ablation"], 4)
    if (
        len(prompts),
        len(targets),
        len(proofs),
        len(original),
        len(ablations),
        len(deletion),
    ) != (
        12,
        12,
        12,
        12,
        36,
        36,
    ):
        raise ValueError("original/ablation counts differ from frozen candidate")
    target_by_id = {row["id"]: row for row in targets}
    if (
        len(target_by_id) != 12
        or {row["id"] for row in prompts} != set(target_by_id)
        or {row["id"] for row in proofs} != set(target_by_id)
    ):
        raise ValueError("prompt/target/proof identity mismatch")
    matches = 0
    by_kind = {kind: {"n": 0, "matched": 0} for kind in ("choice", "noul", "score")}
    for position, (index, item_id, written, _, _) in enumerate(original, start=1):
        if int(index) != position or item_id != prompts[position - 1]["id"]:
            raise ValueError("original review does not align to frozen packet")
        target = target_by_id[item_id]
        kind = target["kind"]
        expected = target["answer"][kind]
        matched = answer_value(kind, written) == expected
        matches += int(matched)
        by_kind[kind]["n"] += 1
        by_kind[kind]["matched"] += int(matched)
    statuses = {"clean": 0, "weak_hint": 0, "leak": 0}
    for position, (fields, source) in enumerate(
        zip(deletion, ablations, strict=True), start=1
    ):
        index, parent, omitted, verdict = fields
        if (
            int(index) != position
            or parent != source["parent_id"]
            or omitted != source["omitted_source"]
        ):
            raise ValueError("ablation review does not align to frozen packet")
        if verdict.startswith("Clean."):
            statuses["clean"] += 1
        elif verdict.startswith("Unprovable, with a weak narrative cue."):
            statuses["weak_hint"] += 1
        elif verdict.startswith("Fails clean deletion."):
            statuses["leak"] += 1
        else:
            raise ValueError("unrecognized ablation verdict")
    return {
        "status": "dev_editorial_diagnostic_not_release_benchmark",
        "input_sha256": dict(FROZEN_SHA),
        "original_items": 12,
        "original_matches": matches,
        "by_kind": by_kind,
        "ablation_items": 36,
        "ablation_verdicts": statuses,
        "gate": "BLOCK_FOR_RELEASE_BENCH",
        "reason": "one direct source-deletion leak plus answer-position and Boolean-balance shortcuts",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
