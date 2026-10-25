"""Append reviewed natural requests without re-splitting a frozen task corpus."""

import argparse
import hashlib
import json
from pathlib import Path

from ..sequence_repair.corpus import write_corpus


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {
        name: read_jsonl(args.corpus / f"{name}.jsonl")
        for name in ["train", "dev", "test"]
    }
    contract = json.loads((args.corpus / "contract.json").read_text())
    excluded = []
    for path in args.annotations:
        for row in read_jsonl(path):
            if row.get("reviewer") != "assistant" or not row.get("review_rationale"):
                raise ValueError(
                    "Every request needs explicit review provenance and rationale"
                )
            if row["label"] is None:
                if not row.get("exclusion_reason"):
                    raise ValueError("Excluded records need an explicit reason")
                excluded.append({"id": row["id"], "reason": row["exclusion_reason"]})
                continue
            if row["label"] not in contract["label2id"] or row["split"] not in splits:
                raise ValueError("Invalid reviewed label or frozen partition")
            splits[row["split"]].append(row)
    provenance = {
        "parent_manifest_sha256": hashlib.sha256(
            (args.corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "annotations": [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in args.annotations
        ],
        "review": "Human-authored source requests; task labels individually assistant-reviewed before model predictions, not human-expert annotations",
        "aya_revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        "aya_license": "Apache-2.0",
        "dolly_revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "dolly_license": "CC-BY-SA-3.0",
        "dolly_attribution": "Copyright 2023 Databricks, Inc.; Wikipedia editors and contributors for source passages",
        "historical_exposure": "Earlier FactCheck training used Dolly creative-writing/brainstorming/summarization instructions. No historical row receipt exists; this is not claimed unseen by the old baseline or base pretraining.",
        "heldout_scope": "Source-group-disjoint from this Vela run's task training; ambiguous requests and identified cross-split source/template duplicates excluded before predictions",
        "policy": "Supplied-context extraction, textual transformations, self-contained computation, subjective/creative requests do not require an external fact check. Specific external facts and factual explanations do. A supplied passage must actually support the requested answer.",
        "excluded": excluded,
    }
    print(json.dumps(write_corpus(args.output, splits, contract, provenance), indent=2))


if __name__ == "__main__":
    main()
