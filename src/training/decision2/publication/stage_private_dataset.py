"""Stage the frozen clean human splits for a private Hugging Face dataset repo.

Only the rights-clean control is eligible for row upload. Noncommercial pilot
manifests and scripts can be archived separately without their restricted raw
TweetEval or CSS text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

EXPECTED = {
    "rights_clean.train.jsonl": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select.jsonl": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal.jsonl": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    "rights_clean.manifest.json": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
}
SCOPE = "trained weights/modelcard only; no raw source rows, SELECT/CAL rows, or individual text predictions"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(source: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Frozen data must be a regular directory")
    for name, digest in EXPECTED.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Frozen clean v2 file differs: {name}")
    receipt = json.loads((source / "rights_clean.manifest.json").read_text())
    if (
        receipt.get("derivation_version") != "decision2-goemotions-human-v2/1"
        or receipt.get("publication_eligible") is not True
        or receipt.get("publication_scope") != SCOPE
    ):
        raise ValueError("Source manifest is not the audited clean v2 split")
    if set(receipt.get("outputs", {})) != set(EXPECTED) - {
        "rights_clean.manifest.json"
    }:
        raise ValueError("Source manifest does not bind exactly three partitions")
    for name in receipt["outputs"]:
        if receipt["outputs"][name]["sha256"] != EXPECTED[name]:
            raise ValueError(f"Split hash not bound by source manifest: {name}")
    roster = receipt.get("counts", {}).get("source", {})
    if (
        not roster
        or any("tweeteval" in key or "multi_nli" in key for key in roster)
        or roster.get("google_goemotions_official_train") != 2800
        or not receipt.get("source_rights")
        or not receipt.get("overlap_audits")
    ):
        raise ValueError("Private upload has excluded or unaccounted source rows")
    return receipt


def card(receipt: dict[str, Any]) -> str:
    train = receipt["outputs"]["rights_clean.train.jsonl"]["rows"]
    select = receipt["outputs"]["select.jsonl"]["rows"]
    cal = receipt["outputs"]["cal.jsonl"]["rows"]
    source_lines = "\n".join(
        f"| {item['source']} | {item['rows']} | {item.get('partition_scope', 'TRAIN')} | "
        f"{item['license']} | {item['evidence']} |"
        for item in receipt["source_rights"]
    )
    return f"""---
license: other
license_name: mixed-source-terms-see-card
language:
- en
- zh
task_categories:
- text-classification
pretty_name: Decision 2.0 rights-clean research splits
---

# Decision 2.0 rights-clean research splits

This repository is **private**. It stores the frozen v2 training, checkpoint
selection and calibration rows for Decision 2.0 research. Access is limited to
the project team. Do not make this repository public or redistribute the source
text without reviewing each upstream license and notice.

The three JSONL files contain {train:,} TRAIN, {select:,} SELECT and {cal:,}
CAL rows. `rights_clean.manifest.json` records exact SHA-256, source counts,
generation code identity and context isolation checks. The external GoEmotions
input was the **official TRAIN split only**, pinned to Google Research revision
`2adf640a14f11025ae5a9d0ec493b78530d276d3`; official dev/test were not used.

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `rights_clean.train.jsonl` | {train:,} | `{EXPECTED['rights_clean.train.jsonl']}` |
| `select.jsonl` | {select:,} | `{EXPECTED['select.jsonl']}` |
| `cal.jsonl` | {cal:,} | `{EXPECTED['cal.jsonl']}` |
| `rights_clean.manifest.json` | — | `{EXPECTED['rights_clean.manifest.json']}` |

## Sources and conditions

Google Research's [repository statement](https://github.com/google-research/google-research/)
licenses datasets there under CC BY 4.0, including the pinned GoEmotions
source. Other source rows retain their own terms and attribution.

| Source | Rows | Partition | Terms | Evidence |
| --- | ---: | --- | --- | --- |
{source_lines}

This private repository does **not** contain original TweetEval or MultiNLI
rows; all 3,600 TweetEval-origin and 267 MultiNLI-origin rows were excluded
from this control. A separate noncommercial pilot used those sources and is
documented by hashes/source links only, without uploading its restricted text.

## Interpretation

GoEmotions comments were chosen from filtered, single-label, non-neutral
annotations. A comment yields one four-option Choice and one binary Noul
question. Human emotion judgments are subjective; negative labels may omit
plausible secondary emotions. TRAIN uses 1,400 source comment groups, SELECT
200 and CAL 200, with whole-comment isolation. SELECT and CAL also retain 300
synthetic oracle rows each. Their mixture is a research design and does not
represent the original GoEmotions prevalence.

The v2 builder checked exact and approximate near-context overlap across all
three splits and against frozen synthetic DEV, CSS and pressure prompt contexts;
all recorded matches were zero. SimHash candidate search is approximate and
does not prove semantic independence. Some existing FLUTE TRAIN rows make CSS
FLUTE same-task supervised. Model initialization and its earlier training
lineage require a separate audit.

Use the exact source SHA-256 receipts, manifest and code revision when citing
or reproducing a run. Raw rows from this private repository must not be copied
into public model cards or the public research gist.
"""


def stage(source: Path, destination: Path) -> dict[str, Any]:
    receipt = verify(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, mode=0o700)
    for name in EXPECTED:
        shutil.copyfile(source / name, destination / name)
    (destination / "README.md").write_text(card(receipt), encoding="utf-8")
    os.chmod(destination / "README.md", 0o600)
    return {name: sha(destination / name) for name in (*EXPECTED, "README.md")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
