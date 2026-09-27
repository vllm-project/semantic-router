"""Read-only, gold-free 4B clean-v2 versus frozen release-prompt audit.

Only TRAIN and three prompt-only JSONL files are accepted. This audit checks
observable IDs and contexts; approximate near matching cannot establish
semantic disjointness or reconstruct upstream pretraining exposure.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted

EXPECTED_SHA = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "rights_manifest": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
    "typed_final": "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
    "css15": "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    "public231": "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
}
EXPECTED_N = {"train": 7455, "typed_final": 1600, "css15": 6547, "public231": 231}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path, *, prompt: bool) -> list[dict[str, Any]]:
    rows = []
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not isinstance(row.get("state"), str)
                or row["id"] in ids
            ):
                raise ValueError(f"{path.name}:{number}: invalid or repeated ID/state")
            if prompt:
                if set(row) != {"id", "state", "questions"}:
                    raise ValueError(f"{path.name}:{number}: not a gold-free prompt")
            elif not isinstance(row.get("source"), str):
                raise ValueError(f"{path.name}:{number}: missing TRAIN source")
            ids.add(row["id"])
            rows.append(row)
    return rows


def audit_panel(
    train: list[dict[str, Any]], prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Match raw/normalized state exactly, then frozen approximate near rule."""
    eval_raw: dict[str, list[int]] = collections.defaultdict(list)
    eval_norm: dict[str, list[int]] = collections.defaultdict(list)
    eval_text = []
    eval_bits = []
    bands: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for i, row in enumerate(prompts):
        raw, norm = targeted.text_hashes(row["state"])
        eval_raw[raw].append(i)
        eval_norm[norm].append(i)
        value = pilot._near_text(targeted.context_rows([row])[0])
        bits = pilot._simhash(value)
        eval_text.append(value)
        eval_bits.append(bits)
        for band in range(8):
            bands[(band, (bits >> (8 * band)) & 255)].append(i)
    result: dict[str, Any] = {
        "same_row_ids": sorted({r["id"] for r in train} & {r["id"] for r in prompts}),
        "exact_raw": [],
        "exact_normalized": [],
        "near": [],
        "near_candidates_examined": 0,
    }
    for row in train:
        raw, norm = targeted.text_hashes(row["state"])
        for i in eval_raw.get(raw, ()):
            result["exact_raw"].append([row["id"], prompts[i]["id"], row["source"]])
        for i in eval_norm.get(norm, ()):
            result["exact_normalized"].append(
                [row["id"], prompts[i]["id"], row["source"]]
            )
        value = pilot._near_text(targeted.context_rows([row])[0])
        bits = pilot._simhash(value)
        candidates = set()
        for band in range(8):
            candidates.update(bands.get((band, (bits >> (8 * band)) & 255), ()))
        result["near_candidates_examined"] += len(candidates)
        for i in sorted(candidates):
            other = eval_text[i]
            if abs(len(value) - len(other)) > 0.08 * max(len(value), len(other)):
                continue
            if (bits ^ eval_bits[i]).bit_count() > 8:
                continue
            ratio = difflib.SequenceMatcher(None, value, other).ratio()
            if ratio >= 0.94:
                result["near"].append(
                    [row["id"], prompts[i]["id"], row["source"], round(ratio, 6)]
                )
    result["counts"] = {
        key: len(result[key])
        for key in ("same_row_ids", "exact_raw", "exact_normalized", "near")
    }
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    inputs = {
        "train": args.train,
        "rights_manifest": args.rights_manifest,
        "typed_final": args.typed_final,
        "css15": args.css15,
        "public231": args.public231,
    }
    actual_sha = {name: digest(path) for name, path in inputs.items()}
    if actual_sha != EXPECTED_SHA:
        raise ValueError("A frozen TRAIN, manifest or prompt SHA-256 changed")
    train = read_jsonl(args.train, prompt=False)
    panels = {
        name: read_jsonl(inputs[name], prompt=True)
        for name in ("typed_final", "css15", "public231")
    }
    for name, rows in {"train": train, **panels}.items():
        if len(rows) != EXPECTED_N[name]:
            raise ValueError(f"{name}: unexpected row count")
    rights = json.loads(args.rights_manifest.read_text(encoding="utf-8"))
    if (
        rights.get("schema_version") != "decision2-rights-clean-splits/1"
        or rights.get("publication_eligible") is not True
        or sum(
            entry["rows"]
            for entry in rights.get("source_rights", [])
            if entry.get("partition_scope", "TRAIN") == "TRAIN"
        )
        != len(train)
    ):
        raise ValueError("Rights manifest does not bind the frozen TRAIN")
    provenance = json.loads(args.package_provenance.read_text(encoding="utf-8"))
    if (
        provenance.get("rights_manifest_sha256") != actual_sha["rights_manifest"]
        or provenance.get("training_data_sha256", {}).get("train")
        != actual_sha["train"]
        or provenance.get("publication_eligible") is not True
    ):
        raise ValueError("Candidate package lacks exact rights/data binding")
    flute_train = {
        row["id"].removeprefix("css_train/flute/")
        for row in train
        if row["source"] == "css_flute_official_train"
        and row["id"].startswith("css_train/flute/")
    }
    flute_eval = {
        row["id"].removeprefix("css/flute/")
        for row in panels["css15"]
        if row["id"].startswith("css/flute/")
    }
    report = {
        "schema_version": "decision2-release4b-cleanv2-overlap/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_code_sha256": digest(Path(__file__)),
        "input_sha256": actual_sha,
        "package_provenance_sha256": digest(args.package_provenance),
        "rows": {"train": len(train), **{k: len(v) for k, v in panels.items()}},
        "train_source_counts": dict(
            sorted(collections.Counter(r["source"] for r in train).items())
        ),
        "rights_source_rows": sum(
            x["rows"]
            for x in rights["source_rights"]
            if x.get("partition_scope", "TRAIN") == "TRAIN"
        ),
        "rights_licenses": [
            {k: entry[k] for k in ("source", "rows", "license")}
            for entry in rights["source_rights"]
        ],
        "rights_conditions": rights["publication_conditions"],
        "css_flute_same_task_source": {
            "train_source_rows": len(flute_train),
            "eval_items": len(flute_eval),
            "shared_original_ids": sorted(flute_train & flute_eval),
        },
        "panels": {name: audit_panel(train, rows) for name, rows in panels.items()},
        "method": {
            "exact": "state raw SHA-256 and NFKC/casefold/whitespace normalized SHA-256",
            "near": "context-only, 8-band 64-bit SimHash candidates, Hamming<=8, relative length<=8%, SequenceMatcher>=.94",
            "limits": "Approximate near search can miss paraphrases; source/task family overlap and upstream base pretraining cannot be excluded by context hashes.",
            "gold_or_scores_read": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output.parent, 0o700)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(args.output, 0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train",
        "rights_manifest",
        "typed_final",
        "css15",
        "public231",
        "package_provenance",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "panel_counts": {
                    name: audit["counts"] for name, audit in result["panels"].items()
                },
                "css_flute_original_id_overlap": len(
                    result["css_flute_same_task_source"]["shared_original_ids"]
                ),
                "output_sha256": digest(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
