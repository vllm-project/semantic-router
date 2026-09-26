"""Audit legacy TRAIN vs CSS15 gold-free contexts with the frozen near rule.

No CSS gold file is an input. Per-task counts use the same eight SimHash bands,
64-bit Hamming <= 8, length delta <= 8%, and SequenceMatcher >= 0.94 as
``build_pilot.near_duplicates`` on context-only rows.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from transfer import build as transfer

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted

LEGACY_SHA256 = "77d4ac13b945f67cc38a213a40085f7391ff8e5399599aa005b2a6c84d6f16c7"
COMBINED_SHA256 = "4e82651181fd4f9b11e82718275a7370b82cbe810f20bbe87db9786cfdf888ad"
CSS_PROMPTS_SHA256 = "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6"
THRESHOLDS = {
    "context_only": True,
    "simhash_bits": 64,
    "simhash_bands": 8,
    "hamming_distance_max": 8,
    "relative_length_delta_max": 0.08,
    "sequence_matcher_ratio_min": 0.94,
    "candidate_search": "approximate; an unmatched pair is not proof of dissimilarity",
}


def task_for(item: dict[str, Any]) -> str:
    parts = item["id"].split("/", 2)
    if (
        len(parts) != 3
        or parts[0] != "css"
        or parts[1] not in transfer.EVALUATION_TASKS
    ):
        raise ValueError("CSS prompt ID is not part of the 15-task evaluation")
    return parts[1]


def hash_ids(ids: set[str]) -> str:
    return hashlib.sha256(pilot.canonical(sorted(ids)).encode()).hexdigest()


def audit(rows: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    expected = set(transfer.EVALUATION_TASKS)
    if {task_for(item) for item in prompts} != expected:
        raise ValueError("CSS evaluation task inventory changed")
    prompt_text = [
        pilot._near_text(targeted.context_rows([item])[0]) for item in prompts
    ]
    prompt_bits = [pilot._simhash(value) for value in prompt_text]
    bands: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    exact_raw: dict[str, set[int]] = collections.defaultdict(set)
    exact_norm: dict[str, set[int]] = collections.defaultdict(set)
    for index, (item, bits) in enumerate(zip(prompts, prompt_bits, strict=True)):
        for band in range(8):
            bands[(band, (bits >> (band * 8)) & 255)].append(index)
        raw, norm = targeted.text_hashes(item["state"])
        exact_raw[raw].add(index)
        exact_norm[norm].add(index)
    by_task = {
        task: {
            "exact_raw_train_ids": set(),
            "exact_raw_css_ids": set(),
            "exact_norm_train_ids": set(),
            "exact_norm_css_ids": set(),
            "near_train_ids": set(),
            "near_css_ids": set(),
            "near_pairs": 0,
            "examples": [],
        }
        for task in transfer.EVALUATION_TASKS
    }
    for number, row in enumerate(rows, 1):
        row_id = row["id"]
        raw, norm = targeted.text_hashes(row["state"])
        for index in exact_raw.get(raw, ()):
            stat = by_task[task_for(prompts[index])]
            stat["exact_raw_train_ids"].add(row_id)
            stat["exact_raw_css_ids"].add(prompts[index]["id"])
        for index in exact_norm.get(norm, ()):
            stat = by_task[task_for(prompts[index])]
            stat["exact_norm_train_ids"].add(row_id)
            stat["exact_norm_css_ids"].add(prompts[index]["id"])
        value = pilot._near_text(targeted.context_rows([row])[0])
        bits = pilot._simhash(value)
        candidates = set()
        for band in range(8):
            candidates.update(bands.get((band, (bits >> (band * 8)) & 255), ()))
        for index in candidates:
            other = prompt_text[index]
            if abs(len(value) - len(other)) > 0.08 * max(len(value), len(other)):
                continue
            if (bits ^ prompt_bits[index]).bit_count() > 8:
                continue
            ratio = difflib.SequenceMatcher(None, value, other).ratio()
            if ratio < 0.94:
                continue
            stat = by_task[task_for(prompts[index])]
            stat["near_pairs"] += 1
            stat["near_train_ids"].add(row_id)
            stat["near_css_ids"].add(prompts[index]["id"])
            if len(stat["examples"]) < 10:
                stat["examples"].append(
                    {
                        "train_id": row_id,
                        "css_id": prompts[index]["id"],
                        "similarity": round(ratio, 5),
                    }
                )
        if number % 10000 == 0:
            print(
                f"audited {number}/{len(rows)} TRAIN contexts",
                file=sys.stderr,
                flush=True,
            )
    result = {}
    for task, stat in by_task.items():
        result[task] = {
            "css_items": sum(task_for(item) == task for item in prompts),
            "exact_raw_train_rows": len(stat["exact_raw_train_ids"]),
            "exact_raw_css_items": len(stat["exact_raw_css_ids"]),
            "exact_normalized_train_rows": len(stat["exact_norm_train_ids"]),
            "exact_normalized_css_items": len(stat["exact_norm_css_ids"]),
            "near_pairs": stat["near_pairs"],
            "near_train_rows": len(stat["near_train_ids"]),
            "near_css_items": len(stat["near_css_ids"]),
            "near_css_ids": sorted(stat["near_css_ids"]),
            "near_train_id_commitment": hash_ids(stat["near_train_ids"]),
            "near_css_id_commitment": hash_ids(stat["near_css_ids"]),
            "examples": stat["examples"],
        }
    return result


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = []
        for line in stream:
            item = json.loads(line)
            rows.append({"id": item["id"], "state": item["state"]})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path.name}: repeated TRAIN IDs")
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.mask_output.exists():
        raise FileExistsError("Audit report or mask output already exists")
    for name, path, expected in (
        ("legacy", args.legacy_train, LEGACY_SHA256),
        ("combined", args.combined_train, COMBINED_SHA256),
        ("css_prompts", args.css_prompts, CSS_PROMPTS_SHA256),
    ):
        if pilot.sha_file(path) != expected:
            raise ValueError(f"{name} differs from frozen SHA-256")
    legacy = load_rows(args.legacy_train)
    combined = load_rows(args.combined_train)
    prompts = load_rows(args.css_prompts)
    if (len(legacy), len(combined), len(prompts)) != (47842, 6000, 6547):
        raise ValueError("Expected source/prompt cardinalities changed")
    report = {
        "schema_version": "decision2-css15-train-near-audit/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "sources": {
            "published_decision1_train": {
                "file": args.legacy_train.name,
                "sha256": LEGACY_SHA256,
                "rows": len(legacy),
            },
            "combined6k_train": {
                "file": args.combined_train.name,
                "sha256": COMBINED_SHA256,
                "rows": len(combined),
            },
        },
        "gold_free_css_prompts": {
            "file": args.css_prompts.name,
            "sha256": CSS_PROMPTS_SHA256,
            "rows": len(prompts),
        },
        "method": pilot.near_duplicates([], [targeted.context_rows(prompts)[0]])[
            "method"
        ],
        "thresholds": THRESHOLDS,
        "by_source": {
            "published_decision1_train": audit(legacy, prompts),
            "combined6k_train": audit(combined, prompts),
        },
        "interpretation": "These are context-level similarity counts, not labels, performance or proof of training contamination; inspect matched source provenance before using CSS15 transfer scores as strictly unseen-text evidence.",
        "sealed_gold": "not read",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pilot._atomic_write(
        args.output, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    )
    source_masks = {
        source: {
            task: values["near_css_ids"]
            for task, values in tasks.items()
            if values["near_css_items"]
        }
        for source, tasks in report["by_source"].items()
    }
    all_css_ids = {row["id"] for row in prompts}
    union = set().union(
        *(set(ids) for tasks in source_masks.values() for ids in tasks.values())
    )
    mask = {
        "schema_version": "decision2-css15-gold-free-overlap-mask/2",
        "css_prompts_sha256": CSS_PROMPTS_SHA256,
        "train_source_sha256": {
            "published_decision1_train": LEGACY_SHA256,
            "combined6k_train": COMBINED_SHA256,
        },
        "method": report["method"],
        "thresholds": THRESHOLDS,
        "gold": "not read",
        "near_css_ids_by_train_source": source_masks,
        "known_overlap_union_css_ids": sorted(union),
        "shared_clean_css_ids": sorted(all_css_ids - union),
        "counts": {
            "full": len(all_css_ids),
            "known_overlap_union": len(union),
            "shared_clean": len(all_css_ids - union),
        },
        "use": "For all models, report full 6,547-item CSS15 scores and scores on the same shared_clean_css_ids. The source-specific masks remain available for diagnosis. A similarity mask is not proof that gold labels leaked; third-party model training data are unknown, and Decision 2.0 models initialized from 1.0 inherit its known-overlap risk.",
    }
    args.mask_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pilot._atomic_write(
        args.mask_output,
        (json.dumps(mask, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-train", type=Path, required=True)
    parser.add_argument("--combined-train", type=Path, required=True)
    parser.add_argument("--css-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args)
    print(
        pilot.canonical(
            {
                source: {
                    task: values["near_css_items"] for task, values in tasks.items()
                }
                for source, tasks in report["by_source"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
