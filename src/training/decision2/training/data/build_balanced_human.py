"""Build a typed-balanced human training candidate from frozen TRAIN sources.

Adds 450 Noul and 150 Score rows from combined6k and the same counts from
targeted2k to the fixed 4,624-row human arm. Whole source groups are selected
after gold-free CSS/DEV and SELECT/CAL context exclusion.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

from training.data import build_human_anchor as human_anchor
from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

BASE_SHA256 = "3a994e58d7084f4b1afea46cecac0ceb3a00ecd010263296315b2c8ddd058266"
BASE_MANIFEST_SHA256 = (
    "de747b03b58eb28491111f424df4ffc097c3eef615d03b4998450e5e01f463af"
)
COMBINED_SHA256 = human_anchor.COMBINED_6K_SHA256
TARGETED_SHA256 = "59d40112ab4d9d0688b3b121212ad5cc36329fe18170839485c9906544f0dee0"
TARGETED_MANIFEST_SHA256 = (
    "f26cc542d76f3b9e0237046edadff865d5a7115cdc1e2f6052cdf35ab7b6866a"
)
SELECT_SHA256 = human_anchor.SELECT_SHA256
CAL_SHA256 = human_anchor.CAL_SHA256
DEV_SHA256 = human_anchor.DEV_PROMPTS_SHA256
CSS_PILOT_SHA256 = human_anchor.CSS_PILOT_PROMPTS_SHA256
CSS_EVALUATION_SHA256 = human_anchor.CSS_EVALUATION_PROMPTS_SHA256
QUOTAS = {
    "combined6k": {"noul": 450, "score": 150},
    "targeted2k": {"noul": 450, "score": 150},
}


def filter_groups(
    source: list[dict[str, Any]], protected: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    protected_ids = {row["id"] for row in protected}
    protected_groups = {
        row["group_id"] for row in protected if isinstance(row.get("group_id"), str)
    }
    protected_inputs = {
        row["input_sha256"]
        for row in protected
        if isinstance(row.get("input_sha256"), str)
    }
    hashes = {targeted.text_hashes(row["state"]) for row in protected}
    raw_hashes, norm_hashes = {pair[0] for pair in hashes}, {pair[1] for pair in hashes}
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in source:
        by_group[row["group_id"]].append(row)
    rejected = collections.Counter()
    candidates: dict[str, list[dict[str, Any]]] = {}
    for group, rows in by_group.items():
        if len({row["task_type"] for row in rows}) != 1 or rows[0]["task_type"] not in (
            "noul",
            "score",
        ):
            rejected["other_or_mixed_type"] += len(rows)
            continue
        if any(
            row["id"] in protected_ids
            or row["group_id"] in protected_groups
            or row["input_sha256"] in protected_inputs
            for row in rows
        ):
            rejected["id_group_or_input"] += len(rows)
            continue
        if any(
            (pair := targeted.text_hashes(row["state"]))[0] in raw_hashes
            or pair[1] in norm_hashes
            for row in rows
        ):
            rejected["exact_context"] += len(rows)
            continue
        candidates[group] = rows
    near = pilot.near_duplicates(
        targeted.context_rows([row for rows in candidates.values() for row in rows]),
        targeted.context_rows(protected),
        collect_left_ids=True,
    )
    bad_ids = set(near.pop("left_ids"))
    bad_groups = {
        group
        for group, rows in candidates.items()
        if any(row["id"] in bad_ids for row in rows)
    }
    for group in bad_groups:
        rejected["near_context"] += len(candidates[group])
        del candidates[group]
    return candidates, {
        "source_rows": len(source),
        "candidate_groups": len(candidates),
        "candidate_rows": sum(map(len, candidates.values())),
        "rejected_rows_by_reason": dict(sorted(rejected.items())),
        "near_audit_before_quarantine": near,
    }


def choose_groups(
    candidates: dict[str, list[dict[str, Any]]],
    task_type: str,
    quota: int,
    seed: str,
    source_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    groups = [
        group for group, rows in candidates.items() if rows[0]["task_type"] == task_type
    ]
    groups.sort(
        key=lambda group: (
            pilot.sha_bytes(f"{seed}\0{source_name}\0{task_type}\0{group}".encode()),
            group,
        )
    )
    chosen = []
    remaining = quota
    for group in groups:
        if len(candidates[group]) <= remaining:
            chosen.append(group)
            remaining -= len(candidates[group])
            if not remaining:
                break
    if remaining:
        raise ValueError(
            f"Insufficient complete safe {source_name}/{task_type} groups for {quota}"
        )
    rows = [row for group in chosen for row in candidates[group]]
    if len(rows) != quota:
        raise AssertionError("Group selection split source group")
    return rows, chosen


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frozen = {
        "base": (args.base_train, BASE_SHA256),
        "base_manifest": (args.base_manifest, BASE_MANIFEST_SHA256),
        "combined6k": (args.combined_train, COMBINED_SHA256),
        "targeted2k": (args.targeted_train, TARGETED_SHA256),
        "targeted_manifest": (args.targeted_manifest, TARGETED_MANIFEST_SHA256),
        "select": (args.select_file, SELECT_SHA256),
        "hard_cal": (args.cal_file, CAL_SHA256),
        "synthetic_dev_prompts": (args.dev_prompts, DEV_SHA256),
        "css_pilot_prompts": (args.css_pilot_prompts, CSS_PILOT_SHA256),
        "css_evaluation_prompts": (args.css_evaluation_prompts, CSS_EVALUATION_SHA256),
    }
    for role, (path, expected) in frozen.items():
        if pilot.sha_file(path) != expected:
            raise ValueError(f"{role} differs from frozen SHA-256")
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    targeted_manifest = json.loads(args.targeted_manifest.read_text(encoding="utf-8"))
    if (
        base_manifest["outputs"]["human_anchor_4624.train.jsonl"]["sha256"]
        != BASE_SHA256
        or targeted_manifest["output"]["sha256"] != TARGETED_SHA256
    ):
        raise ValueError("Source manifests do not bind the frozen input files")
    base = load_partition(args.base_train, "train")
    combined = load_partition(args.combined_train, "train")
    targeted_rows = load_partition(args.targeted_train, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    if (len(base), len(combined), len(targeted_rows), len(select), len(cal)) != (
        4624,
        6000,
        2000,
        600,
        900,
    ):
        raise ValueError("Frozen source cardinality changed")
    references = []
    context_receipts = {}
    for name, path, expected in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_3task_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
        (
            "css_15task_evaluation",
            args.css_evaluation_prompts,
            "css-evaluation.prompts.jsonl",
        ),
    ):
        rows, receipt = targeted.load_context_reference(path, expected_name=expected)
        references.extend(rows)
        context_receipts[name] = receipt
    protected = [*base, *select, *cal, *references]
    combined_candidates, combined_audit = filter_groups(combined, protected)
    chosen_combined, chosen_combined_groups = [], []
    for task_type, quota in QUOTAS["combined6k"].items():
        rows, groups = choose_groups(
            combined_candidates, task_type, quota, args.seed, "combined6k"
        )
        chosen_combined += rows
        chosen_combined_groups += groups
    targeted_candidates, targeted_audit = filter_groups(
        targeted_rows, [*protected, *chosen_combined]
    )
    chosen_targeted, chosen_targeted_groups = [], []
    for task_type, quota in QUOTAS["targeted2k"].items():
        rows, groups = choose_groups(
            targeted_candidates, task_type, quota, args.seed, "targeted2k"
        )
        chosen_targeted += rows
        chosen_targeted_groups += groups
    supplements = [*chosen_combined, *chosen_targeted]
    if len(supplements) != 1200 or len({row["id"] for row in supplements}) != 1200:
        raise AssertionError("Supplement size or ID uniqueness changed")
    targeted.context_overlap(
        chosen_combined, targeted.context_rows(chosen_targeted), approximate=True
    )
    merged = [*base, *supplements]
    merged.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{args.seed}\0shuffle\0{row['id']}".encode()),
            row["id"],
        )
    )
    check_partition_isolation({"train": merged, "select": select, "cal": cal})
    if len(merged) != 5824 or len({row["group_id"] for row in merged}) < len(
        chosen_combined_groups
    ) + len(chosen_targeted_groups):
        raise AssertionError("Balanced mixed TRAIN shape changed")
    counts = collections.Counter(row["task_type"] for row in merged)
    if counts != {"choice": 4350, "noul": 1112, "score": 362}:
        raise AssertionError(f"Balanced type quotas differ: {dict(counts)}")
    if pilot.train_consistency_audit(merged)["conflicting_gold_groups"]:
        raise ValueError("Merged TRAIN has conflicting gold for identical inputs")
    holdout_audits = {}
    for role, rows in (("select", select), ("hard_cal", cal)):
        holdout_audits[role] = targeted.context_overlap(
            merged, targeted.context_rows(rows), approximate=True
        )
    context_audits = {}
    for role, path, expected in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_3task_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
        (
            "css_15task_evaluation",
            args.css_evaluation_prompts,
            "css-evaluation.prompts.jsonl",
        ),
    ):
        reference, _ = targeted.load_context_reference(path, expected_name=expected)
        context_audits[role] = targeted.context_overlap(
            merged, reference, approximate=True
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in merged}
    if max(lengths.values()) > args.max_row_tokens:
        raise ValueError("Balanced TRAIN contains overlength row")
    token_by_source = collections.Counter()
    combined_ids = {row["id"] for row in chosen_combined}
    targeted_ids = {row["id"] for row in chosen_targeted}
    for row in merged:
        bucket = (
            "existing_combined6k_supplement"
            if row["id"] in combined_ids
            else (
                "existing_targeted2k_supplement"
                if row["id"] in targeted_ids
                else "human_anchor_base"
            )
        )
        token_by_source[bucket] += lengths[row["id"]]
    output_name = "balanced_human_5824.train.jsonl"
    payloads = {
        output_name: pilot.jsonl_bytes(merged),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(args.output_dir / name, payload)
    check_partition_isolation(
        {
            "train": load_partition(args.output_dir / output_name, "train"),
            "select": load_partition(args.output_dir / "select.jsonl", "select"),
            "cal": load_partition(args.output_dir / "cal.jsonl", "cal"),
        }
    )
    manifest = {
        "schema_version": "decision2-balanced-human-5824/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "seed": args.seed,
        "quotas": QUOTAS,
        "inputs": {
            role: {"file": path.name, "sha256": expected}
            for role, (path, expected) in frozen.items()
        },
        "counts": {
            field: dict(
                sorted(collections.Counter(row[field] for row in merged).items())
            )
            for field in ("task_type", "family", "source", "language")
        },
        "selection": {
            "combined6k": {
                "group_ids": chosen_combined_groups,
                "rows": len(chosen_combined),
                "filter_audit": combined_audit,
            },
            "targeted2k": {
                "group_ids": chosen_targeted_groups,
                "rows": len(chosen_targeted),
                "filter_audit": targeted_audit,
            },
        },
        "token_audit": {
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(lengths.values()),
            "maximum": max(lengths.values()),
            "cap": args.max_row_tokens,
            "by_source": dict(sorted(token_by_source.items())),
        },
        "holdout_context_audits": holdout_audits,
        "gold_free_context_audits": {
            role: {"source": context_receipts[role], "overlap": result}
            for role, result in context_audits.items()
        },
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "bytes": len(payload),
                "rows": len(payload.splitlines()),
            }
            for name, payload in payloads.items()
        },
        "interpretation": "A training candidate with all 3,600 human Choice rows, 1,020 unchanged anchor rows, four contamination-safe FLUTE backfills, and 1,200 Noul/Score supplements. It is not an evaluated improvement.",
        "limitations": [
            "The added Noul/Score rows are existing TRAIN sources, including synthetic oracle cases; their diversity is limited.",
            "Combined6k and targeted2k supplements differ in length, family and wording; this is a type-balance intervention, not a causal source isolation.",
            "CSS FLUTE remains same-task supervised because the base contains 120 FLUTE TRAIN rows.",
            "The CSS final labels were not read; gold-free prompts were used solely to exclude context overlaps.",
        ],
        "sealed_gold": "CSS15 and benchmark final not read",
    }
    pilot._atomic_write(
        args.output_dir / "balanced_human_5824.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--combined-train", type=Path, required=True)
    parser.add_argument("--targeted-train", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--css-pilot-prompts", type=Path, required=True)
    parser.add_argument("--css-evaluation-prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default="decision2-balanced-human-5824-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args)
    print(
        pilot.canonical(
            {
                "rows": report["outputs"]["balanced_human_5824.train.jsonl"]["rows"],
                "sha256": report["outputs"]["balanced_human_5824.train.jsonl"][
                    "sha256"
                ],
                "tokens": report["token_audit"]["total"],
            }
        )
    )


if __name__ == "__main__":
    main()
