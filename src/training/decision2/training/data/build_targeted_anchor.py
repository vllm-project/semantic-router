"""Combine frozen 1,024-row anchor and fresh 2,000-row targeted TRAIN candidate.

DEV and CSS three-task pilot prompts are read only for exclusion checks. The
builder has no input for benchmark final or CSS fifteen-task evaluation data.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

from training.data import build_budget_subsets as budgets
from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

FROZEN_ANCHOR_SHA256 = (
    "7ab3df2a4e2c5ac74e12923b6ba94ef192ed90b94a8a612138ff9fe0cc8b1ddd"
)
FROZEN_TARGETED_SHA256 = (
    "59d40112ab4d9d0688b3b121212ad5cc36329fe18170839485c9906544f0dee0"
)
FROZEN_BUDGET_MANIFEST_SHA256 = (
    "b236cdfe149689306a620143f61575c8b55c736f3609055d2bc1ff4d25308b04"
)
FROZEN_TARGETED_MANIFEST_SHA256 = (
    "f26cc542d76f3b9e0237046edadff865d5a7115cdc1e2f6052cdf35ab7b6866a"
)
FROZEN_DEV_PROMPTS_SHA256 = (
    "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a"
)
FROZEN_CSS_PILOT_PROMPTS_SHA256 = (
    "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda"
)


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(row[field] for row in rows).items()))


def source_bucket(row: dict[str, Any]) -> str:
    if row["source"] == targeted.SOURCE:
        return "targeted"
    return budgets.bucket(row)


def verify_group_integrity(
    anchor: list[dict[str, Any]],
    full: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
) -> dict[str, Any]:
    full_sizes = collections.Counter(row["group_id"] for row in full)
    anchor_sizes = collections.Counter(row["group_id"] for row in anchor)
    fresh_sizes = collections.Counter(row["group_id"] for row in fresh)
    partial = {
        group: {"selected": count, "source": full_sizes[group]}
        for group, count in anchor_sizes.items()
        if count != full_sizes[group]
    }
    if partial:
        raise ValueError(f"Anchor contains split frozen source groups: {partial}")
    if len(fresh_sizes) != 1000 or set(fresh_sizes.values()) != {2}:
        raise ValueError("Targeted source must contain 1,000 complete two-row groups")
    if set(anchor_sizes) & set(fresh_sizes):
        raise ValueError("Anchor and targeted source share a group ID")
    return {
        "frozen_anchor_groups": len(anchor_sizes),
        "targeted_complete_pair_groups": len(fresh_sizes),
        "partial_source_groups": 0,
    }


def ordered_merge(
    anchor: list[dict[str, Any]], fresh: list[dict[str, Any]], seed: str
) -> list[dict[str, Any]]:
    if not seed.strip():
        raise ValueError("Shuffle seed must be nonempty")
    merged = [*anchor, *fresh]
    merged.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{seed}\0shuffle\0{row['id']}".encode()),
            row["id"],
        )
    )
    if len(merged) != 3024 or len({row["id"] for row in merged}) != 3024:
        raise AssertionError("Merged candidate is not 3,024 unique rows")
    return merged


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frozen = (
        (args.anchor_train, FROZEN_ANCHOR_SHA256),
        (args.targeted_train, FROZEN_TARGETED_SHA256),
        (args.combined_6k, targeted.FROZEN_COMBINED_SHA256),
        (args.budget_manifest, FROZEN_BUDGET_MANIFEST_SHA256),
        (args.targeted_manifest, FROZEN_TARGETED_MANIFEST_SHA256),
        (args.select_file, targeted.FROZEN_SELECT_SHA256),
        (args.cal_file, targeted.FROZEN_CAL_SHA256),
        (args.dev_prompts, FROZEN_DEV_PROMPTS_SHA256),
        (args.css_pilot_prompts, FROZEN_CSS_PILOT_PROMPTS_SHA256),
    )
    for path, expected in frozen:
        if pilot.sha_file(path) != expected:
            raise ValueError(f"Frozen input differs: {path.name}")
    budget_manifest = json.loads(args.budget_manifest.read_text())
    targeted_manifest = json.loads(args.targeted_manifest.read_text())
    if (
        budget_manifest["budgets"]["1024"]["sha256"] != FROZEN_ANCHOR_SHA256
        or targeted_manifest["output"]["sha256"] != FROZEN_TARGETED_SHA256
    ):
        raise ValueError("Source manifests do not bind the two frozen training inputs")
    anchor = load_partition(args.anchor_train, "train")
    fresh = load_partition(args.targeted_train, "train")
    full = load_partition(args.combined_6k, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    if len(anchor) != 1024 or len(fresh) != 2000 or len(full) != 6000:
        raise ValueError("Frozen source cardinality changed")
    anchor_mix = collections.Counter(budgets.bucket(row) for row in anchor)
    if anchor_mix != {"legacy": 853, "programmatic": 51, "flute": 120}:
        raise ValueError(f"Anchor source mix changed: {dict(anchor_mix)}")
    if {row["source"] for row in fresh} != {targeted.SOURCE}:
        raise ValueError("Targeted source contains unexpected rows")
    group_receipt = verify_group_integrity(anchor, full, fresh)
    source_overlap = pilot.overlap_audit(fresh, anchor)
    if (
        pilot.audit_has_exact_overlap(source_overlap)
        or source_overlap["near_duplicate"]["count"]
    ):
        raise ValueError(f"Two TRAIN sources overlap: {source_overlap}")
    merged = ordered_merge(anchor, fresh, args.seed)
    check_partition_isolation({"train": merged, "select": select, "cal": cal})
    holdout_audits = {}
    for role, rows in (("select", select), ("cal", cal)):
        audit = pilot.overlap_audit(merged, rows)
        if pilot.audit_has_exact_overlap(audit) or audit["near_duplicate"]["count"]:
            raise ValueError(f"TRAIN overlaps {role}: {audit}")
        holdout_audits[role] = audit
    if pilot.train_consistency_audit(merged)["conflicting_gold_groups"]:
        raise ValueError("Merged TRAIN has conflicting labels for identical inputs")
    context_audits = {}
    for role, path, expected in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_3task_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
    ):
        reference, receipt = targeted.load_context_reference(
            path, expected_name=expected
        )
        context_audits[role] = {
            "source": receipt,
            "overlap": targeted.context_overlap(merged, reference, approximate=True),
        }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in merged}
    if max(lengths.values()) > args.max_row_tokens:
        raise ValueError("Merged TRAIN has a row over the exact token cap")
    token_by_source = collections.Counter()
    for row in merged:
        token_by_source[source_bucket(row)] += lengths[row["id"]]
    if sum(token_by_source.values()) != sum(lengths.values()):
        raise AssertionError("Token buckets do not sum to the total")
    train_name = "targeted_anchor_3024.train.jsonl"
    payloads = {
        train_name: pilot.jsonl_bytes(merged),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(output_dir / name, payload)
    loaded = load_partition(output_dir / train_name, "train")
    check_partition_isolation(
        {
            "train": loaded,
            "select": load_partition(output_dir / "select.jsonl", "select"),
            "cal": load_partition(output_dir / "cal.jsonl", "cal"),
        }
    )
    manifest = {
        "schema_version": "decision2-targeted-anchor-3024/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "inputs": {path.name: pilot.sha_file(path) for path, _ in frozen},
        "lineage": {
            "anchor": "unchanged combined_1024 with 853 legacy, 51 original programmatic, 120 FLUTE rows",
            "targeted": "unchanged targeted_2k_v1 oracle source",
            "anchor_manifest_sha256": FROZEN_BUDGET_MANIFEST_SHA256,
            "targeted_manifest_sha256": FROZEN_TARGETED_MANIFEST_SHA256,
        },
        "shuffle": {
            "seed": args.seed,
            "method": "sort by SHA-256(seed + NUL + shuffle + NUL + row id), then row id",
        },
        "counts": {
            field: count_by(merged, field)
            for field in ("family", "task_type", "language", "source")
        },
        "source_bucket_counts": dict(
            sorted(collections.Counter(source_bucket(row) for row in merged).items())
        ),
        "groups": group_receipt,
        "token_audit": {
            "method": "decoder-v2 segmented exact Qwen tokenizer",
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(lengths.values()),
            "minimum": min(lengths.values()),
            "maximum": max(lengths.values()),
            "max_row_tokens": args.max_row_tokens,
            "by_source_bucket": dict(sorted(token_by_source.items())),
            "anchor_total": sum(lengths[row["id"]] for row in anchor),
            "targeted_total": sum(lengths[row["id"]] for row in fresh),
        },
        "source_overlap": source_overlap,
        "holdout_audits": holdout_audits,
        "context_audits": context_audits,
        "sealed_holdouts": {
            "benchmark_final": "not read",
            "css_15task_evaluation": "not read",
        },
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "rows": len(payload.splitlines()),
                "bytes": len(payload),
            }
            for name, payload in payloads.items()
        },
        "interpretation": "A training candidate for Lux/2B LoRA comparison, not an evaluated improvement or a token-matched causal contrast.",
        "limitations": [
            "The extra 2,000 rows are short; row, token and source distributions differ from both frozen comparators.",
            "CSS FLUTE evaluation remains supervised same-task because the anchor includes FLUTE train.",
            "No final or CSS fifteen-task evaluation text or labels were read; no empirical near-duplicate audit against sealed sets is claimed.",
        ],
    }
    pilot._atomic_write(
        output_dir / "targeted_anchor_3024.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-train", type=Path, required=True)
    parser.add_argument("--targeted-train", type=Path, required=True)
    parser.add_argument("--combined-6k", type=Path, required=True)
    parser.add_argument("--budget-manifest", type=Path, required=True)
    parser.add_argument("--targeted-manifest", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--css-pilot-prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default="decision2-targeted-anchor-3024-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "rows": manifest["outputs"]["targeted_anchor_3024.train.jsonl"]["rows"],
                "train_sha256": manifest["outputs"]["targeted_anchor_3024.train.jsonl"][
                    "sha256"
                ],
                "tokens": manifest["token_audit"]["total"],
            }
        )
    )


if __name__ == "__main__":
    main()
