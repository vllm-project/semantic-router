"""Merge frozen 1,024-row anchor with 3,600 TweetEval human TRAIN rows.

The CSS 15-task evaluation is accessed only through gold-free prompt inputs
for ID/context overlap auditing. Its labels are never loaded.
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

ANCHOR_SHA256 = "7ab3df2a4e2c5ac74e12923b6ba94ef192ed90b94a8a612138ff9fe0cc8b1ddd"
HUMAN_SHA256 = "a3c21e6b3c3d9c1abed8facd3112cfa0fdf42779519f94e4fc62d95ef1f05339"
BUDGET_MANIFEST_SHA256 = (
    "b236cdfe149689306a620143f61575c8b55c736f3609055d2bc1ff4d25308b04"
)
HUMAN_MANIFEST_SHA256 = (
    "4b000a401f61d0ecbcee7e25d42d5e76c226859092cc8e8b00338d0fd33943bd"
)
COMBINED_6K_SHA256 = "4e82651181fd4f9b11e82718275a7370b82cbe810f20bbe87db9786cfdf888ad"
SELECT_SHA256 = "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38"
CAL_SHA256 = "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf"
DEV_PROMPTS_SHA256 = "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a"
CSS_PILOT_PROMPTS_SHA256 = (
    "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda"
)
CSS_EVALUATION_PROMPTS_SHA256 = (
    "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6"
)


def source_bucket(row: dict[str, Any]) -> str:
    return (
        "human_tweeteval"
        if row["source"].startswith("tweeteval_train:")
        else budgets.bucket(row)
    )


def ordered_merge(
    anchor: list[dict[str, Any]], human: list[dict[str, Any]], seed: str
) -> list[dict[str, Any]]:
    if not seed.strip():
        raise ValueError("Nonempty shuffle seed required")
    merged = [*anchor, *human]
    merged.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{seed}\0shuffle\0{row['id']}".encode()),
            row["id"],
        )
    )
    if len(merged) != 4624 or len({row["id"] for row in merged}) != 4624:
        raise ValueError("Merged human candidate requires 4,624 unique rows")
    return merged


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(row[field] for row in rows).items()))


def repair_anchor(
    anchor: list[dict[str, Any]],
    full: list[dict[str, Any]],
    human: list[dict[str, Any]],
    holdouts: list[dict[str, Any]],
    references: list[dict[str, Any]],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Quarantine final-prompt near neighbors, then backfill safe FLUTE groups."""
    full_by_id = {row["id"]: row for row in full}
    if any(
        row["id"] not in full_by_id
        or pilot.canonical(row) != pilot.canonical(full_by_id[row["id"]])
        for row in anchor
    ):
        raise ValueError("Anchor rows differ from the frozen combined6k source")
    near_anchor = pilot.near_duplicates(
        targeted.context_rows(anchor), references, collect_left_ids=True
    )
    flagged_ids = set(near_anchor.pop("left_ids"))
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in anchor:
        by_group[row["group_id"]].append(row)
    flagged_groups = {row["group_id"] for row in anchor if row["id"] in flagged_ids}
    removed = [row for row in anchor if row["group_id"] in flagged_groups]
    if len(removed) != 4 or any(budgets.bucket(row) != "flute" for row in removed):
        raise ValueError(f"Expected four FLUTE near-context rows; got {len(removed)}")
    if any(len(by_group[group]) != 1 for group in flagged_groups):
        raise ValueError("Near-context FLUTE source group is not a singleton")
    kept = [row for row in anchor if row["group_id"] not in flagged_groups]
    anchor_ids = {row["id"] for row in anchor}
    protected = [*kept, *human, *holdouts, *references]
    blocked_ids = {row["id"] for row in protected}
    blocked_groups = {
        row["group_id"] for row in protected if isinstance(row.get("group_id"), str)
    }
    blocked_inputs = {
        row["input_sha256"]
        for row in protected
        if isinstance(row.get("input_sha256"), str)
    }
    blocked_hashes = {targeted.text_hashes(row["state"]) for row in protected}
    raw_hashes = {pair[0] for pair in blocked_hashes}
    norm_hashes = {pair[1] for pair in blocked_hashes}
    group_sizes = collections.Counter(row["group_id"] for row in full)
    donor_pool = []
    for row in full:
        if (
            budgets.bucket(row) != "flute"
            or row["id"] in anchor_ids
            or group_sizes[row["group_id"]] != 1
        ):
            continue
        raw, norm = targeted.text_hashes(row["state"])
        if (
            row["id"] in blocked_ids
            or row["group_id"] in blocked_groups
            or row["input_sha256"] in blocked_inputs
            or raw in raw_hashes
            or norm in norm_hashes
        ):
            continue
        donor_pool.append(row)
    near_donors = pilot.near_duplicates(
        targeted.context_rows(donor_pool),
        [*references, *targeted.context_rows([*kept, *human, *holdouts])],
        collect_left_ids=True,
    )
    near_donor_ids = set(near_donors.pop("left_ids"))
    safe_donors = [row for row in donor_pool if row["id"] not in near_donor_ids]
    safe_donors.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{seed}\0flute-backfill\0{row['id']}".encode()),
            row["id"],
        )
    )
    if len(safe_donors) < len(removed):
        raise ValueError("Insufficient safe FLUTE backfill rows")
    replacements = safe_donors[: len(removed)]
    repaired = [*kept, *replacements]
    if len(repaired) != 1024 or collections.Counter(
        budgets.bucket(row) for row in repaired
    ) != {"legacy": 853, "programmatic": 51, "flute": 120}:
        raise AssertionError("Repaired anchor cardinality or source mix changed")
    targeted.context_overlap(repaired, references, approximate=True)
    receipt = {
        "quarantined_rows": len(removed),
        "quarantined_groups": len(flagged_groups),
        "unchanged_anchor_rows": len(kept),
        "quarantined_source_ids": sorted(row["id"] for row in removed),
        "backfill_source_ids": sorted(row["id"] for row in replacements),
        "backfill_pool_rows_before_near_filter": len(donor_pool),
        "backfill_pool_rows_after_near_filter": len(safe_donors),
        "original_anchor_near_context_audit": near_anchor,
        "donor_pool_near_context_audit": near_donors,
        "selection": "complete singleton FLUTE groups ranked by SHA-256(seed + NUL + flute-backfill + NUL + row id); gold-free references only exclude",
    }
    return repaired, receipt


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frozen = {
        "anchor": (args.anchor_train, ANCHOR_SHA256),
        "human": (args.human_train, HUMAN_SHA256),
        "combined_6k": (args.combined_6k, COMBINED_6K_SHA256),
        "budget_manifest": (args.budget_manifest, BUDGET_MANIFEST_SHA256),
        "human_manifest": (args.human_manifest, HUMAN_MANIFEST_SHA256),
        "select": (args.select_file, SELECT_SHA256),
        "hard_cal": (args.cal_file, CAL_SHA256),
        "synthetic_dev_prompts": (args.dev_prompts, DEV_PROMPTS_SHA256),
        "css_pilot_prompts": (args.css_pilot_prompts, CSS_PILOT_PROMPTS_SHA256),
        "css_evaluation_prompts": (
            args.css_evaluation_prompts,
            CSS_EVALUATION_PROMPTS_SHA256,
        ),
    }
    for name, (path, expected) in frozen.items():
        if pilot.sha_file(path) != expected:
            raise ValueError(f"{name} differs from its predeclared SHA-256")
    budget_manifest = json.loads(args.budget_manifest.read_text(encoding="utf-8"))
    human_manifest = json.loads(args.human_manifest.read_text(encoding="utf-8"))
    if (
        budget_manifest["budgets"]["1024"]["sha256"] != ANCHOR_SHA256
        or human_manifest["output"]["sha256"] != HUMAN_SHA256
    ):
        raise ValueError("Source manifests do not bind the frozen input files")
    anchor = load_partition(args.anchor_train, "train")
    human = load_partition(args.human_train, "train")
    full = load_partition(args.combined_6k, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    if (len(anchor), len(human), len(full), len(select), len(cal)) != (
        1024,
        3600,
        6000,
        600,
        900,
    ):
        raise ValueError("Frozen partition cardinality changed")
    anchor_mix = collections.Counter(budgets.bucket(row) for row in anchor)
    if anchor_mix != {"legacy": 853, "programmatic": 51, "flute": 120}:
        raise ValueError("Frozen anchor source mix changed")
    if any(not row["source"].startswith("tweeteval_train:") for row in human):
        raise ValueError("Human source contains an unexpected row")
    if len({row["group_id"] for row in human}) != len(human):
        raise ValueError("Human TRAIN source repeats a normalized context")
    context_audits = {}
    references = []
    for name, path, expected in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_3task_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
        (
            "css_15task_evaluation",
            args.css_evaluation_prompts,
            "css-evaluation.prompts.jsonl",
        ),
    ):
        reference, receipt = targeted.load_context_reference(
            path, expected_name=expected
        )
        context_audits[name] = {"source": receipt}
        references.extend(reference)
    safe_anchor, anchor_repair = repair_anchor(
        anchor, full, human, [*select, *cal], references, args.seed
    )
    source_overlap = pilot.overlap_audit(human, safe_anchor)
    if (
        pilot.audit_has_exact_overlap(source_overlap)
        or source_overlap["near_duplicate"]["count"]
    ):
        raise ValueError(f"Human and anchor TRAIN overlap: {source_overlap}")
    source_context = targeted.context_overlap(
        human, targeted.context_rows(safe_anchor), approximate=True
    )
    merged = ordered_merge(safe_anchor, human, args.seed)
    check_partition_isolation({"train": merged, "select": select, "cal": cal})
    holdout_audits = {}
    for name, rows in (("select", select), ("hard_cal", cal)):
        input_audit = pilot.overlap_audit(merged, rows)
        if (
            pilot.audit_has_exact_overlap(input_audit)
            or input_audit["near_duplicate"]["count"]
        ):
            raise ValueError(f"TRAIN overlaps {name} by input: {input_audit}")
        context_audit = targeted.context_overlap(
            merged, targeted.context_rows(rows), approximate=True
        )
        holdout_audits[name] = {"input": input_audit, "context": context_audit}
    if pilot.train_consistency_audit(merged)["conflicting_gold_groups"]:
        raise ValueError("Merged TRAIN has conflicting labels for identical inputs")
    for name, path, expected in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_3task_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
        (
            "css_15task_evaluation",
            args.css_evaluation_prompts,
            "css-evaluation.prompts.jsonl",
        ),
    ):
        reference, receipt = targeted.load_context_reference(
            path, expected_name=expected
        )
        context_audits[name] = {
            "source": receipt,
            "overlap": targeted.context_overlap(merged, reference, approximate=True),
        }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in merged}
    if max(lengths.values()) > args.max_row_tokens:
        raise ValueError("Merged TRAIN has row over tokenizer length cap")
    token_by_source = collections.Counter()
    token_by_type = collections.Counter()
    for row in merged:
        token_by_source[source_bucket(row)] += lengths[row["id"]]
        token_by_type[row["task_type"]] += lengths[row["id"]]
    if sum(token_by_source.values()) != sum(lengths.values()):
        raise AssertionError("Source token accounting differs from total")
    name = "human_anchor_4624.train.jsonl"
    payloads = {
        name: pilot.jsonl_bytes(merged),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    output_dir.mkdir(parents=True, mode=0o700)
    for file_name, payload in payloads.items():
        pilot._atomic_write(output_dir / file_name, payload)
    check_partition_isolation(
        {
            "train": load_partition(output_dir / name, "train"),
            "select": load_partition(output_dir / "select.jsonl", "select"),
            "cal": load_partition(output_dir / "cal.jsonl", "cal"),
        }
    )
    manifest = {
        "schema_version": "decision2-human-anchor-4624/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "inputs": {
            key: {"file": path.name, "sha256": expected}
            for key, (path, expected) in frozen.items()
        },
        "lineage": {
            "anchor": "1,020 unchanged combined_1024 rows, four FLUTE rows replaced from combined6k after gold-free near-context quarantine; source mix 853 legacy, 51 programmatic, 120 FLUTE",
            "human": "unchanged TweetEval TRAIN-only 3,600, ten buckets, class-balanced",
            "data_rights": human_manifest["rights"],
        },
        "shuffle": {
            "seed": args.seed,
            "method": "sort by SHA-256(seed + NUL + shuffle + NUL + row id), then id",
        },
        "counts": {
            field: count_by(merged, field)
            for field in ("family", "task_type", "language", "source")
        },
        "source_bucket_counts": dict(
            sorted(collections.Counter(source_bucket(row) for row in merged).items())
        ),
        "token_audit": {
            "method": "decoder-v2 segmented exact Qwen tokenizer",
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(lengths.values()),
            "minimum": min(lengths.values()),
            "maximum": max(lengths.values()),
            "max_row_tokens": args.max_row_tokens,
            "anchor_total": sum(lengths[row["id"]] for row in safe_anchor),
            "human_total": sum(lengths[row["id"]] for row in human),
            "by_source_bucket": dict(sorted(token_by_source.items())),
            "by_task_type": dict(sorted(token_by_type.items())),
        },
        "source_overlap": {"input": source_overlap, "context": source_context},
        "anchor_repair": anchor_repair,
        "holdout_audits": holdout_audits,
        "context_audits": context_audits,
        "sealed_gold": {
            "css_15task_evaluation": "not read",
            "benchmark_final": "not read",
        },
        "outputs": {
            file_name: {
                "sha256": pilot.sha_bytes(payload),
                "rows": len(payload.splitlines()),
                "bytes": len(payload),
            }
            for file_name, payload in payloads.items()
        },
        "evaluation_interpretation": human_manifest["evaluation_interpretation"],
        "limitations": [
            "This intervention changes row count, token count, task mix, label priors and source together; compare against an equal-budget control before attributing a gain.",
            "Four original anchor FLUTE rows were near CSS final prompts, so this arm shares 1,020 rather than 1,024 exact rows with the original anchor; the replacements were picked by a fixed hash rank after gold-free exclusion.",
            "The anchor includes 120 FLUTE TRAIN rows, making CSS FLUTE same-task supervised.",
            "The human rows are all Choice, sharply reducing the Noul/Score proportions from the targeted candidate; typed-task regression is a material training risk.",
            "TweetEval umbrella rights defer to original tasks and Twitter; raw text remains private.",
            "CSS 15-task labels and benchmark final labels were not read; gold-free CSS contexts were used solely for overlap auditing.",
        ],
    }
    pilot._atomic_write(
        output_dir / "human_anchor_4624.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-train", type=Path, required=True)
    parser.add_argument("--human-train", type=Path, required=True)
    parser.add_argument("--combined-6k", type=Path, required=True)
    parser.add_argument("--budget-manifest", type=Path, required=True)
    parser.add_argument("--human-manifest", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--css-pilot-prompts", type=Path, required=True)
    parser.add_argument("--css-evaluation-prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default="decision2-human-anchor-4624-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "rows": manifest["outputs"]["human_anchor_4624.train.jsonl"]["rows"],
                "train_sha256": manifest["outputs"]["human_anchor_4624.train.jsonl"][
                    "sha256"
                ],
                "tokens": manifest["token_audit"]["total"],
            }
        )
    )


if __name__ == "__main__":
    main()
