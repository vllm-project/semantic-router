"""Add audited structured replay to frozen balanced-human TRAIN for Nox-4B.

The original Decision 1.0 Nox-4B transition score fell after human-only
adaptation. This arm retains every balanced-human row and adds only earlier
TRAIN rows, grouped deterministically. DEV/CSS files are gold-free exclusions.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

BALANCED_SHA256 = "e83fb07021b779bb86d6b1d773b007c2dda9d91052aedf1f72f89bebbfef50e2"
BALANCED_MANIFEST_SHA256 = (
    "869a94c0c74b9e80f2b60bf414eb7440cda17cbce1e61906621bbe206ea5aa9f"
)
COMBINED_SHA256 = "4e82651181fd4f9b11e82718275a7370b82cbe810f20bbe87db9786cfdf888ad"
COMBINED_MANIFEST_SHA256 = (
    "9bd7d64ba41ebc5dc053667bd34d2f2e4bb0e0023c39c94f183215a2fa834c0a"
)
SELECT_SHA256 = "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38"
CAL_SHA256 = "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf"
DEV_SHA256 = "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a"
CSS_PILOT_SHA256 = "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda"
CSS_EVALUATION_SHA256 = (
    "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6"
)
QUOTAS = {
    "legacy:stage4-general-composition-v2": 1220,
    "legacy:stage3_replay": 450,
    "decision2_programmatic_original_v1": 180,
    "legacy:cosmos_qa": 300,
    "legacy:snli": 200,
    "legacy:nyu-mll/multi_nli": 200,
    "legacy:squad2_answerability": 150,
}
SEED = "decision2-nox4b-structured-replay-v1"
OUTPUT_NAME = "nox4b_structured.train.jsonl"


def _count(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(row[field] for row in rows).items()))


def _group_candidates(
    source: list[dict[str, Any]],
    protected: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Quarantine whole groups for IDs, inputs, contexts and near contexts."""
    protected_ids = {row["id"] for row in protected}
    protected_groups = {
        row.get("group_id") for row in protected if isinstance(row.get("group_id"), str)
    }
    protected_inputs = {
        row.get("input_sha256")
        for row in protected
        if isinstance(row.get("input_sha256"), str)
    }
    fingerprints = {targeted.text_hashes(row["state"]) for row in protected}
    raw, normalized = ({pair[index] for pair in fingerprints} for index in (0, 1))
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in source:
        if row["source"] in QUOTAS:
            grouped[row["group_id"]].append(row)
    rejected: collections.Counter[str] = collections.Counter()
    eligible: dict[str, list[dict[str, Any]]] = {}
    for group_id, rows in grouped.items():
        if len({row["source"] for row in rows}) != 1:
            rejected["mixed_source_group"] += len(rows)
            continue
        if any(
            row["id"] in protected_ids
            or row["group_id"] in protected_groups
            or row["input_sha256"] in protected_inputs
            for row in rows
        ):
            rejected["id_group_input"] += len(rows)
            continue
        if any(
            (pair := targeted.text_hashes(row["state"]))[0] in raw
            or pair[1] in normalized
            for row in rows
        ):
            rejected["exact_context"] += len(rows)
            continue
        eligible[group_id] = rows
    near = pilot.near_duplicates(
        targeted.context_rows([row for rows in eligible.values() for row in rows]),
        targeted.context_rows(protected),
        collect_left_ids=True,
    )
    near_ids = set(near.pop("left_ids"))
    for group_id in list(eligible):
        if any(row["id"] in near_ids for row in eligible[group_id]):
            rejected["near_context"] += len(eligible.pop(group_id))
    return eligible, {
        "input_groups": len(grouped),
        "input_rows": sum(len(rows) for rows in grouped.values()),
        "eligible_groups": len(eligible),
        "eligible_rows": sum(len(rows) for rows in eligible.values()),
        "rejected_rows": dict(sorted(rejected.items())),
        "near_audit_before_quarantine": near,
    }


def _select_source(
    candidates: dict[str, list[dict[str, Any]]],
    source: str,
    quota: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Interleave source families before filling an exact whole-group quota."""
    by_family: dict[str, list[str]] = collections.defaultdict(list)
    for group_id, rows in candidates.items():
        if rows[0]["source"] != source:
            continue
        families = {row["family"] for row in rows}
        if len(families) != 1:
            raise ValueError("Replay group mixes families")
        by_family[rows[0]["family"]].append(group_id)
    for family, groups in by_family.items():
        groups.sort(
            key=lambda group: (
                pilot.sha_bytes(f"{seed}\0{source}\0{family}\0{group}".encode()),
                group,
            )
        )
    chosen: list[str] = []
    remaining = quota
    while remaining and any(by_family.values()):
        progressed = False
        for family in sorted(by_family):
            groups = by_family[family]
            while groups:
                group = groups.pop(0)
                if len(candidates[group]) <= remaining:
                    chosen.append(group)
                    remaining -= len(candidates[group])
                    progressed = True
                    break
            if not remaining:
                break
        if not progressed:
            break
    rows = [row for group in chosen for row in candidates[group]]
    if len(rows) < quota * 0.98:
        available = sum(
            len(items) for items in candidates.values() if items[0]["source"] == source
        )
        raise ValueError(
            f"Not enough safe whole groups for {source}: missing {remaining}/{quota}, eligible {available}"
        )
    return rows, chosen


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sources = {
        "balanced": (args.balanced_train, BALANCED_SHA256),
        "balanced_manifest": (args.balanced_manifest, BALANCED_MANIFEST_SHA256),
        "combined": (args.combined_train, COMBINED_SHA256),
        "combined_manifest": (args.combined_manifest, COMBINED_MANIFEST_SHA256),
        "select": (args.select_file, SELECT_SHA256),
        "cal": (args.cal_file, CAL_SHA256),
        "synthetic_dev": (args.dev_prompts, DEV_SHA256),
        "css_pilot": (args.css_pilot_prompts, CSS_PILOT_SHA256),
        "css_evaluation": (args.css_evaluation_prompts, CSS_EVALUATION_SHA256),
    }
    for role, (path, digest) in sources.items():
        if pilot.sha_file(path) != digest:
            raise ValueError(f"Frozen {role} input SHA-256 differs")
    balanced_manifest = json.loads(args.balanced_manifest.read_text(encoding="utf-8"))
    combined_manifest = json.loads(args.combined_manifest.read_text(encoding="utf-8"))
    if (
        balanced_manifest["outputs"]["balanced_human_5824.train.jsonl"]["sha256"]
        != BALANCED_SHA256
    ):
        raise ValueError("Balanced builder manifest differs from its TRAIN bytes")
    if (
        combined_manifest["outputs"]["combined_6k.train.jsonl"]["sha256"]
        != COMBINED_SHA256
    ):
        raise ValueError("Combined builder manifest differs from its TRAIN bytes")
    balanced = load_partition(args.balanced_train, "train")
    combined = load_partition(args.combined_train, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    if (len(balanced), len(combined), len(select), len(cal)) != (5824, 6000, 600, 900):
        raise ValueError("Frozen training/holdout row counts differ")
    references = {}
    reference_receipts = {}
    for role, path, expected_name in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
        ("css_evaluation", args.css_evaluation_prompts, "css-evaluation.prompts.jsonl"),
    ):
        rows, receipt = targeted.load_context_reference(
            path, expected_name=expected_name
        )
        references[role] = rows
        reference_receipts[role] = receipt
    protected = [
        *balanced,
        *select,
        *cal,
        *(row for rows in references.values() for row in rows),
    ]
    candidates, filter_audit = _group_candidates(combined, protected)
    additions, selected_groups = [], {}
    for source, quota in QUOTAS.items():
        rows, groups = _select_source(candidates, source, quota, args.seed)
        additions.extend(rows)
        selected_groups[source] = groups
    if len(additions) < sum(QUOTAS.values()) * 0.98:
        raise AssertionError(
            "Structured replay fell below the minimum audited quota fill"
        )
    train = [*balanced, *additions]
    train.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{args.seed}\0shuffle\0{row['id']}".encode()),
            row["id"],
        )
    )
    check_partition_isolation({"train": train, "select": select, "cal": cal})
    if pilot.train_consistency_audit(train)["conflicting_gold_groups"]:
        raise ValueError("Structured replay has conflicting labels for the same input")
    overlap = {
        role: targeted.context_overlap(train, rows, approximate=True)
        for role, rows in {
            "select": targeted.context_rows(select),
            "cal": targeted.context_rows(cal),
            **references,
        }.items()
    }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = [pilot.count_tokens(row, tokenizer) for row in train]
    if max(lengths) > args.max_row_tokens:
        raise ValueError("Structured replay contains an overlength row")
    payloads = {
        OUTPUT_NAME: pilot.jsonl_bytes(train),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(args.output_dir / name, payload)
    report = {
        "schema_version": "decision2-nox4b-structured-replay/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "seed": args.seed,
        "inputs": {
            role: {"file": path.name, "sha256": digest}
            for role, (path, digest) in sources.items()
        },
        "target_quotas": QUOTAS,
        "realized_source_counts": _count(additions, "source"),
        "filter_audit": filter_audit,
        "selected_group_ids_by_source": selected_groups,
        "added_counts": {
            field: _count(additions, field)
            for field in ("task_type", "family", "source", "language")
        },
        "counts": {
            field: _count(train, field)
            for field in ("task_type", "family", "source", "language")
        },
        "reference_receipts": reference_receipts,
        "holdout_context_audits": overlap,
        "token_audit": {
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(lengths),
            "maximum": max(lengths),
            "cap": args.max_row_tokens,
        },
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "bytes": len(payload),
                "rows": len(payload.splitlines()),
            }
            for name, payload in payloads.items()
        },
        "limitations": [
            "This arm changes training row count, source mix and effective type weighting together; it is not a causal replay ablation.",
            "The structured replay comes from old TRAIN sources that initialized Decision 1.0; it can preserve those skills without proving new generalization.",
            "The balanced base contains 120 FLUTE TRAIN rows, so CSS FLUTE remains same-task supervised.",
            "Approximate near-context audit cannot prove semantic independence; CSS final labels and benchmark final gold were not accessed.",
        ],
    }
    pilot._atomic_write(
        args.output_dir / "nox4b_structured.manifest.json",
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-train", type=Path, required=True)
    parser.add_argument("--balanced-manifest", type=Path, required=True)
    parser.add_argument("--combined-train", type=Path, required=True)
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--css-pilot-prompts", type=Path, required=True)
    parser.add_argument("--css-evaluation-prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args)
    print(
        json.dumps(
            {
                "rows": result["outputs"][OUTPUT_NAME]["rows"],
                "sha256": result["outputs"][OUTPUT_NAME]["sha256"],
                "added_counts": result["added_counts"]["task_type"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
