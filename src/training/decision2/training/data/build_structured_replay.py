"""Add quarantined legacy structure rows to frozen balanced human TRAIN.

Only gold-free DEV/CSS prompt inputs are inspected for evaluation overlap.
The existing SELECT600 and hard CAL900 are copied byte-for-byte.
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

FROZEN_SHA256 = {
    "base": "e83fb07021b779bb86d6b1d773b007c2dda9d91052aedf1f72f89bebbfef50e2",
    "base_manifest": "869a94c0c74b9e80f2b60bf414eb7440cda17cbce1e61906621bbe206ea5aa9f",
    "legacy": "603600a6d1aeabe9179e5d3a85f275817f79a614e372bd64cb16584f50b83e17",
    "legacy_manifest": "84be52e110984eead1d204313dd1f864e503cd175f2bfe6c98bc77b0522d38b7",
    "select": "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38",
    "hard_cal": "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf",
    "synthetic_dev": "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    "css_pilot": "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
    "css15_gold_free": "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
}
ALLOWED_SOURCES = {"legacy:stage4-general-composition-v2", "legacy:stage3_replay"}
SEED = "decision2-sol2b-human-structured-replay-v1"


def filter_groups(
    source: list[dict[str, Any]],
    protected: list[dict[str, Any]],
    *,
    near_duplicate_search: Any = pilot.near_duplicates,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep complete Stage4 groups and quarantine exact/near protected contexts."""
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in source:
        by_group[row["group_id"]].append(row)
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
    raw_hashes = {raw for raw, _ in hashes}
    norm_hashes = {norm for _, norm in hashes}
    rejected = collections.Counter()
    candidates = {}
    for group, rows in by_group.items():
        if not all(
            row["family"].startswith("stage4_") and row["source"] in ALLOWED_SOURCES
            for row in rows
        ):
            rejected["outside_structured_source"] += len(rows)
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
    before_near = sum(map(len, candidates.values()))
    near = near_duplicate_search(
        targeted.context_rows([row for rows in candidates.values() for row in rows]),
        targeted.context_rows(protected),
        collect_left_ids=True,
    )
    near_ids = set(near.pop("left_ids"))
    for group in list(candidates):
        if any(row["id"] in near_ids for row in candidates[group]):
            rejected["near_context"] += len(candidates[group])
            del candidates[group]
    selected = [row for group in sorted(candidates) for row in candidates[group]]
    return selected, {
        "source_rows": len(source),
        "candidate_rows_before_near": before_near,
        "selected_rows": len(selected),
        "selected_groups": len(candidates),
        "rejected_rows_by_reason": dict(sorted(rejected.items())),
        "near_audit_before_quarantine": near,
        "near_flagged_rows": len(near_ids),
        "selection": "all clean, whole Stage4 groups from two documented legacy sources",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inputs = {
        "base": args.base_train,
        "base_manifest": args.base_manifest,
        "legacy": args.legacy_train,
        "legacy_manifest": args.legacy_manifest,
        "select": args.select_file,
        "hard_cal": args.cal_file,
        "synthetic_dev": args.dev_prompts,
        "css_pilot": args.css_pilot_prompts,
        "css15_gold_free": args.css15_prompts,
    }
    for role, path in inputs.items():
        if pilot.sha_file(path) != FROZEN_SHA256[role]:
            raise ValueError(f"{role} differs from frozen SHA-256")
    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    legacy_manifest = json.loads(args.legacy_manifest.read_text(encoding="utf-8"))
    if (
        base_manifest["outputs"]["balanced_human_5824.train.jsonl"]["sha256"]
        != FROZEN_SHA256["base"]
        or legacy_manifest["output"]["sha256"] != FROZEN_SHA256["legacy"]
    ):
        raise ValueError("Source manifest does not bind source TRAIN bytes")
    licenses = {
        source: legacy_manifest["source_attribution"][source]
        for source in sorted(ALLOWED_SOURCES)
    }
    if any(
        not all(
            record.get(key)
            for key in ("license", "attribution", "evidence", "rights_status")
        )
        for record in licenses.values()
    ):
        raise ValueError("Legacy source license evidence incomplete")
    base = load_partition(args.base_train, "train")
    legacy = load_partition(args.legacy_train, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    if (len(base), len(legacy), len(select), len(cal)) != (5824, 6000, 600, 900):
        raise ValueError("Frozen source cardinality changed")
    references = {}
    for role, path, name in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
        ("css15_gold_free", args.css15_prompts, "css-evaluation.prompts.jsonl"),
    ):
        references[role] = targeted.load_context_reference(path, expected_name=name)[0]
    protected = [
        *base,
        *select,
        *cal,
        *references["synthetic_dev"],
        *references["css_pilot"],
        *references["css15_gold_free"],
    ]
    added, selection_audit = filter_groups(legacy, protected)
    if not 2000 <= len(added) <= 3000:
        raise ValueError(
            f"Expected about 2k–3k clean structured rows; got {len(added)}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in added}
    if max(lengths.values()) > args.max_row_tokens:
        raise ValueError("Structured replay contains an overlength row")
    merged = [*base, *added]
    merged.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{args.seed}\0shuffle\0{row['id']}".encode()),
            row["id"],
        )
    )
    check_partition_isolation({"train": merged, "select": select, "cal": cal})
    if pilot.train_consistency_audit(merged)["conflicting_gold_groups"]:
        raise ValueError(
            "Merged TRAIN contains conflicting labels for identical inputs"
        )
    holdout_audits = {
        role: targeted.context_overlap(
            added, targeted.context_rows(rows), approximate=True
        )
        for role, rows in (("select", select), ("hard_cal", cal))
    }
    context_audits = {
        role: targeted.context_overlap(added, rows, approximate=True)
        for role, rows in references.items()
    }
    name = "human_structured_replay.train.jsonl"
    payloads = {
        name: pilot.jsonl_bytes(merged),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for filename, payload in payloads.items():
        pilot._atomic_write(args.output_dir / filename, payload)
    selected_ids = sorted(row["id"] for row in added)
    manifest = {
        "schema_version": "decision2-human-structured-replay/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "seed": args.seed,
        "inputs": {
            role: {"file": path.name, "sha256": FROZEN_SHA256[role]}
            for role, path in inputs.items()
        },
        "source_license_evidence": licenses,
        "selection_audit": selection_audit,
        "selected_added_ids": selected_ids,
        "selected_added_id_input_sha256": {
            row["id"]: row["input_sha256"] for row in added
        },
        "selected_added_group_ids": sorted({row["group_id"] for row in added}),
        "added_counts": {
            field: dict(
                sorted(collections.Counter(row[field] for row in added).items())
            )
            for field in ("family", "source", "task_type", "language")
        },
        "merged_counts": {
            field: dict(
                sorted(collections.Counter(row[field] for row in merged).items())
            )
            for field in ("family", "source", "task_type", "language")
        },
        "added_token_audit": {
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(lengths.values()),
            "maximum": max(lengths.values()),
            "cap": args.max_row_tokens,
        },
        "holdout_context_audits": holdout_audits,
        "gold_free_context_audits": context_audits,
        "outputs": {
            filename: {
                "sha256": pilot.sha_bytes(payload),
                "bytes": len(payload),
                "rows": len(payload.splitlines()),
            }
            for filename, payload in payloads.items()
        },
        "sealed_gold": "CSS15 and family-disjoint final labels never read",
        "limits": [
            "Near search uses approximate SimHash candidates and is not mathematically exhaustive.",
            "The existing balanced-human base's own overlap clearance is inherited from its pinned manifest.",
            "Stage3 replay source includes attributed CLINC/Banking rows; model publication needs source review.",
        ],
    }
    pilot._atomic_write(
        args.output_dir / "human_structured_replay.manifest.json",
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--legacy-train", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--css-pilot-prompts", type=Path, required=True)
    parser.add_argument("--css15-prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args)
    print(
        pilot.canonical(
            {
                "rows": report["outputs"]["human_structured_replay.train.jsonl"][
                    "rows"
                ],
                "sha256": report["outputs"]["human_structured_replay.train.jsonl"][
                    "sha256"
                ],
                "added": report["selection_audit"]["selected_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
