"""Build the preregistered small, short structured TRAIN replay control.

Only legacy TRAIN rows provide new labels. Protected prompts are read solely
for input overlap; their gold files are neither accepted nor opened.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

FROZEN = {
    "base": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "base_manifest": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
    "legacy": "603600a6d1aeabe9179e5d3a85f275817f79a614e372bd64cb16584f50b83e17",
    "legacy_manifest": "84be52e110984eead1d204313dd1f864e503cd175f2bfe6c98bc77b0522d38b7",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
FAMILIES = frozenset(
    {
        "stage4_arithmetic",
        "stage4_ordinal",
        "stage4_boolean",
        "stage4_registers",
        "stage4_replay_mapping",
        "stage4_replay_policy",
        "stage4_replay_stage3_logic_score",
        "stage4_replay_stage3_logic_choice",
        "stage4_replay_evidence",
        "stage4_replay_stage3_logic_noul",
        "stage4_replay_stage3_transition_set",
        "stage4_replay_rubric",
        "stage4_replay_authorization",
    }
)
SOURCES = frozenset({"legacy:stage4-general-composition-v2", "legacy:stage3_replay"})
SEED = "decision20-short-reasoning-replay-prereg-20260927"


def _group_candidates(
    legacy: list[dict[str, Any]],
    protected: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], Counter[str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in legacy:
        groups[row["group_id"]].append(row)
    ids = {row["id"] for row in protected}
    protected_groups = {row["group_id"] for row in protected if row.get("group_id")}
    inputs = {row["input_sha256"] for row in protected if row.get("input_sha256")}
    lengths: dict[str, int] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    rejected: Counter[str] = Counter()
    for group, rows in sorted(groups.items()):
        if any(
            row["family"] not in FAMILIES or row["source"] not in SOURCES
            for row in rows
        ):
            rejected["outside_eligible_family_or_source"] += len(rows)
            continue
        if any(
            row["id"] in ids
            or row["group_id"] in protected_groups
            or row["input_sha256"] in inputs
            for row in rows
        ):
            rejected["exact_partition_overlap"] += len(rows)
            continue
        row_lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in rows}
        if max(row_lengths.values()) > max_tokens:
            rejected["over_token_cap"] += len(rows)
            continue
        lengths.update(row_lengths)
        selected[group] = rows
    return selected, lengths, rejected


def _protected_prompt_contexts(
    inventory: Path, extra_prompt: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = json.loads(inventory.read_text(encoding="utf-8"))
    if not isinstance(sources, list):
        raise ValueError("Protected inventory must be an array")
    sources.append(
        {
            "kind": "prompts",
            "name": "authored-v4-release-goldfree",
            "path": str(extra_prompt),
        }
    )
    context = []
    evidence = []
    for source in sources:
        path = Path(source["path"])
        if source["kind"] == "training" and path.name not in (
            "select.jsonl",
            "cal.jsonl",
        ):
            continue
        if source["kind"] not in ("training", "prompts") or "gold" in path.name.lower():
            raise ValueError(
                f"Protected path is not a permitted prompt/partition: {path.name}"
            )
        observed = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if source["kind"] == "prompts" and set(row) != {
                    "id",
                    "state",
                    "questions",
                }:
                    raise ValueError("Gold-free protected prompt has unexpected fields")
                if not isinstance(row.get("state"), (str, dict, list)):
                    raise ValueError("Protected row has no valid context")
                context.append(
                    {
                        "id": f"{source['name']}/{row['id']}",
                        "state": row["state"],
                        "instructions": "",
                        "options": [],
                        "task_type": "context",
                    }
                )
                observed += 1
        evidence.append(
            {
                "name": source["name"],
                "kind": source["kind"],
                "rows": observed,
                "sha256": pilot.sha_file(path),
            }
        )
    return context, evidence


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    files = {
        "base": args.base_train,
        "base_manifest": args.base_manifest,
        "legacy": args.legacy_train,
        "legacy_manifest": args.legacy_manifest,
        "select": args.select_file,
        "cal": args.cal_file,
    }
    for role, path in files.items():
        if pilot.sha_file(path) != FROZEN[role]:
            raise ValueError(f"Frozen {role} SHA mismatch")
    base = load_partition(args.base_train, "train")
    legacy = load_partition(args.legacy_train, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    if (len(base), len(legacy), len(select), len(cal)) != (7455, 6000, 700, 700):
        raise ValueError("Frozen partition cardinality changed")
    legacy_manifest = json.loads(args.legacy_manifest.read_text(encoding="utf-8"))
    rights = {
        source: legacy_manifest["source_attribution"][source]
        for source in sorted(SOURCES)
    }
    if any(
        not all(
            record.get(field)
            for field in ("license", "attribution", "evidence", "rights_status")
        )
        for record in rights.values()
    ):
        raise ValueError("Legacy source rights evidence incomplete")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    grouped, lengths, rejected = _group_candidates(
        legacy, [*base, *select, *cal], tokenizer, args.max_row_tokens
    )
    refs, evidence = _protected_prompt_contexts(
        args.protected_list, args.extra_protected_prompt
    )
    all_protected = [
        *targeted.context_rows(base),
        *targeted.context_rows(select),
        *targeted.context_rows(cal),
        *refs,
    ]
    hashes = {targeted.text_hashes(row["state"]) for row in all_protected}
    raw = {pair[0] for pair in hashes}
    normalized = {pair[1] for pair in hashes}
    for group in list(grouped):
        if any(
            (pair := targeted.text_hashes(row["state"]))[0] in raw
            or pair[1] in normalized
            for row in grouped[group]
        ):
            rejected["exact_context"] += len(grouped.pop(group))
    pre_near = [row for rows in grouped.values() for row in rows]
    near = pilot.near_duplicates(
        targeted.context_rows(pre_near), all_protected, collect_left_ids=True
    )
    near_ids = set(near.pop("left_ids"))
    for group in list(grouped):
        if any(row["id"] in near_ids for row in grouped[group]):
            rejected["near_context"] += len(grouped.pop(group))
    added = [row for group in sorted(grouped) for row in grouped[group]]
    if not 256 <= len(added) <= 296:
        raise ValueError(
            f"Pre-registered short replay row count unavailable: {len(added)}"
        )
    base_tokens = sum(pilot.count_tokens(row, tokenizer) for row in base)
    added_tokens = sum(lengths[row["id"]] for row in added)
    if added_tokens / (base_tokens + added_tokens) > 0.20:
        raise ValueError("Added input-token share exceeds pre-registered 20%")
    merged = [*base, *added]
    merged.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{args.seed}\0shuffle\0{row['id']}".encode()),
            row["id"],
        )
    )
    check_partition_isolation({"train": merged, "select": select, "cal": cal})
    if pilot.train_consistency_audit(merged)["conflicting_gold_groups"]:
        raise ValueError("Merged TRAIN has conflicting labels for identical inputs")
    # Report all source-specific contexts without exposing held-out answers.
    overlap = targeted.context_overlap(added, all_protected, approximate=True)
    payloads = {
        "short_reasoning.train.jsonl": pilot.jsonl_bytes(merged),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(args.output_dir / name, payload)
    manifest = {
        "schema_version": "decision20-short-reasoning-replay/1",
        "builder_sha256": pilot.sha_file(Path(__file__)),
        "seed": args.seed,
        "inputs": {
            role: {"sha256": FROZEN[role], "basename": path.name}
            for role, path in files.items()
        },
        "protected_list_sha256": pilot.sha_file(args.protected_list),
        "protected_sources": evidence,
        "source_rights": rights,
        "selected_ids": sorted(row["id"] for row in added),
        "selected_group_ids": sorted(grouped),
        "selected_id_input_sha256": {row["id"]: row["input_sha256"] for row in added},
        "rejected_rows": dict(sorted(rejected.items())),
        "near_before_quarantine": near,
        "post_quarantine_overlap": overlap,
        "added_counts": {
            field: dict(sorted(Counter(row[field] for row in added).items()))
            for field in ("task_type", "family", "source", "language")
        },
        "token_audit": {
            "tokenizer_revision": args.tokenizer_revision,
            "max_row_tokens": args.max_row_tokens,
            "base_tokens": base_tokens,
            "added_tokens": added_tokens,
            "added_share": added_tokens / (base_tokens + added_tokens),
        },
        "outputs": {
            name: {"sha256": pilot.sha_bytes(data), "rows": len(data.splitlines())}
            for name, data in payloads.items()
        },
        "limits": [
            "Approximate SimHash candidate screening can miss paraphrases.",
            "Legacy source rights are retained; noncommercial research scope and attribution apply.",
            "No sealed FINAL or CSS15 gold was read.",
        ],
    }
    pilot._atomic_write(
        args.output_dir / "manifest.json",
        (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-train",
        "base-manifest",
        "legacy-train",
        "legacy-manifest",
        "select-file",
        "cal-file",
        "tokenizer",
        "protected-list",
        "extra-protected-prompt",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=1024)
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()
    report = build(args)
    print(
        pilot.canonical(
            {
                "rows": report["outputs"]["short_reasoning.train.jsonl"]["rows"],
                "sha256": report["outputs"]["short_reasoning.train.jsonl"]["sha256"],
                "added": len(report["selected_ids"]),
                "added_token_share": report["token_audit"]["added_share"],
            }
        )
    )


if __name__ == "__main__":
    main()
