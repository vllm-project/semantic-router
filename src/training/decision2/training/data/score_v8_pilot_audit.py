"""CPU-only source separation and gold-free overlap audit for Score v8.

Only the v8 candidate has labels. Protected benchmark references and the old
v7p SELECT blind packet are consumed as prompts without any answer key.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from training.data import score_v7p_admit as prior_audit
from training.data.score_v8_pilot import (
    GROUPS,
    LEVELS,
    MECHANISMS,
    VERSION,
    rendered_oracle,
)
from training.model.data import (
    check_partition_isolation,
    digest,
    file_sha256,
    load_partition,
)

SCHEMA = "decision2-score-v8-pilot-audit/1"
GLOBAL_CUES = {
    "dated_update": ("suspended", "review", "active", "unsigned"),
    "numeric_limits": ("NA", "missing", "current"),
    "evidence_sufficiency": (
        "normal observed",
        "unavailable observed",
        "outage observed",
    ),
    "scoped_exception": ("pending", "signed"),
    "long_memo": ("unconfirmed", "confirmed", "controlling deadline"),
}


def _blind_rows(path: Path) -> list[dict[str, Any]]:
    """Flatten a gold-free prior review packet; never open the sealed key."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            group = json.loads(line)
            for item in group["items"]:
                core = {
                    "state": item["state"],
                    "instructions": item["instructions"],
                    "options": item["options"],
                    "task_type": "score",
                }
                rows.append(
                    {
                        "id": item["review_id"],
                        "group_id": group["review_group"],
                        **core,
                        "input_sha256": digest(core),
                    }
                )
    if len(rows) != 240:
        raise ValueError("Prior SELECT blind packet count differs")
    return rows


def _candidate(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    expected = GROUPS[role] * len(MECHANISMS) * len(LEVELS)
    if len(rows) != expected:
        raise ValueError(f"{role}: candidate cardinality differs")
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    counts = collections.Counter(row["family"] for row in rows)
    if counts != {f"score_v8_{name}": GROUPS[role] * 3 for name in MECHANISMS}:
        raise ValueError(f"{role}: five mechanisms are not balanced")
    if len(groups) != GROUPS[role] * len(MECHANISMS):
        raise ValueError(f"{role}: group count differs")
    for group, triplet in groups.items():
        if len(triplet) != 3 or {row["label"] for row in triplet} != set(LEVELS):
            raise ValueError(f"{role}: incomplete triplet {group}")
        if len({row["audit_metadata"]["record"] for row in triplet}) != 1:
            raise ValueError(f"{role}: target record changed in triplet {group}")
        for row in triplet:
            if row["source"] != VERSION or row["language"] != "en":
                raise ValueError(f"{role}: source or language differs")
            meta = row["audit_metadata"]
            verdict = rendered_oracle(
                row["state"],
                meta["mechanism"],
                meta["record"],
                meta["review_day"],
                site=meta["site"],
                activity=meta["activity"],
            )
            if verdict != row["label"]:
                raise ValueError(f"{role}: rendered oracle mismatch")
    lengths = collections.defaultdict(list)
    for row in rows:
        lengths[row["label"]].append(len(row["state"]))
    return {
        "rows": len(rows),
        "groups": len(groups),
        "level_counts": dict(collections.Counter(row["label"] for row in rows)),
        "mechanism_counts": dict(sorted(counts.items())),
        "state_chars_by_level": {
            str(level): [min(lengths[level]), max(lengths[level])] for level in LEVELS
        },
        "long_memo_chars": [
            min(
                len(row["state"])
                for row in rows
                if row["family"] == "score_v8_long_memo"
            ),
            max(
                len(row["state"])
                for row in rows
                if row["family"] == "score_v8_long_memo"
            ),
        ],
    }


def _shortcut_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mechanism, cues in GLOBAL_CUES.items():
        subset = [
            row for row in rows if row["audit_metadata"]["mechanism"] == mechanism
        ]
        cue_counts: dict[str, dict[str, list[int]]] = {}
        flags = []
        for cue in cues:
            by_level = {
                str(level): sorted(
                    row["state"].count(cue) for row in subset if row["label"] == level
                )
                for level in LEVELS
            }
            cue_counts[cue] = by_level
            for level in LEVELS:
                own = set(by_level[str(level)])
                others = {
                    value
                    for other in LEVELS
                    if other != level
                    for value in by_level[str(other)]
                }
                if len(own) == 1 and own.isdisjoint(others):
                    flags.append(
                        {"cue": cue, "level": level, "global_count": next(iter(own))}
                    )
        summary[mechanism] = {
            "global_cue_counts_by_level": cue_counts,
            "single_level_global_cues": flags,
        }
    return summary


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(
        (args.candidate_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("status") != "PENDING_INDEPENDENT_BLIND_REVIEW"
    ):
        raise ValueError("v8 candidate manifest differs")
    train_path = args.candidate_dir / "train.jsonl"
    select_path = args.candidate_dir / "select.jsonl"
    for role, path in (("train", train_path), ("select", select_path)):
        if manifest["roles"][role]["rows_sha256"] != file_sha256(path):
            raise ValueError(f"{role} candidate hash differs")
        for name, suffix in (
            ("packet", "-blind-packet.jsonl"),
            ("key", "-sealed-key.json"),
        ):
            if manifest["roles"][role][f"{name}_sha256"] != file_sha256(
                args.candidate_dir / f"{role}{suffix}"
            ):
                raise ValueError(f"{role} {name} hash differs")
    pinned = {
        "parent_train": (args.parent_train, prior_audit.PARENT_SHA),
        "parent_select": (args.parent_select, prior_audit.SELECT_SHA),
        "parent_cal": (args.parent_cal, prior_audit.CAL_SHA),
    }
    for name, (path, expected) in pinned.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{name} frozen identity changed")
    if file_sha256(args.protected_list) != args.protected_sha256:
        raise ValueError("Gold-free protected inventory identity differs")
    train = load_partition(train_path, "train")
    select = load_partition(select_path, "select")
    parent_train = load_partition(args.parent_train, "train")
    parent_select = load_partition(args.parent_select, "select")
    parent_cal = load_partition(args.parent_cal, "cal")
    check_partition_isolation(
        {
            "train": [*parent_train, *train],
            "select": [*parent_select, *select],
            "cal": parent_cal,
        }
    )
    prior_arm = load_partition(args.prior_arm, "train")
    prior_train = [
        row
        for row in prior_arm
        if row["source"] == "decision2-score-v7p-evidence-state/1"
    ]
    if len(prior_train) != 240:
        raise ValueError("Prior v7p admitted arm lacks its candidate 240")
    protected, protected_receipts = prior_audit._protected(args.protected_list)
    references = {
        "candidate_select": select,
        "parent_train": parent_train,
        "parent_select": parent_select,
        "parent_cal": parent_cal,
        "prior_v7p_train": prior_train,
        "prior_v7p_select_goldfree": _blind_rows(args.prior_select_packet),
        **protected,
    }
    findings: dict[str, Any] = {}
    for role, candidates in (("train", train), ("select", select)):
        for name, reference in references.items():
            if role == "select" and name == "candidate_select":
                continue
            if role == "train" and name == "candidate_select":
                name = "candidate_train_vs_select"
            finding = prior_audit._overlap(candidates, reference)
            findings[f"{role}_vs_{name}"] = {
                "matched_rows": finding["matched_rows"],
                "matched_groups": finding["matched_groups"],
                "near_context": finding["near_context"],
                "near_full": finding["near_full"],
            }
    flagged = {name: value for name, value in findings.items() if value["matched_rows"]}
    shortcuts = _shortcut_inventory([*train, *select])
    shortcut_flags = {
        mechanism: value["single_level_global_cues"]
        for mechanism, value in shortcuts.items()
        if value["single_level_global_cues"]
    }
    return {
        "schema_version": SCHEMA,
        "status": (
            "HOLD_OVERLAP_REVIEW"
            if flagged
            else (
                "HOLD_SHORTCUT"
                if shortcut_flags
                else "PENDING_INDEPENDENT_BLIND_REVIEW"
            )
        ),
        "source_sha256": {
            "candidate_manifest": file_sha256(args.candidate_dir / "manifest.json"),
            "parent_train": prior_audit.PARENT_SHA,
            "parent_select": prior_audit.SELECT_SHA,
            "parent_cal": prior_audit.CAL_SHA,
            "protected_inventory": args.protected_sha256,
            "prior_arm": file_sha256(args.prior_arm),
            "prior_select_packet": file_sha256(args.prior_select_packet),
        },
        "candidate": {
            "train": _candidate(train, "train"),
            "select": _candidate(select, "select"),
        },
        "protected_roles": protected_receipts,
        "overlap": findings,
        "flagged_comparisons": sorted(flagged),
        "shortcut_inventory": shortcuts,
        "flagged_global_cues": shortcut_flags,
        "limitations": [
            "Mechanism families remain shared across TRAIN and SELECT even when source records differ",
            "Approximate near-text matching cannot prove semantic independence",
            "The builder's two oracles are not an independent blind editorial review",
            "English only; no multilingual evidence",
            "This 75-row pilot cannot support a training-effect or release claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate-dir",
        "parent-train",
        "parent-select",
        "parent-cal",
        "protected-list",
        "prior-arm",
        "prior-select-packet",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--protected-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output already exists")
    report = audit(args)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate": report["candidate"],
                "flagged_comparisons": report["flagged_comparisons"],
                "flagged_global_cues": report["flagged_global_cues"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
