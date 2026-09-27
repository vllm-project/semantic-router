"""CPU-only, gold-free Score v8.2 source and shortcut admission audit."""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
from pathlib import Path
from typing import Any

from training.data import score_v7p_admit as overlap
from training.data import score_v8_pilot as previous
from training.data import score_v8_pilot_audit as previous_audit
from training.data import score_v82_pilot as pilot
from training.model.data import (
    check_partition_isolation,
    digest,
    file_sha256,
    load_partition,
)

SCHEMA = "decision2-score-v8.2-pilot-audit/1"


def _prior_blind(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
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
    if len(rows) != expected_rows:
        raise ValueError(
            f"Prior blind packet expected {expected_rows}, found {len(rows)}"
        )
    return rows


def _candidate(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    expected = pilot.GROUPS[role] * len(pilot.MECHANISMS) * len(pilot.LEVELS)
    if len(rows) != expected:
        raise ValueError(f"{role}: candidate row count differs")
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(groups) != pilot.GROUPS[role] * len(pilot.MECHANISMS):
        raise ValueError(f"{role}: group count differs")
    by_mechanism = collections.Counter(
        row["audit_metadata"]["mechanism"] for row in rows
    )
    if by_mechanism != {name: pilot.GROUPS[role] * 3 for name in pilot.MECHANISMS}:
        raise ValueError(f"{role}: mechanism mixture differs")
    for group, items in groups.items():
        if len(items) != 3 or {item["label"] for item in items} != set(pilot.LEVELS):
            raise ValueError(f"{role}: incomplete triplet {group}")
        if len({item["audit_metadata"]["record"] for item in items}) != 1:
            raise ValueError(f"{role}: named record changed within triplet")
        for row in items:
            if row["source"] != pilot.VERSION or row["language"] != "en":
                raise ValueError(f"{role}: source or language changed")
            meta = row["audit_metadata"]
            rendered = (
                pilot._rendered_evidence_oracle(row["state"], meta["record"])
                if meta["mechanism"] == "evidence_sufficiency"
                else previous.rendered_oracle(
                    row["state"],
                    meta["mechanism"],
                    meta["record"],
                    meta["review_day"],
                    site=meta["site"],
                    activity=meta["activity"],
                )
            )
            if rendered != row["label"]:
                raise ValueError(f"{role}: rendered answer differs for {row['id']}")
    return {
        "rows": len(rows),
        "groups": len(groups),
        "level_counts": dict(collections.Counter(row["label"] for row in rows)),
        "mechanism_counts": dict(sorted(by_mechanism.items())),
    }


def _source_necessity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["audit_metadata"]["mechanism"] == "evidence_sufficiency":
            groups[row["group_id"]].append(row)
    result = {}
    for group, items in groups.items():
        target_values = []
        decoys: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        for row in items:
            target = row["audit_metadata"]["record"]
            matches = re.findall(
                r"(?m)^Dispatch diary for (R-[A-Z0-9]+), hours (\d+-\d+): (outage|normal|unavailable) observed\.\n"
                r"Meter trace for \1, hours \2: (outage|normal|unavailable) observed\.$",
                row["state"],
            )
            selected = [entry for entry in matches if entry[0] == target]
            if len(matches) != 4 or len(selected) != 1:
                raise ValueError(f"{group}: evidence sources cannot be joined")
            target_values.append((row["label"], selected[0][2], selected[0][3]))
            for record, _, dispatch, meter in matches:
                if record != target:
                    decoys[record].append((dispatch, meter))
        target_values.sort()
        dispatch = [entry[1] for entry in target_values]
        meter = [entry[2] for entry in target_values]
        if not (1 < len(set(dispatch)) < 3 and 1 < len(set(meter)) < 3):
            raise ValueError(
                f"{group}: a single source decodes all levels or is constant"
            )
        # A stable distractor cannot by itself complement the target verdict.
        # A changing record with one observation is insufficient to decode a
        # three-level group. This is a narrow shortcut check, not a proof.
        if any(
            len(values) == 3 and len(set(values)) == 3 for values in decoys.values()
        ):
            raise ValueError(f"{group}: a distractor's three states identify the level")
        result[group] = {
            "dispatch_projection": dispatch,
            "meter_projection": meter,
            "non_target_records": len(decoys),
            "both_single_source_projections_ambiguous": True,
        }
    if len(result) != 5:
        raise ValueError("Expected five evidence triplets")
    return result


def _document_realism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    documents = [
        row
        for row in rows
        if row["audit_metadata"]["mechanism"] == "long_memo" and row["label"] == 0
    ]
    if len(documents) != 5:
        raise ValueError("Expected five long-document source groups")
    groups = []
    for row in documents:
        lengths = [
            len(item["state"]) for item in rows if item["group_id"] == row["group_id"]
        ]
        if len(lengths) != 3 or min(lengths) < 1500:
            raise ValueError(
                f"{row['group_id']}: document shorter than preregistered floor"
            )
        groups.append(
            {
                "group": row["group_id"],
                "min_chars": min(lengths),
                "max_chars": max(lengths),
            }
        )
    normalize = lambda s: re.sub(r"\d+", "N", re.sub(r"R-[A-Z0-9]+", "R-X", s))
    pairwise = [
        difflib.SequenceMatcher(
            None, normalize(a["state"]), normalize(b["state"])
        ).ratio()
        for i, a in enumerate(documents)
        for b in documents[i + 1 :]
    ]
    if max(pairwise) >= 0.85:
        raise ValueError("Long-document template similarity exceeds frozen ceiling")
    return {
        "groups": groups,
        "normalized_pair_similarity_min": min(pairwise),
        "normalized_pair_similarity_max": max(pairwise),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.candidate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != pilot.VERSION:
        raise ValueError("Candidate version differs")
    for role in pilot.GROUPS:
        for key, filename in (
            ("rows", f"{role}.jsonl"),
            ("packet", f"{role}-blind-packet.jsonl"),
            ("key", f"{role}-sealed-key.json"),
        ):
            if manifest["roles"][role][f"{key}_sha256"] != file_sha256(
                args.candidate_dir / filename
            ):
                raise ValueError(f"{role} {key} bytes differ from manifest")
    pins = {
        "parent_train": (args.parent_train, overlap.PARENT_SHA),
        "parent_select": (args.parent_select, overlap.SELECT_SHA),
        "parent_cal": (args.parent_cal, overlap.CAL_SHA),
    }
    for name, (path, expected) in pins.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{name} identity changed")
    if file_sha256(args.protected_list) != args.protected_sha256:
        raise ValueError("Protected gold-free inventory changed")
    train = load_partition(args.candidate_dir / "train.jsonl", "train")
    select = load_partition(args.candidate_dir / "select.jsonl", "select")
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
    prior_arm = load_partition(args.prior_v7p_train, "train")
    references: dict[str, list[dict[str, Any]]] = {
        "candidate_select": select,
        "parent_train": parent_train,
        "parent_select": parent_select,
        "parent_cal": parent_cal,
        "prior_v7p_train": [
            row
            for row in prior_arm
            if row["source"] == "decision2-score-v7p-evidence-state/1"
        ],
        "prior_v7p_select_goldfree": previous_audit._blind_rows(
            args.prior_v7p_select_packet
        ),
        "prior_v8_train": load_partition(args.prior_v8_train, "train"),
        "prior_v8_select_goldfree": _prior_blind(args.prior_v8_select_packet, 30),
        "prior_v81_train": load_partition(args.prior_v81_train, "train"),
        "prior_v81_select_goldfree": _prior_blind(args.prior_v81_select_packet, 30),
    }
    protected, protected_receipts = overlap._protected(args.protected_list)
    references.update(protected)
    findings = {}
    for role, candidates in (("train", train), ("select", select)):
        for name, reference in references.items():
            if role == "select" and name == "candidate_select":
                continue
            finding = overlap._overlap(candidates, reference)
            findings[f"{role}_vs_{name}"] = {
                field: finding[field]
                for field in (
                    "matched_rows",
                    "matched_groups",
                    "near_context",
                    "near_full",
                )
            }
    flagged = sorted(name for name, value in findings.items() if value["matched_rows"])
    candidate = {
        "train": _candidate(train, "train"),
        "select": _candidate(select, "select"),
    }
    source_necessity = _source_necessity([*train, *select])
    document_realism = _document_realism([*train, *select])
    shortcut_inventory = previous_audit._shortcut_inventory([*train, *select])
    shortcut_flags = {
        mechanism: value["single_level_global_cues"]
        for mechanism, value in shortcut_inventory.items()
        if value["single_level_global_cues"]
    }
    return {
        "schema_version": SCHEMA,
        "status": (
            "HOLD_OVERLAP"
            if flagged
            else (
                "HOLD_SHORTCUT"
                if shortcut_flags
                else "PENDING_INDEPENDENT_BLIND_REVIEW"
            )
        ),
        "candidate_manifest_sha256": file_sha256(manifest_path),
        "source_sha256": {
            name: file_sha256(path)
            for name, path in (
                ("parent_train", args.parent_train),
                ("parent_select", args.parent_select),
                ("parent_cal", args.parent_cal),
                ("protected_inventory", args.protected_list),
                ("prior_v7p_train", args.prior_v7p_train),
                ("prior_v7p_select", args.prior_v7p_select_packet),
                ("prior_v8_train", args.prior_v8_train),
                ("prior_v8_select", args.prior_v8_select_packet),
                ("prior_v81_train", args.prior_v81_train),
                ("prior_v81_select", args.prior_v81_select_packet),
            )
        },
        "candidate": candidate,
        "protected_roles": protected_receipts,
        "overlap": findings,
        "flagged_comparisons": flagged,
        "source_necessity": source_necessity,
        "document_realism": document_realism,
        "shortcut_inventory": shortcut_inventory,
        "flagged_global_cues": shortcut_flags,
        "limitations": [
            "English-only small synthetic pilot; no training or transfer claim",
            "Group-level source projection and lexical counts are bounded shortcut checks",
            "Near-text matching cannot certify semantic independence",
            "Independent rendered blind review still required",
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
        "prior-v7p-train",
        "prior-v7p-select-packet",
        "prior-v8-train",
        "prior-v8-select-packet",
        "prior-v81-train",
        "prior-v81-select-packet",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--protected-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output must be fresh")
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
                "document_similarity_max": report["document_realism"][
                    "normalized_pair_similarity_max"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
