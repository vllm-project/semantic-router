"""Gold-free admission audit for the frozen Score v8.3 CPU pilot."""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
from pathlib import Path
from typing import Any

from training.data import score_v7p_admit as overlap
from training.data import score_v8_pilot_audit as v8_audit
from training.data import score_v82_pilot_audit as v82_audit
from training.data import score_v83_pilot as pilot
from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)

SCHEMA = "decision2-score-v8.3-pilot-audit/1"
NORMALIZE_ID = re.compile(
    r"\b(?:JOB|HANDOFF|CONN|IN|OUT|OTHER|PICK|SKU)-[A-Z]+\d+\b", re.I
)
NORMALIZE_NUMBER = re.compile(r"\d+(?::\d+)?")


def _candidate(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    expected = pilot.GROUPS[role] * len(pilot.MECHANISMS) * 3
    if len(rows) != expected:
        raise ValueError(f"{role}: expected {expected} rows")
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    if len(by_group) != pilot.GROUPS[role] * len(pilot.MECHANISMS):
        raise ValueError(f"{role}: incomplete group count")
    mechanisms = collections.Counter(row["audit_metadata"]["mechanism"] for row in rows)
    if mechanisms != {name: pilot.GROUPS[role] * 3 for name in pilot.MECHANISMS}:
        raise ValueError(f"{role}: mechanism balance changed")
    if len({row["audit_metadata"]["case"] for row in rows}) != len(by_group):
        raise ValueError(f"{role}: source cases reused")
    for group, items in by_group.items():
        if len(items) != 3 or {row["label"] for row in items} != set(pilot.LEVELS):
            raise ValueError(f"{role}: invalid triplet {group}")
        for row in items:
            meta = row["audit_metadata"]
            if row["source"] != pilot.VERSION or row["language"] != "en":
                raise ValueError(f"{role}: wrong source or language")
            if (
                pilot.rendered_oracle(row["state"], meta["mechanism"], meta["case"])
                != row["label"]
            ):
                raise ValueError(f"{role}: rendered answer disagrees for {row['id']}")
    return {
        "rows": len(rows),
        "groups": len(by_group),
        "levels": dict(
            sorted(collections.Counter(str(row["label"]) for row in rows).items())
        ),
        "mechanisms": dict(sorted(mechanisms.items())),
        "state_chars": [
            min(len(row["state"]) for row in rows),
            max(len(row["state"]) for row in rows),
        ],
    }


def _normalize(text: str) -> str:
    return NORMALIZE_NUMBER.sub("n", NORMALIZE_ID.sub("id", text.lower()))


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z]+", text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _candidate_near(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = {}
    for row in rows:
        if row["label"] == 0:
            first[row["group_id"]] = row
    if len(first) != 18:
        raise ValueError("Expected 18 first-variant groups")
    pairs = []
    keys = sorted(first)
    for i, a in enumerate(keys):
        ta = _normalize(first[a]["state"])
        ga = _ngrams(ta)
        for b in keys[i + 1 :]:
            tb = _normalize(first[b]["state"])
            gb = _ngrams(tb)
            sequence = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
            jaccard = len(ga & gb) / len(ga | gb) if ga or gb else 1.0
            if sequence >= 0.88 or jaccard >= 0.60:
                pairs.append(
                    {
                        "group_a": a,
                        "group_b": b,
                        "sequence": round(sequence, 5),
                        "fivegram_jaccard": round(jaccard, 5),
                    }
                )
    return pairs


def _single_field_projections(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check each mechanism's one-field projections within its triplets."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    flags = []
    for group, triplet in groups.items():
        mechanism = triplet[0]["audit_metadata"]["mechanism"]
        values = []
        for row in sorted(triplet, key=lambda item: item["label"]):
            state = row["state"]
            case = re.escape(row["audit_metadata"]["case"])
            if mechanism == "dependency":
                cards = re.search(rf"Status cards \[{case}\]: ([^\n]+)\.", state)
                if cards is None:
                    raise ValueError("Cannot project dependency statuses")
                values.append(tuple(re.findall(r"=(failed|queued|complete)", cards[1])))
            elif mechanism == "connection":
                arrival = re.search(
                    rf"Service update \[{case}\]: [^\n]*?arrival window (\d\d:\d\d) to (\d\d:\d\d)",
                    state,
                )
                if arrival is None:
                    raise ValueError("Cannot project arrival endpoints")
                values.append(arrival.group(1, 2))
            else:
                stock = re.search(
                    rf"\[{case}\]: [^\n]*?counted (\d+) on hand; (\d+) already reserved",
                    state,
                )
                inbound = re.search(
                    rf"\[{case}\]: [^\n]*?has (\d+) confirmed incoming units and (\d+) tentative units",
                    state,
                )
                if stock is None or inbound is None:
                    raise ValueError("Cannot project stock fields")
                values.append((stock[1], stock[2], inbound[1], inbound[2]))
        width = len(values[0])
        if any(len(v) != width for v in values):
            raise ValueError("Field projection width varies")
        for col in range(width):
            if len({v[col] for v in values}) == 3:
                flags.append({"group": group, "field_index": col})
    return {"groups": len(groups), "single_field_decoders": flags}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.candidate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != pilot.VERSION:
        raise ValueError("Candidate version changed")
    for role in pilot.GROUPS:
        for kind, filename in (
            ("rows", f"{role}.jsonl"),
            ("packet", f"{role}-blind-packet.jsonl"),
            ("key", f"{role}-sealed-key.json"),
        ):
            if manifest["roles"][role][f"{kind}_sha256"] != file_sha256(
                args.candidate_dir / filename
            ):
                raise ValueError(f"{role}: sealed {kind} bytes changed")
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
    if {row["audit_metadata"]["case"] for row in train} & {
        row["audit_metadata"]["case"] for row in select
    }:
        raise ValueError("TRAIN and SELECT cases overlap")
    prior_v7p = load_partition(args.prior_v7p_train, "train")
    references: dict[str, list[dict[str, Any]]] = {
        "candidate_select": select,
        "parent_train": parent_train,
        "parent_select": parent_select,
        "parent_cal": parent_cal,
        "prior_v7p_train": prior_v7p,
        "prior_v7p_select_goldfree": v8_audit._blind_rows(args.prior_v7p_select_packet),
        "prior_v8_train": load_partition(args.prior_v8_train, "train"),
        "prior_v8_select_goldfree": v82_audit._prior_blind(
            args.prior_v8_select_packet, 30
        ),
        "prior_v81_train": load_partition(args.prior_v81_train, "train"),
        "prior_v81_select_goldfree": v82_audit._prior_blind(
            args.prior_v81_select_packet, 30
        ),
        "prior_v82_train": load_partition(args.prior_v82_train, "train"),
        "prior_v82_select_goldfree": v82_audit._prior_blind(
            args.prior_v82_select_packet, 30
        ),
    }
    protected, receipts = overlap._protected(args.protected_list)
    references.update(protected)
    findings = {}
    for role, candidates in (("train", train), ("select", select)):
        for name, reference in references.items():
            if role == "select" and name == "candidate_select":
                continue
            finding = overlap._overlap(candidates, reference)
            findings[f"{role}_vs_{name}"] = {
                key: finding[key]
                for key in (
                    "matched_rows",
                    "matched_groups",
                    "near_context",
                    "near_full",
                )
            }
    flagged = sorted(
        name for name, finding in findings.items() if finding["matched_rows"]
    )
    candidate_near = _candidate_near([*train, *select])
    single_fields = _single_field_projections([*train, *select])
    status = (
        "HOLD_OVERLAP"
        if flagged or candidate_near
        else (
            "HOLD_SHORTCUT"
            if single_fields["single_field_decoders"]
            else "PENDING_INDEPENDENT_BLIND_REVIEW"
        )
    )
    source_paths = {
        name: getattr(args, name)
        for name in (
            "parent_train",
            "parent_select",
            "parent_cal",
            "protected_list",
            "prior_v7p_train",
            "prior_v7p_select_packet",
            "prior_v8_train",
            "prior_v8_select_packet",
            "prior_v81_train",
            "prior_v81_select_packet",
            "prior_v82_train",
            "prior_v82_select_packet",
        )
    }
    return {
        "schema_version": SCHEMA,
        "status": status,
        "candidate_manifest_sha256": file_sha256(manifest_path),
        "candidate": {
            "train": _candidate(train, "train"),
            "select": _candidate(select, "select"),
        },
        "source_sha256": {
            name: file_sha256(path) for name, path in source_paths.items()
        },
        "protected_roles": receipts,
        "overlap": findings,
        "flagged_comparisons": flagged,
        "within_candidate_near_pairs": candidate_near,
        "single_field_projections": single_fields,
        "limitations": [
            "Small English-only synthetic pilot",
            "Bounded text overlap is not semantic independence",
            "Independent rendered-text blind review remains required",
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
        "prior-v82-train",
        "prior-v82-select-packet",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output must be fresh")
    result = audit(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate": result["candidate"],
                "flagged_comparisons": result["flagged_comparisons"],
                "candidate_near_pairs": len(result["within_candidate_near_pairs"]),
                "single_field_decoders": len(
                    result["single_field_projections"]["single_field_decoders"]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
