"""CPU-only frozen Score v8.5 quality audit; no model or protected key access."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from training.data import score_v85_pilot as pilot
from training.data.score_v84_audit import (
    _cross_group_near,
    _load_protected,
    _reference_overlap,
    _references,
)
from training.model.data import file_sha256

SCHEMA = "decision2-score-v8.5-deep-dossier-audit/1"
REFERENCE_LIST_SHA = "9964d99aedac3085133c1d59ca4bd9cab5a8cad58f475458333e641bdbf374f7"
TOKENIZER_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
TOKENIZER_CONFIG_SHA = (
    "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    candidate = args.candidate_dir
    manifest = json.loads(
        (candidate / "private-manifest.json").read_text(encoding="utf-8")
    )
    if manifest["schema_version"] != pilot.VERSION:
        raise ValueError("Candidate version changed")
    if args.seed_file.stat().st_mode & 0o077:
        raise PermissionError("Seed file is not private")
    seed = args.seed_file.read_bytes()
    if hashlib.sha256(seed).hexdigest() != manifest["seed_sha256"]:
        raise ValueError("Seed does not match frozen manifest")
    roles = {}
    for role in pilot.GROUPS:
        path = candidate / f"{role}.jsonl"
        if file_sha256(path) != manifest["roles"][role]["sha256"]:
            raise ValueError(f"{role} bytes changed")
        roles[role] = _rows(path)
        rebuilt = pilot.build(seed, role)
        if roles[role] != rebuilt:
            raise ValueError(
                f"{role} differs from deterministic source and rendered oracles"
            )
    all_rows = [*roles["train"], *roles["select"]]
    if len(all_rows) != 90:
        raise ValueError("Expected 90 rows")
    group_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in all_rows:
        group_rows[row["group_id"]].append(row)
    if (
        len(group_rows) != 30
        or len({r["audit_metadata"]["case"] for r in all_rows}) != 30
    ):
        raise ValueError("Source cases or groups are not isolated")
    group_roles = collections.defaultdict(set)
    per_mechanism = collections.defaultdict(set)
    deep_positions = set()
    long_groups = set()
    for gid, triplet in group_rows.items():
        if len(triplet) != 3 or {r["label"] for r in triplet} != {0, 1, 2}:
            raise ValueError(f"Triplet incomplete: {gid}")
        if (
            triplet[0]["audit_metadata"]["long"]
            and len(
                {
                    r["state"].split(
                        (
                            "随案补充材料"
                            if r["language"] == "zh"
                            else "Filed case dossier"
                        ),
                        1,
                    )[0]
                    for r in triplet
                }
            )
            != 1
        ):
            raise ValueError(f"Core changed across counterfactuals: {gid}")
        for row in triplet:
            meta = row["audit_metadata"]
            group_roles[gid].add(row["split"])
            per_mechanism[meta["mechanism"]].add(gid)
            if meta["long"]:
                long_groups.add(gid)
                deep_positions.add(meta["deep_position"])
                if meta["deep_position"] < 6 or meta["deep_position"] > 21:
                    raise ValueError(f"Decisive source not deep: {gid}")
                dossier = row["state"].split(
                    "随案补充材料" if row["language"] == "zh" else "Filed case dossier",
                    1,
                )[1]
                if meta["target"] not in dossier or meta["near"] not in dossier:
                    raise ValueError(
                        f"Target or near-field dossier record missing: {gid}"
                    )
                core = row["state"].split(
                    "随案补充材料" if row["language"] == "zh" else "Filed case dossier",
                    1,
                )[0]
                try:
                    pilot.rendered_oracle(
                        core, meta["mechanism"], meta["case"], meta["target"]
                    )
                except ValueError:
                    pass
                else:
                    raise ValueError(f"Core alone still decides long item: {gid}")
            if (
                pilot.rendered_oracle(
                    row["state"], meta["mechanism"], meta["case"], meta["target"]
                )
                != row["label"]
            ):
                raise ValueError(f"Rendered oracle differs: {gid}")
    if any(len(v) != 1 for v in group_roles.values()) or any(
        len(v) != 5 for v in per_mechanism.values()
    ):
        raise ValueError("Group split or mechanism coverage changed")
    if len(long_groups) != 24 or len(deep_positions) < 4:
        raise ValueError("Deep dossier coverage or position randomization failed")
    for role, n in pilot.GROUPS.items():
        groups = {r["group_id"] for r in roles[role]}
        if len(groups) != n * len(pilot.MECHANISMS):
            raise ValueError(f"{role} group count changed")
    packet_path = candidate / "blind-packet.jsonl"
    key_path = candidate / "sealed-key.json"
    if (
        file_sha256(packet_path) != manifest["packet_sha256"]
        or file_sha256(key_path) != manifest["key_sha256"]
    ):
        raise ValueError("Blind packet or key changed")
    packet = _rows(packet_path)
    if len(packet) != 30:
        raise ValueError("Blind packet has wrong group count")
    review_ids = set()
    for group in packet:
        if set(group) != {"review_group", "items"} or len(group["items"]) != 3:
            raise ValueError("Blind group schema leaks metadata or is incomplete")
        for item in group["items"]:
            if set(item) != {"review_id", "state", "instructions", "options"}:
                raise ValueError("Blind item schema leaks metadata")
            review_ids.add(item["review_id"])
    if len(review_ids) != 90:
        raise ValueError("Review IDs are duplicated")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if set(key) != review_ids:
        raise ValueError("Key and blind packet IDs differ")
    if (
        file_sha256(args.tokenizer_dir / "tokenizer.json") != TOKENIZER_SHA
        or file_sha256(args.tokenizer_dir / "tokenizer_config.json")
        != TOKENIZER_CONFIG_SHA
    ):
        raise ValueError("Frozen tokenizer bytes changed")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    tokens = {
        r["id"]: len(
            tokenizer(
                r["state"]
                + r["instructions"]
                + json.dumps(r["options"], ensure_ascii=False),
                add_special_tokens=False,
            ).input_ids
        )
        for r in all_rows
    }
    token_summary = {
        "min": min(tokens.values()),
        "max": max(tokens.values()),
        "over_700": sum(n > 700 for n in tokens.values()),
        "over_1500": sum(n > 1500 for n in tokens.values()),
    }
    copy_fault_groups = set()
    for gid, triplet in group_rows.items():
        row = triplet[0]
        if row["language"] == "en":
            core = row["state"].split("Filed case dossier", 1)[0]
            if re.search(r";[A-Za-z]|\.[A-Z]|:[A-Z]|,[a-z]", core):
                copy_fault_groups.add(gid)
    if file_sha256(args.reference_list) != REFERENCE_LIST_SHA:
        raise ValueError("Frozen v8.4 reference list changed")
    prior = json.loads(args.prior_audit.read_text(encoding="utf-8"))
    reference, reference_hashes = _references(args.reference_list)
    for name, sha in reference_hashes.items():
        if prior["source_sha256"].get(name) != sha:
            raise ValueError(f"Frozen reference target changed: {name}")
    protected, protected_receipts = _load_protected(args.protected_list)
    comparisons = {}
    for role, rows in roles.items():
        for name, ref in {
            "other_candidate_role": roles["select" if role == "train" else "train"],
            **reference,
            **protected,
        }.items():
            if role == "select" and name == "other_candidate_role":
                continue
            comparisons[f"{role}_vs_{name}"] = _reference_overlap(rows, ref)
    flagged = {
        name: value for name, value in comparisons.items() if value["flagged_groups"]
    }
    within_near = _cross_group_near(all_rows)
    exclusions = collections.defaultdict(set)
    for name, result in flagged.items():
        for gid in result["flagged_groups"]:
            exclusions[gid].add(name)
    for pair in within_near:
        exclusions[pair["group_a"]].add("cross_group_near")
        exclusions[pair["group_b"]].add("cross_group_near")
    if token_summary["over_700"] < 24 or token_summary["over_1500"] < 6:
        for gid in group_rows:
            exclusions[gid].add("preregistered_length_gate")
    if copy_fault_groups:
        for gid in copy_fault_groups:
            exclusions[gid].add("english_copy_fault")
    status = (
        "HOLD_AUTOMATED_QUALITY" if exclusions else "PENDING_INDEPENDENT_BLIND_REVIEW"
    )
    return {
        "schema_version": SCHEMA,
        "status": status,
        "candidate_manifest_sha256": file_sha256(candidate / "private-manifest.json"),
        "seed_sha256": manifest["seed_sha256"],
        "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "source_sha256": {
            "reference_list": file_sha256(args.reference_list),
            "prior_audit": file_sha256(args.prior_audit),
            "protected_inventory": file_sha256(args.protected_list),
            **reference_hashes,
        },
        "rows": len(all_rows),
        "groups": len(group_rows),
        "language_rows": dict(collections.Counter(r["language"] for r in all_rows)),
        "mechanism_groups": {k: len(v) for k, v in sorted(per_mechanism.items())},
        "deep_dossier_groups": len(long_groups),
        "deep_positions": sorted(deep_positions),
        "tokens": token_summary,
        "english_copy_fault_groups": len(copy_fault_groups),
        "reference_comparison_count": len(comparisons),
        "protected_roles": len(protected_receipts),
        "flagged_comparisons": sorted(flagged),
        "within_candidate_near_pairs": within_near,
        "excluded_groups": {
            gid: sorted(reasons) for gid, reasons in sorted(exclusions.items())
        },
        "retained_groups": len(group_rows) - len(exclusions),
        "limitations": [
            "Automated checks do not establish document realism or semantic independence",
            "A different reviewer must assess all 90 blind questions before any training or release",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--reference-list", type=Path, required=True)
    parser.add_argument("--protected-list", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output must be new")
    result = audit(args)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "rows",
                    "groups",
                    "deep_dossier_groups",
                    "tokens",
                    "reference_comparison_count",
                    "flagged_comparisons",
                    "retained_groups",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
