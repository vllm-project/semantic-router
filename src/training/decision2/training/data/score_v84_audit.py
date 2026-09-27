"""Gold-free, source-aware quality audit of one frozen Score v8.4 pilot.

All paths are caller-supplied private inputs. This auditor never opens
protected answer files; the protected inventory is restricted to prompts.
Its bounded textual matching cannot certify semantic independence.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from training.data import score_v84_pilot as pilot
from training.model.data import (
    check_partition_isolation,
    digest,
    file_sha256,
    load_partition,
)

SCHEMA = "decision2-score-v8.4-pilot-audit/1"
ROOT_REFERENCE_SHA = {
    "parent_train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "parent_select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "parent_cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    "protected_inventory": "99f271a8a681cccc66c0e68e77228691cb6f9628ed2cf93ce4c414c93baa842c",
}
ID_PATTERN = re.compile(r"\b(?:CASE|ITEM|TASK|IN|OUT|DOC)-[A-Z]+\d+\b", re.I)
NUMBER_PATTERN = re.compile(r"\d+(?::\d+)?")
SPACE_PATTERN = re.compile(r"\s+")
PROTECTED_ROLES = {
    "typed_dev",
    "css_pilot",
    "typed_final_goldfree",
    "css15_goldfree",
    "jevbench_public231",
}


def _clean(text: Any) -> str:
    if not isinstance(text, str):
        text = json.dumps(
            text, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return SPACE_PATTERN.sub(
        " ", NUMBER_PATTERN.sub("n", ID_PATTERN.sub("id", text.lower()))
    ).strip()


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    # Unicode \\w covers Chinese. For Chinese, a character n-gram also catches
    # copying that a whitespace-token index would miss.
    tokens = re.findall(r"[a-z]+|[\u4e00-\u9fff]", text, re.UNICODE)
    return {tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


def _load_blind(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        group = json.loads(line)
        if set(group) != {"review_group", "items"}:
            raise ValueError(f"{path}: blind packet has unexpected fields")
        for item in group["items"]:
            if set(item) != {"review_id", "state", "instructions", "options"}:
                raise ValueError(f"{path}: blind item has unexpected fields")
            fields = {
                "state": item["state"],
                "instructions": item["instructions"],
                "options": item["options"],
                "task_type": "score",
            }
            rows.append(
                {
                    "id": item["review_id"],
                    "group_id": group["review_group"],
                    **fields,
                    "input_sha256": digest(fields),
                }
            )
    return rows


def _load_protected(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if file_sha256(path) != ROOT_REFERENCE_SHA["protected_inventory"]:
        raise ValueError("Frozen protected inventory hash changed")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inventory, list):
        raise ValueError("Protected inventory is not a list")
    roles = {entry["role"] for entry in inventory}
    if not roles.issuperset(PROTECTED_ROLES) or len(roles) != len(inventory):
        raise ValueError("Protected prompt roles incomplete or duplicated")
    references = {}
    receipts = []
    for item in inventory:
        file = Path(item["path"])
        if not file.name.endswith(("prompts.jsonl", "packet.jsonl")):
            raise ValueError("Protected source must be a gold-free prompt file")
        sha = file_sha256(file)
        if sha != item["sha256"]:
            raise ValueError(f"Protected role {item['role']} changed")
        prompts = []
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if set(row) != {"id", "state", "questions"}:
                    raise ValueError(
                        f"Protected role {item['role']} has non-prompt fields"
                    )
                prompts.append(
                    {
                        "id": row["id"],
                        "state": row["state"],
                        "group_id": None,
                        "instructions": row["questions"],
                        "options": [],
                        "task_type": "context",
                        "input_sha256": None,
                    }
                )
        references[item["role"]] = prompts
        receipts.append({"role": item["role"], "sha256": sha, "rows": len(prompts)})
    return references, sorted(receipts, key=lambda r: r["role"])


def _references(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(
        {entry["name"] for entry in entries}
    ) != len(entries):
        raise ValueError("Reference list invalid or duplicate")
    result = {}
    hashes = {}
    for entry in entries:
        name, kind, file = entry["name"], entry["kind"], Path(entry["path"])
        if kind not in {"train", "select", "cal", "blind"}:
            raise ValueError(f"{name}: unknown reference kind")
        hashes[name] = file_sha256(file)
        if name in ROOT_REFERENCE_SHA and hashes[name] != ROOT_REFERENCE_SHA[name]:
            raise ValueError(f"{name}: frozen reference SHA differs")
        result[name] = (
            _load_blind(file) if kind == "blind" else load_partition(file, kind)
        )
    if not {"parent_train", "parent_select", "parent_cal"}.issubset(result):
        raise ValueError("Parent TRAIN/SELECT/CAL absent")
    return result, hashes


def _candidate(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    if len(rows) != pilot.GROUPS[role] * len(pilot.MECHANISMS) * 3:
        raise ValueError(f"{role}: cardinality changed")
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    if len(by_group) != pilot.GROUPS[role] * len(pilot.MECHANISMS):
        raise ValueError(f"{role}: incomplete case groups")
    mechanisms = collections.Counter(row["audit_metadata"]["mechanism"] for row in rows)
    if mechanisms != dict.fromkeys(pilot.MECHANISMS, pilot.GROUPS[role] * 3):
        raise ValueError(f"{role}: mechanisms not balanced")
    expected_zh = 6  # one Chinese group per mechanism and role, three rows each
    actual_zh = sum(row["language"] == "zh" for row in rows)
    if actual_zh != expected_zh * 3:
        raise ValueError(f"{role}: Chinese case quota changed")
    for group, triplet in by_group.items():
        if len(triplet) != 3 or {r["label"] for r in triplet} != {0, 1, 2}:
            raise ValueError(f"Incomplete ordinal triplet: {group}")
        if len({r["audit_metadata"]["case"] for r in triplet}) != 1:
            raise ValueError(f"Triplet source case changed: {group}")
        for row in triplet:
            meta = row["audit_metadata"]
            if meta["target"] == meta["near"] or row["source"] != pilot.VERSION:
                raise ValueError(f"Target/near or source invalid: {group}")
            if (
                row["state"].count(meta["target"]) < 2
                or row["state"].count(meta["near"]) < 1
            ):
                raise ValueError(f"Source identity not explicit: {group}")
            if (
                pilot.rendered_oracle(row["state"], meta["mechanism"], meta["case"])
                != row["label"]
            ):
                raise ValueError(f"Rendered oracle disagrees: {group}")
    return {
        "rows": len(rows),
        "groups": len(by_group),
        "mechanisms": dict(sorted(mechanisms.items())),
        "language_rows": dict(
            sorted(collections.Counter(r["language"] for r in rows).items())
        ),
        "levels": dict(
            sorted(collections.Counter(str(r["label"]) for r in rows).items())
        ),
        "unique_template_families": len({r["render_template"] for r in rows}),
    }


def _cross_group_near(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = {r["group_id"]: r for r in rows if r["label"] == 0}
    results = []
    keys = sorted(first)
    for i, a in enumerate(keys):
        ta = _clean(first[a]["state"])
        ga = _ngrams(ta)
        for b in keys[i + 1 :]:
            tb = _clean(first[b]["state"])
            gb = _ngrams(tb)
            jac = len(ga & gb) / len(ga | gb) if ga or gb else 1.0
            # Sequence matching is bounded by character length before its
            # quadratic dynamic program is invoked.
            length = min(len(ta), len(tb)) / max(len(ta), len(tb))
            seq = (
                difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
                if length > 0.70 and jac > 0.20
                else 0.0
            )
            if jac >= 0.60 or seq >= 0.88:
                results.append(
                    {
                        "group_a": a,
                        "group_b": b,
                        "fivegram_jaccard": round(jac, 5),
                        "sequence": round(seq, 5),
                    }
                )
    return results


def _one_field_decoders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag a single visible target field that uniquely identifies all levels."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    patterns = {
        "workflow": r"(?:Reconciled status|当前核对) \[[^]]+\]\s*\([^)]*\):\s*(TASK-[A-Z]+\d+)=(\w+);\s*(TASK-[A-Z]+\d+)=(\w+)",
        "connection": r"(?:Arrival window|到站区间) \[[^]]+\] IN-[A-Z]+\d+:\s*(\d\d:\d\d) — (\d\d:\d\d)",
        "stock": r"(?:Stock|库存) \[[^]]+\] ITEM-[A-Z]+\d+:\s*(\d+);\s*(?:reserved|已预约) (\d+)",
        "policy": r"(?:License|许可证) \[[^]]+\] ITEM-[A-Z]+\d+:\s*([^;]+);\s*(?:inspection|现场检查) (\w+)",
        "evidence": r"(?:Stability|稳定性) \[[^]]+\] ITEM-[A-Z]+\d+:\s*([^;]+);",
        "service_level": r"(?:Service date|交付日) \[[^]]+\] ITEM-[A-Z]+\d+:\s*(\d{4}-\d\d-\d\d);\s*(?:prerequisite|前置条件) (\w+)",
    }
    flagged = []
    for group, items in groups.items():
        mechanism = items[0]["audit_metadata"]["mechanism"]
        parsed = []
        for row in sorted(items, key=lambda item: item["label"]):
            state = row["state"].translate(str.maketrans("：；，。（）", ":;,.()"))
            pattern = patterns[mechanism].replace(
                r"ITEM-[A-Z]+\d+", re.escape(row["audit_metadata"]["target"])
            )
            match = re.search(pattern, state)
            if match is None:
                raise ValueError(f"Cannot inspect target fields: {group}")
            values = match.groups()
            parsed.append(values[1::2] if mechanism == "workflow" else values)
        for column in range(len(parsed[0])):
            if len({values[column] for values in parsed}) == 3:
                flagged.append({"group": group, "field_index": column})
    return flagged


def _reference_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    ids = {r["id"] for r in right}
    groups = {r.get("group_id") for r in right if r.get("group_id")}
    inputs = {r.get("input_sha256") for r in right if r.get("input_sha256")}
    state_sha = {hashlib.sha256(_clean(r["state"]).encode()).hexdigest() for r in right}
    exact = {
        r["group_id"]
        for r in left
        if r["id"] in ids
        or r["group_id"] in groups
        or r["input_sha256"] in inputs
        or hashlib.sha256(_clean(r["state"]).encode()).hexdigest() in state_sha
    }

    # Bounded near scan: only source rows with enough shared normalized
    # 5-grams can reach the fixed 0.60 threshold. Keep a separate loose flag
    # for suspicious 0.35-0.60 cases that require semantic adjudication.
    right_shingles = [_ngrams(_clean(r["state"])) for r in right]
    index: dict[tuple[str, ...], set[int]] = collections.defaultdict(set)
    for j, grams in enumerate(right_shingles):
        for gram in grams:
            index[gram].add(j)
    near_groups, suspicious_groups = set(), set()
    max_jaccard = 0.0
    for row in left:
        grams = _ngrams(_clean(row["state"]))
        candidate_indices: set[int] = set()
        for gram in grams:
            candidate_indices.update(index.get(gram, ()))
        for j in candidate_indices:
            other = right_shingles[j]
            jac = len(grams & other) / len(grams | other) if grams or other else 1.0
            max_jaccard = max(jac, max_jaccard)
            if jac >= 0.60:
                near_groups.add(row["group_id"])
            elif jac >= 0.35:
                suspicious_groups.add(row["group_id"])
    return {
        "exact_groups": len(exact),
        "near_groups": len(near_groups),
        "suspicious_groups": len(suspicious_groups - near_groups),
        "max_fivegram_jaccard": round(max_jaccard, 5),
        "flagged_groups": sorted(exact | near_groups | suspicious_groups),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer  # runtime only, no GPU requirement

    manifest_path = args.candidate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != pilot.VERSION:
        raise ValueError("Candidate manifest version changed")
    for role in pilot.GROUPS:
        for kind, name in (
            ("rows", f"{role}.jsonl"),
            ("packet", f"{role}-blind-packet.jsonl"),
            ("key", f"{role}-sealed-key.json"),
        ):
            if manifest["roles"][role][f"{kind}_sha256"] != file_sha256(
                args.candidate_dir / name
            ):
                raise ValueError(f"Candidate {role} {kind} bytes changed")
    train = load_partition(args.candidate_dir / "train.jsonl", "train")
    select = load_partition(args.candidate_dir / "select.jsonl", "select")
    references, ref_hashes = _references(args.reference_list)
    protected, protected_receipts = _load_protected(args.protected_list)
    check_partition_isolation(
        {
            "train": [*train, *references["parent_train"]],
            "select": [*select, *references["parent_select"]],
            "cal": references["parent_cal"],
        }
    )
    if {r["audit_metadata"]["case"] for r in train} & {
        r["audit_metadata"]["case"] for r in select
    }:
        raise ValueError("Source cases cross TRAIN and SELECT")
    candidate = {
        "train": _candidate(train, "train"),
        "select": _candidate(select, "select"),
    }
    by_mech = collections.defaultdict(set)
    for row in [*train, *select]:
        by_mech[row["audit_metadata"]["mechanism"]].add(row["render_template"])
    if any(len(styles) < 3 for styles in by_mech.values()):
        raise ValueError("A mechanism has fewer than three genres")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    toks = {
        row["id"]: len(
            tokenizer(
                row["state"]
                + row["instructions"]
                + json.dumps(row["options"], ensure_ascii=False),
                add_special_tokens=False,
            ).input_ids
        )
        for row in [*train, *select]
    }
    longer700 = sum(n > 700 for n in toks.values())
    longer1500 = sum(n > 1500 for n in toks.values())
    length_hold = longer700 < 24 or longer1500 < 6
    within_near = _cross_group_near([*train, *select])
    field_decoders = _one_field_decoders([*train, *select])
    comparisons = {}
    for role, rows in (("train", train), ("select", select)):
        for name, reference in {
            "candidate_other_role": select if role == "train" else train,
            **references,
            **protected,
        }.items():
            if role == "select" and name == "candidate_other_role":
                continue
            comparisons[f"{role}_vs_{name}"] = _reference_overlap(rows, reference)
    flagged = {
        name: receipt
        for name, receipt in comparisons.items()
        if receipt["flagged_groups"]
    }
    exclusions = collections.defaultdict(set)
    for name, receipt in flagged.items():
        for group in receipt["flagged_groups"]:
            exclusions[group].add(name)
    for near in within_near:
        exclusions[near["group_a"]].add("cross_group_near")
        exclusions[near["group_b"]].add("cross_group_near")
    for item in field_decoders:
        exclusions[item["group"]].add("single_field_decoder")
    retained = {r["group_id"] for r in [*train, *select]} - set(exclusions)
    by_mechanism = collections.Counter(
        next(
            r["audit_metadata"]["mechanism"]
            for r in [*train, *select]
            if r["group_id"] == group
        )
        for group in retained
    )
    structural_hold = len(retained) < 27 or any(
        by_mechanism[m] < 4 for m in pilot.MECHANISMS
    )
    protected_flag = any(name.split("_vs_", 1)[1] in protected for name in flagged)
    status = (
        "HOLD_PROTECTED_OVERLAP"
        if protected_flag
        else (
            "HOLD_AUTOMATED_QUALITY"
            if (
                length_hold
                or structural_hold
                or flagged
                or within_near
                or field_decoders
            )
            else "PENDING_INDEPENDENT_BLIND_REVIEW"
        )
    )
    return {
        "schema_version": SCHEMA,
        "status": status,
        "candidate_manifest_sha256": file_sha256(manifest_path),
        "source_sha256": {
            "reference_list": file_sha256(args.reference_list),
            "protected_inventory": ROOT_REFERENCE_SHA["protected_inventory"],
            **ref_hashes,
            "tokenizer_json": file_sha256(args.tokenizer_dir / "tokenizer.json"),
            "tokenizer_config": file_sha256(
                args.tokenizer_dir / "tokenizer_config.json"
            ),
        },
        "tokenizer_revision": args.tokenizer_revision,
        "candidate": candidate,
        "tokens": {
            "min": min(toks.values()),
            "max": max(toks.values()),
            "over_700": longer700,
            "over_1500": longer1500,
            "by_language": {
                lang: [
                    min(
                        toks[r["id"]]
                        for r in [*train, *select]
                        if r["language"] == lang
                    ),
                    max(
                        toks[r["id"]]
                        for r in [*train, *select]
                        if r["language"] == lang
                    ),
                ]
                for lang in ("en", "zh")
            },
        },
        "protected_roles": protected_receipts,
        "comparison_count": len(comparisons),
        "overlap": comparisons,
        "flagged_comparisons": sorted(flagged),
        "within_candidate_near_pairs": within_near,
        "single_field_decoders": field_decoders,
        "excluded_groups": {
            group: sorted(reasons) for group, reasons in sorted(exclusions.items())
        },
        "retained_groups": len(retained),
        "retained_mechanism_groups": dict(sorted(by_mechanism.items())),
        "limitations": [
            "Programmatic oracle agreement is not independent editorial review",
            "Bounded exact/near matching cannot prove semantic independence",
            "Chinese cases require an independent Chinese-language reviewer",
            "This 90-row pilot authorizes no training or release claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--reference-list", required=True, type=Path)
    parser.add_argument("--protected-list", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit receipt must be new")
    report = audit(args)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "candidate",
                    "tokens",
                    "comparison_count",
                    "flagged_comparisons",
                    "retained_groups",
                    "retained_mechanism_groups",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
