"""Prepare, but do not admit, the matched 0.6B Score data contrast D.

The input is the previously mechanically audited v6 TRAIN-only candidate.
This script samples complete independent source groups, checks their labels
with a second oracle, replaces whole original Score groups, and emits a
gold-free blind-review packet. The output stays HOLD until independent review.
No protected answer, model output, or benchmark scoring code is read.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import hmac
import json
import stat
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_score_curriculum_v6 as v6
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

SCHEMA = "decision2-score-replacement-d06/1"
SEED = "decision2-score-replacement-d06-20260928"
PARENT_SHA = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
PARENT_MANIFEST_SHA = "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8"
SELECT_SHA = "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"
CAL_SHA = "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a"
V6_SHA = "6aee966cc5499a87d2a77241676586c9c9f801b3c662a078daf025f001169f54"
V6_MANIFEST_SHA = "1217eb9a9003615562715ba11f9164c0e3cf09277433169d8b92ef3d741fe736"
PARENT_TOKENS = 4_094_489
REPLACEMENT_GROUPS = 128
REPLACEMENT_ROWS = 384
LANGUAGE_GROUPS = {"en": 24, "zh": 8}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_check(path: Path, expected: str, role: str) -> None:
    if pilot.sha_file(path) != expected:
        raise ValueError(f"Frozen {role} bytes changed")


def _independent_oracle(family: str, state: dict[str, Any]) -> int:
    """Resolve displayed facts without importing the authoring oracle."""
    if family == "score_obligation_review":
        active: dict[str, tuple[int, str]] = {}
        for event in state["events"]:
            if event["scope"] == "core":
                key = event["control"]
                candidate = (int(event["timestamp"]), event["assessment"])
                if key not in active or candidate[0] > active[key][0]:
                    active[key] = candidate
        if len(active) != 2:
            raise ValueError("Missing two core controls")
        labels = {item[1] for item in active.values()}
        if "rejected" in labels:
            return 0
        return 1 if "unresolved" in labels else 2
    if family == "score_evidence_intersection":
        documents = state["documents"]
        if len(documents) != 2:
            raise ValueError("Expected two independent documents")
        claims = []
        for document in documents:
            fmt = document["format"]
            if fmt == "checklist":
                values = document["checked"]
            elif fmt == "table":
                values = [
                    r["claim"] for r in document["rows"] if r["status"] == "active"
                ]
            elif fmt == "memo":
                values = document["current_attestations"].split("; ")
            elif fmt == "ticket":
                values = [document["line_one"], document["line_two"]]
            else:
                raise ValueError("Unrecognized evidence format")
            if len(values) != 2 or len(set(values)) != 2:
                raise ValueError("Source must independently attest two claims")
            claims.append(set(values))
        if not all(values <= set(state["eligible_claims"]) for values in claims):
            raise ValueError("Unrecognized claim")
        return len(claims[0] & claims[1])
    if family == "score_route_depth":
        adjacency: dict[str, list[str]] = collections.defaultdict(list)
        for edge in state["links"]:
            adjacency[edge["from"]].append(edge["to"])
        queue = collections.deque([(state["start"], 0)])
        seen = {state["start"]}
        while queue:
            node, distance = queue.popleft()
            if node == state["finish"]:
                return 2 if distance <= 2 else 1
            for next_node in adjacency[node]:
                if next_node not in seen:
                    seen.add(next_node)
                    queue.append((next_node, distance + 1))
        return 0
    if family == "score_timely_streak":
        by_day = {
            int(record["day"]): bool(record["on_time"]) for record in state["days"]
        }
        if sorted(by_day) != list(range(14)):
            raise ValueError("Expected exactly days 0..13")
        longest = current = 0
        for day in range(1, 13):
            current = current + 1 if by_day[day] else 0
            longest = max(longest, current)
        return 0 if longest <= 2 else (1 if longest == 3 else 2)
    raise ValueError(f"Unknown Score mechanism {family}")


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return dict(groups)


def _sample_v6(candidate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(candidate) != 969 or len(_groups(candidate)) != 323:
        raise ValueError("Mechanically screened v6 corpus changed")
    by_stratum: dict[tuple[str, str], list[tuple[str, list[dict[str, Any]]]]] = (
        collections.defaultdict(list)
    )
    for group_id, triplet in _groups(candidate).items():
        if len(triplet) != 3 or {r["label"] for r in triplet} != {0, 1, 2}:
            raise ValueError("Incomplete v6 source group")
        if (
            len({r["language"] for r in triplet}) != 1
            or len({r["family"] for r in triplet}) != 1
        ):
            raise ValueError("Source group mixes languages or mechanisms")
        by_stratum[(triplet[0]["family"], triplet[0]["language"])].append(
            (group_id, triplet)
        )
    chosen = []
    for family in (f"score_{name}" for name in v6.FAMILIES):
        for language, count in LANGUAGE_GROUPS.items():
            pool = by_stratum[(family, language)]
            if len(pool) < count:
                raise ValueError("Insufficient independent groups in stratum")
            pool.sort(key=lambda item: (_hash(f"{SEED}\0group\0{item[0]}"), item[0]))
            chosen.extend(pool[:count])
    if len(chosen) != REPLACEMENT_GROUPS:
        raise AssertionError("D roster cardinality changed")
    rows = []
    for group_id, triplet in chosen:
        ordered = sorted(triplet, key=lambda r: r["label"])
        if len({pilot.canonical(row["state"]) for row in ordered}) != 3:
            raise ValueError("Triplet did not vary the decisive state")
        if len({pilot.canonical(row["instructions"]) for row in ordered}) != 1:
            raise ValueError("Triplet changes its decision rule")
        if len({pilot.canonical(row["options"]) for row in ordered}) != 1:
            raise ValueError("Triplet changes its Score levels")
        for row in ordered:
            if row["source"] != v6.SOURCE or row["task_type"] != "score":
                raise ValueError("Candidate source or type changed")
            if row["group_id"] != group_id or row["input_sha256"] != pilot.input_sha256(
                row
            ):
                raise ValueError("Candidate lineage or input hash changed")
            if _independent_oracle(row["family"], row["state"]) != row["label"]:
                raise ValueError("Independent oracle disagrees with candidate label")
            rows.append(row)
    if len(rows) != REPLACEMENT_ROWS:
        raise AssertionError("D roster row cardinality changed")
    return rows


def _matched_score_groups(
    original_score: list[dict[str, Any]],
    lengths: dict[str, int],
    target: int,
    replace_rows: int,
) -> tuple[set[str], dict[str, int]]:
    """Match tokens while replacing only complete original Score groups."""
    groups = _groups(original_score)
    by_size: dict[int, list[tuple[str, int]]] = {1: [], 2: []}
    for group_id, rows in groups.items():
        if len(rows) not in by_size:
            raise ValueError("Unexpected original Score group size")
        by_size[len(rows)].append((group_id, sum(lengths[row["id"]] for row in rows)))
    all_rows = len(original_score)
    proposed_pairs = round(len(by_size[2]) * replace_rows / all_rows)
    feasible_pairs = [
        count
        for count in range(len(by_size[2]) + 1)
        if 0 <= replace_rows - 2 * count <= len(by_size[1])
    ]
    if not feasible_pairs:
        raise ValueError("Cannot replace complete Score groups")
    pair_count = min(feasible_pairs, key=lambda n: (abs(n - proposed_pairs), n))
    counts = {1: replace_rows - 2 * pair_count, 2: pair_count}
    target_per_row = target / replace_rows
    selected: dict[int, list[tuple[str, int]]] = {}
    unselected: dict[int, list[tuple[str, int]]] = {}
    for size, items in by_size.items():
        ordered = sorted(
            items,
            key=lambda item: (
                abs(item[1] / size - target_per_row),
                _hash(f"{SEED}\0remove\0{item[0]}"),
            ),
        )
        selected[size], unselected[size] = (
            ordered[: counts[size]],
            ordered[counts[size] :],
        )
    removed_tokens = sum(value for size in (1, 2) for _, value in selected[size])
    for _ in range(128):
        best: tuple[int, int, int, int, int] | None = None
        old_error = abs(removed_tokens - target)
        for size in (1, 2):
            available = sorted(unselected[size], key=lambda item: (item[1], item[0]))
            values = [item[1] for item in available]
            for old_index, (_, old_tokens) in enumerate(selected[size]):
                desired = target - removed_tokens + old_tokens
                index = bisect.bisect_left(values, desired)
                for new_index in (index - 1, index):
                    if not 0 <= new_index < len(available):
                        continue
                    new_tokens = removed_tokens - old_tokens + values[new_index]
                    error = abs(new_tokens - target)
                    proposal = (error, size, old_index, new_index, new_tokens)
                    if error < old_error and (best is None or proposal < best):
                        best = proposal
            if best is not None and best[1] == size:
                unselected[size] = available
        if best is None:
            break
        _, size, old_index, new_index, removed_tokens = best
        selected[size][old_index], unselected[size][new_index] = (
            unselected[size][new_index],
            selected[size][old_index],
        )
        if removed_tokens == target:
            break
    chosen = {group_id for size in (1, 2) for group_id, _ in selected[size]}
    if sum(len(groups[group_id]) for group_id in chosen) != replace_rows:
        raise AssertionError("Original Score group was split")
    return chosen, {
        "removed_rows": replace_rows,
        "removed_groups": len(chosen),
        "removed_single_groups": counts[1],
        "removed_pair_groups": counts[2],
        "removed_tokens": removed_tokens,
        "target_tokens": target,
        "token_abs_delta": abs(removed_tokens - target),
    }


def _blind_packet(rows: list[dict[str, Any]], secret: bytes) -> tuple[bytes, bytes]:
    packet, key = [], []
    for row in rows:
        alias = hmac.new(
            secret, f"{SEED}\0{row['id']}".encode(), hashlib.sha256
        ).hexdigest()[:24]
        packet.append(
            {
                "review_id": alias,
                "state": row["state"],
                "instructions": row["instructions"],
                "options": row["options"],
            }
        )
        key.append(
            {
                "review_id": alias,
                "row_id": row["id"],
                "group_id": row["group_id"],
                "family": row["family"],
                "language": row["language"],
                "label": row["label"],
            }
        )
    packet.sort(key=lambda row: _hash(f"{SEED}\0packet\0{row['review_id']}"))
    if len({row["review_id"] for row in packet}) != len(packet):
        raise ValueError("Blind alias collision")
    return pilot.jsonl_bytes(packet), pilot.jsonl_bytes(key)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path, digest, role in (
        (args.parent_train, PARENT_SHA, "TRAIN"),
        (args.parent_manifest, PARENT_MANIFEST_SHA, "rights manifest"),
        (args.select, SELECT_SHA, "SELECT"),
        (args.cal, CAL_SHA, "CAL"),
        (args.candidate_v6, V6_SHA, "v6 candidate"),
        (args.v6_manifest, V6_MANIFEST_SHA, "v6 QA manifest"),
    ):
        _file_check(path, digest, role)
    if stat.S_IMODE(args.blind_secret.stat().st_mode) != 0o600:
        raise PermissionError("Blind packet secret must have mode 0600")
    secret = args.blind_secret.read_bytes()
    if len(secret) != 32:
        raise ValueError("Blind packet secret must be exactly 32 bytes")
    rights = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    if not isinstance(rights.get("source_rights"), dict):
        raise ValueError("Parent source-rights ledger is absent")
    v6_manifest = json.loads(args.v6_manifest.read_text(encoding="utf-8"))
    if (
        v6_manifest.get("status") != "research_candidate_not_training_approved"
        or v6_manifest.get("outputs", {})
        .get("score_curriculum.train.jsonl", {})
        .get("sha256")
        != V6_SHA
        or v6_manifest.get("candidate_groups") != 323
        or v6_manifest.get("rights", {}).get("external_text_in_added_rows") is not False
    ):
        raise ValueError("v6 source QA or rights state changed")
    parent = load_partition(args.parent_train, "train")
    select = load_partition(args.select, "select")
    cal = load_partition(args.cal, "cal")
    v6_rows = load_partition(args.candidate_v6, "train")
    if (len(parent), len(select), len(cal)) != (7455, 700, 700):
        raise ValueError("Frozen partition count changed")
    candidate = _sample_v6(v6_rows)
    references, receipts = v6._load_protected(args.protected_inventory)
    if "typed_final_goldfree" not in references:
        raise ValueError("Typed FINAL gold-free prompt inventory is required")
    overlap = {}
    for role, reference in (
        ("parent", parent),
        ("select", select),
        ("cal", cal),
        *sorted(references.items()),
    ):
        overlap[role] = targeted.context_overlap(candidate, reference, approximate=True)
        full = pilot.near_duplicates(candidate, reference)
        if full["count"]:
            raise ValueError(f"Near-complete prompt overlap with {role}")
        overlap[role]["near_full_prompt_count"] = full["count"]
    check_partition_isolation(
        {"train": [*parent, *candidate], "select": select, "cal": cal}
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    parent_lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in parent}
    if sum(parent_lengths.values()) != PARENT_TOKENS:
        raise ValueError("Original Qwen3-0.6B native-token total changed")
    candidate_lengths = {
        row["id"]: pilot.count_tokens(row, tokenizer) for row in candidate
    }
    if max(candidate_lengths.values()) > 8192:
        raise ValueError("Candidate exceeds complete-input cap")
    original_score = [row for row in parent if row["task_type"] == "score"]
    if len(original_score) != 516:
        raise ValueError("Original Score cohort changed")
    removed_groups, removal = _matched_score_groups(
        original_score,
        parent_lengths,
        sum(candidate_lengths.values()),
        REPLACEMENT_ROWS,
    )
    replacements = iter(candidate)
    merged = [
        next(replacements) if row["group_id"] in removed_groups else row
        for row in parent
    ]
    if next(replacements, None) is not None or len(merged) != len(parent):
        raise AssertionError("Candidate did not replace exactly 384 old rows")
    if [r["id"] for r in merged if r["task_type"] != "score"] != [
        r["id"] for r in parent if r["task_type"] != "score"
    ]:
        raise AssertionError("Choice or Noul identity/order changed")
    check_partition_isolation({"train": merged, "select": select, "cal": cal})
    merged_tokens = (
        PARENT_TOKENS - removal["removed_tokens"] + sum(candidate_lengths.values())
    )
    if abs(merged_tokens - PARENT_TOKENS) * 200 > PARENT_TOKENS:
        raise ValueError("D exceeds the prospective +/-0.5% native-token budget")
    packet, key = _blind_packet(candidate, secret)
    outputs = {
        "candidate-score384.jsonl": pilot.jsonl_bytes(candidate),
        "train-d.jsonl": pilot.jsonl_bytes(merged),
        "blind-review.jsonl": packet,
        "sealed-review-key.jsonl": key,
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for name, content in outputs.items():
        pilot._atomic_write(args.output_dir / name, content)
        (args.output_dir / name).chmod(0o600)
    summary = {
        "schema": SCHEMA,
        "status": "HOLD_PENDING_INDEPENDENT_BLIND_REVIEW",
        "training_approved": False,
        "publication_eligible": False,
        "seed_sha256": _hash(SEED),
        "source": "previously mechanically audited internal v6 TRAIN-only candidate",
        "input_sha256": {
            "parent_train": PARENT_SHA,
            "parent_manifest": PARENT_MANIFEST_SHA,
            "select": SELECT_SHA,
            "cal": CAL_SHA,
            "candidate_v6": V6_SHA,
            "v6_manifest": V6_MANIFEST_SHA,
            "protected_inventory": pilot.sha_file(args.protected_inventory),
        },
        "protected_sources": receipts,
        "overlap_counts": {
            role: {key: value for key, value in audit.items() if key != "near_context"}
            | {"near_context_count": audit["near_context"]["count"]}
            for role, audit in overlap.items()
        },
        "replacement": removal,
        "candidate_groups": REPLACEMENT_GROUPS,
        "candidate_rows": REPLACEMENT_ROWS,
        "family_groups": dict(
            sorted(
                collections.Counter(
                    row["family"] for row in candidate if row["label"] == 0
                ).items()
            )
        ),
        "language_groups": dict(
            sorted(
                collections.Counter(
                    row["language"] for row in candidate if row["label"] == 0
                ).items()
            )
        ),
        "label_positions": dict(
            sorted(collections.Counter(str(row["label"]) for row in candidate).items())
        ),
        "tokenizer_revision": args.tokenizer_revision,
        "parent_tokens": PARENT_TOKENS,
        "candidate_tokens": sum(candidate_lengths.values()),
        "merged_tokens": merged_tokens,
        "relative_token_change": (merged_tokens - PARENT_TOKENS) / PARENT_TOKENS,
        "max_candidate_tokens": max(candidate_lengths.values()),
        "rights": {
            "added_rows": "internally authored synthetic v6, private TRAIN only",
            "parent_source_rights_preserved": True,
            "external_text_copied_into_added_rows": False,
        },
        "outputs": {
            name: {"sha256": pilot.sha_bytes(data), "rows": len(data.splitlines())}
            for name, data in outputs.items()
        },
        "limitations": [
            "v6's independent gold-blind editorial review is still pending; subset reuse does not clear that HOLD.",
            "Mechanically verifiable synthetic tasks do not establish transfer to independent real decisions.",
            "Approximate text overlap cannot exclude every semantic paraphrase.",
            "The original Score replacement is length-selected and may shift its task-family composition.",
        ],
    }
    pilot._atomic_write(
        args.output_dir / "manifest.json",
        (
            json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        ).encode(),
    )
    (args.output_dir / "manifest.json").chmod(0o600)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-train", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--candidate-v6", type=Path, required=True)
    parser.add_argument("--v6-manifest", type=Path, required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--blind-secret", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_groups": result["candidate_groups"],
                "candidate_rows": result["candidate_rows"],
                "relative_token_change": result["relative_token_change"],
                "train_sha256": result["outputs"]["train-d.jsonl"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
