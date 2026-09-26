"""Seal-first private aggregate for the frozen MASSIVE v5 blind pilot.

Read both blind receipts before opening the answer key. Emit aggregate-only
counts; never print source IDs, raw utterances or reviewer rationales.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path

from training.data import build_massive_v5_intent_pilot as v5
from training.data import build_pilot as pilot

EXPECTED = {
    "candidate_manifest": "a73d7a2a5d6f735af451801726bf8b47b67c352de38a83e461157632691c2a39",
    "key": "2e60566611f7a3e2488cff0ae75769a085aca5496b4250cd27e0eb134ca4e532",
    "local_packet": "abd893f5dbde658d8a8f4602439b8ed34ec19adfcf650f18afca1ece093dee85",
    "parallel_packet": "7fc9f976039d65ca58812233cdc2229dc68c0032d2f20308cf582d4fe43e3d2b",
    "stage1_manifest": "e0aefd6cb4e1ae0cbfde9c1e7f9d2943a6a2dd88afda24e65549f4e2fa4e433c",
    "stage1_rows": "1bc0159c85b28de6efc46df4c57d2913aec17da20aa94c771662f40d613ba770",
    "stage1_receipt": "1dd13ad436d4f8ec2609384ca3d0740f0f628411e92cb34577d4a8d7a08e00f3",
    "stage2_manifest": "9e967b1c7481aa9968acc50419eae23b6bde0d3aa38dc84568c23dde221f7a64",
    "stage2_locale": "5aec6d0779a934c2e80b4f8939f7677d6e0b4fabe62a6cc14cc7c55987d165a3",
    "stage2_group": "782e28ccb97b6cfce3a3c1f561cb5a1386d280c20f185b5a4d86a8d6ffc23918",
    "stage2_receipt": "9a29e46489b1d1d0380a8133a64219c446ee713b459309a0bb0f015291706f38",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(candidate: Path, stage1: Path, stage2: Path, output: Path) -> dict:
    paths = {
        "candidate_manifest": candidate / "manifest.json",
        "key": candidate / "private-key.jsonl",
        "local_packet": stage1 / "packet.jsonl",
        "parallel_packet": stage2 / "packet.jsonl",
        "stage1_manifest": stage1 / "manifest.json",
        "stage1_rows": stage1.parent.parent
        / "massive-v5-intent-r2-independent-stage1-20260927"
        / "stage1-judgments.private.jsonl",
        "stage1_receipt": stage1.parent.parent
        / "massive-v5-intent-r2-independent-stage1-20260927"
        / "stage1-receipt.private.json",
        "stage2_manifest": stage2 / "manifest.json",
        "stage2_locale": stage2.parent.parent
        / "massive-v5-intent-r2-independent-stage2-20260927"
        / "stage2-locale-judgments.private.jsonl",
        "stage2_group": stage2.parent.parent
        / "massive-v5-intent-r2-independent-stage2-20260927"
        / "stage2-group-judgments.private.jsonl",
        "stage2_receipt": stage2.parent.parent
        / "massive-v5-intent-r2-independent-stage2-20260927"
        / "stage2-receipt.private.json",
    }
    # No key or candidate rows are opened before *all* blinded evidence and
    # chronology have been checked against the precommitted SHA-256 values.
    for name, path in paths.items():
        if pilot.sha_file(path) != EXPECTED[name]:
            raise ValueError(f"Frozen {name} SHA mismatch")
    first = _json(paths["stage1_receipt"])
    second = _json(paths["stage2_receipt"])
    if (
        first.get("judgments_sha256") != EXPECTED["stage1_rows"]
        or first.get("packet_sha256") != EXPECTED["local_packet"]
        or first.get("manifest_sha256") != EXPECTED["stage1_manifest"]
        or first.get("row_count") != 84
        or second.get("stage1_judgments_sha256") != EXPECTED["stage1_rows"]
        or second.get("stage1_receipt_sha256") != EXPECTED["stage1_receipt"]
        or second.get("stage2_packet_sha256") != EXPECTED["parallel_packet"]
        or second.get("stage2_manifest_sha256") != EXPECTED["stage2_manifest"]
        or second.get("locale_judgments_sha256") != EXPECTED["stage2_locale"]
        or second.get("group_judgments_sha256") != EXPECTED["stage2_group"]
        or second.get("locale_row_count") != 72
        or second.get("group_count") != 12
    ):
        raise ValueError("Blind receipt references changed")
    first_time = dt.datetime.fromisoformat(first["review_completed_utc"])
    second_time = dt.datetime.fromisoformat(second["review_completed_utc"])
    if not (
        first_time < second_time
        and paths["stage1_receipt"].stat().st_mtime_ns
        < paths["parallel_packet"].stat().st_mtime_ns
        < paths["stage2_receipt"].stat().st_mtime_ns
    ):
        raise ValueError("Blind stage seal order changed")
    manifest = _json(paths["candidate_manifest"])
    if manifest.get("training_approved") is not False:
        raise ValueError("Candidate status changed")
    key = v5._jsonl(paths["key"])
    blind1 = v5._jsonl(paths["stage1_rows"])
    blind2 = v5._jsonl(paths["stage2_locale"])
    blind_groups = v5._jsonl(paths["stage2_group"])
    if (len(key), len(blind1), len(blind2), len(blind_groups)) != (84, 84, 72, 12):
        raise ValueError("Blind/key row inventory changed")
    key_by_id = {row["review_id"]: row for row in key}
    if len(key_by_id) != 84 or {row["review_id"] for row in blind1} != set(key_by_id):
        raise ValueError("Stage1/key join incomplete")
    group_to_intent = {}
    for row in key:
        intent = row["intent"]
        token = row["group_token"]
        if token in group_to_intent and group_to_intent[token] != intent:
            raise ValueError("One group has multiple intent keys")
        group_to_intent[token] = intent
    if len(group_to_intent) != 12:
        raise ValueError("Private group key count changed")
    exact_by_intent = collections.Counter()
    exact_by_locale = collections.Counter()
    key_agree_exact = 0
    closest_agree_nonexact = 0
    fully_exact_groups = collections.Counter()
    for row in blind1:
        private = key_by_id[row["review_id"]]
        if row["locale"] != private["locale"]:
            raise ValueError("Stage1/key locale disagreement")
        if row["exact_unique_option_fit"]:
            exact_by_intent[private["intent"]] += 1
            exact_by_locale[private["locale"]] += 1
            fully_exact_groups[private["group_token"]] += 1
            key_agree_exact += row["inferred_id"] == private["gold_option_key"]
        else:
            closest_agree_nonexact += (
                row.get("closest_id_if_not_exact") == private["gold_option_key"]
            )
    if len({row["group_token"] for row in blind_groups}) != 12:
        raise ValueError("Stage2 group token repeat")
    drift_by_intent = collections.defaultdict(collections.Counter)
    drift_by_locale = collections.defaultdict(collections.Counter)
    for row in blind2:
        private = key_by_id[row["stage1_review_id"]]
        if (
            row["group_token"] != private["group_token"]
            or row["locale"] != private["locale"]
        ):
            raise ValueError("Stage2/key row join changed")
        drift_by_intent[private["intent"]][row["semantic_drift"]] += 1
        drift_by_locale[private["locale"]][row["semantic_drift"]] += 1
    pass_tokens = {
        row["group_token"] for row in blind_groups if row["blind_pair_gate"] == "PASS"
    }
    if len(pass_tokens) != 1 or len(blind2) != 72:
        raise ValueError("Stage2 blind aggregate changed")
    result = {
        "schema_version": "massive-v5-seal-first-postkey-aggregate/1",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_approved": False,
        "decision": "HOLD_FOR_TRAINING",
        "input_sha256": EXPECTED,
        "rows": 84,
        "groups": 12,
        "stage1_exact_unique": sum(exact_by_intent.values()),
        "stage1_no_exact": 84 - sum(exact_by_intent.values()),
        "stage1_key_agree_among_exact": key_agree_exact,
        "stage1_closest_key_agree_among_nonexact_not_a_pass": closest_agree_nonexact,
        "stage1_exact_by_intent": dict(sorted(exact_by_intent.items())),
        "stage1_exact_by_locale": dict(sorted(exact_by_locale.items())),
        "stage1_all_seven_exact_groups": sum(
            value == 7 for value in fully_exact_groups.values()
        ),
        "stage2_parallel_pass_groups": len(pass_tokens),
        "stage2_parallel_drift_by_intent": {
            key: dict(sorted(value.items()))
            for key, value in sorted(drift_by_intent.items())
        },
        "stage2_parallel_drift_by_locale": {
            key: dict(sorted(value.items()))
            for key, value in sorted(drift_by_locale.items())
        },
        "combined_exact_and_parallel_pass_groups": sum(
            fully_exact_groups.get(token) == 7 for token in pass_tokens
        ),
        "reviewer_locale_qualification": "HOLD_PEND_NATIVE_BILINGUAL_REVIEW",
    }
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = output / "aggregate.private.json"
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(pilot.canonical(result) + "\n")
    return {
        key: result[key]
        for key in (
            "decision",
            "stage1_exact_unique",
            "stage1_no_exact",
            "stage1_key_agree_among_exact",
            "stage1_all_seven_exact_groups",
            "stage2_parallel_pass_groups",
            "combined_exact_and_parallel_pass_groups",
        )
    } | {"aggregate_sha256": pilot.sha_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--stage2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(pilot.canonical(audit(args.candidate, args.stage1, args.stage2, args.output)))


if __name__ == "__main__":
    main()
