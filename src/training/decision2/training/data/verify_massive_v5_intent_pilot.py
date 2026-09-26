"""Independently verify a private MASSIVE v5 pilot freeze without printing rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from training.data import build_massive_multilingual as massive
from training.data import build_massive_v5_intent_pilot as v5
from training.data import build_pilot as pilot
from training.model.data import validate_row

FILENAMES = (
    "candidate.private.jsonl",
    "blind-local.jsonl",
    "blind-parallel.jsonl",
    "private-key.jsonl",
    "LICENSE",
    "NOTICE.md",
)
LOCAL_KEYS = {"review_id", "locale", "state", "instructions", "options"}
PARALLEL_KEYS = {"group_token", "english_anchor", "localized"}
KEY_KEYS = {
    "review_id",
    "group_token",
    "source_id",
    "locale",
    "intent",
    "gold_option_key",
    "candidate_id",
}
TOKEN = re.compile(r"[0-9a-f]{32}\Z")


def verify(
    directory: Path, archive: Path, source_directory: Path, legacy_v1_train: Path
) -> dict:
    if directory.stat().st_mode & 0o777 != 0o700:
        raise ValueError("Pilot directory permissions changed")
    if (directory / "manifest.json").stat().st_mode & 0o777 != 0o600:
        raise ValueError("Manifest permissions changed")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "decision2-massive-v5-stable-intent-pilot/1"
        or manifest.get("training_approved") is not False
        or manifest.get("pilot_status") != "AWAIT_INDEPENDENT_BLIND_REVIEW"
    ):
        raise ValueError("Pilot status or schema changed")
    for filename in FILENAMES:
        path = directory / filename
        if path.stat().st_mode & 0o777 != 0o600:
            raise ValueError(f"{filename} permissions changed")
        if pilot.sha_file(path) != manifest["outputs"][filename]:
            raise ValueError(f"{filename} SHA mismatch")
    if (
        manifest["source_archive_sha256"] != massive.ARCHIVE_SHA
        or manifest["source_license_sha256"] != massive.LICENSE_SHA
        or manifest["source_notice_sha256"] != massive.NOTICE_SHA
        or manifest["source_locale_sha256"] != massive.SOURCE_SHA
        or manifest["legacy_v1_train_sha256"] != v5.LEGACY_TRAIN_SHA
    ):
        raise ValueError("Source or rights receipt changed")
    source, _ = massive.load_source(source_directory, archive)
    legacy = v5.legacy_ids(legacy_v1_train)
    rows = v5._jsonl(directory / "candidate.private.jsonl")
    local = v5._jsonl(directory / "blind-local.jsonl")
    parallel = v5._jsonl(directory / "blind-parallel.jsonl")
    key = v5._jsonl(directory / "private-key.jsonl")
    if (len(rows), len(local), len(parallel), len(key)) != (84, 84, 12, 84):
        raise ValueError("Candidate or packet count changed")
    if any(set(item) != LOCAL_KEYS for item in local):
        raise ValueError("Blind local schema leaks join or gold")
    if any(set(item) != PARALLEL_KEYS for item in parallel):
        raise ValueError("Blind parallel schema changed")
    if any(set(item) != KEY_KEYS for item in key):
        raise ValueError("Private join schema changed")
    if len({item["review_id"] for item in local}) != 84:
        raise ValueError("Blind local IDs repeat")
    if len({item["group_token"] for item in parallel}) != 12:
        raise ValueError("Blind parallel group IDs repeat")
    if any(not TOKEN.fullmatch(item["review_id"]) for item in local):
        raise ValueError("Nonopaque review row token")
    if any(not TOKEN.fullmatch(item["group_token"]) for item in parallel):
        raise ValueError("Nonopaque review group token")
    local_by_id = {item["review_id"]: item for item in local}
    key_by_candidate = {item["candidate_id"]: item for item in key}
    parallel_by_group = {item["group_token"]: item for item in parallel}
    if (
        len(key_by_candidate) != 84
        or set(local_by_id) != {item["review_id"] for item in key}
        or set(parallel_by_group) != {item["group_token"] for item in key}
    ):
        raise ValueError("Private join completeness changed")
    groups = defaultdict(list)
    for row in rows:
        validate_row(row, "train")
        private = key_by_candidate[row["id"]]
        visible = local_by_id[private["review_id"]]
        metadata = row["audit_metadata"]
        source_id, locale = metadata["source_id"], metadata["source_locale"]
        if (
            source_id in legacy
            or metadata["source_partition"] != "train"
            or row["family"] != "massive_v5_stable_intent_choice"
            or row["task_type"] != "choice"
            or row["options"]
            != [
                {"key": intent, "description": v5.INTENTS[intent]}
                for intent in v5.SOURCE_INTENT_ORDER
            ]
            or row["label"] != v5.SOURCE_INTENT_ORDER.index(metadata["intent"])
            or source[locale][source_id]["partition"] != "train"
            or source[locale][source_id]["utt"] != row["state"]
            or source[locale][source_id]["intent"] != metadata["intent"]
            or private["source_id"] != source_id
            or private["locale"] != locale
            or private["intent"] != metadata["intent"]
            or private["gold_option_key"] != row["options"][row["label"]]["key"]
            or visible["locale"] != locale
            or visible["state"] != row["state"]
            or visible["instructions"] != row["instructions"]
            or visible["options"] != row["options"]
        ):
            raise ValueError("Candidate/source/packet lineage changed")
        groups[source_id].append((row, private))
    if len(groups) != 12:
        raise ValueError("Source group count changed")
    for source_id, group in groups.items():
        if {item[0]["audit_metadata"]["source_locale"] for item in group} != set(
            massive.LOCALES
        ):
            raise ValueError("Seven-locale lineage incomplete")
        tokens = {item[1]["group_token"] for item in group}
        if len(tokens) != 1:
            raise ValueError("One source ID spans multiple blind groups")
        token = next(iter(tokens))
        packet = parallel_by_group[token]
        expected = {source[locale][source_id]["utt"] for locale in massive.LOCALES[1:]}
        if (
            packet["english_anchor"] != source["en-US"][source_id]["utt"]
            or {item["utterance"] for item in packet["localized"]} != expected
            or {item["locale"] for item in packet["localized"]}
            != set(massive.LOCALES[1:])
        ):
            raise ValueError("Parallel packet does not match source")
    intent_counts = Counter(
        source["en-US"][identifier]["intent"] for identifier in groups
    )
    if intent_counts != Counter(dict.fromkeys(v5.SOURCE_INTENT_ORDER, 3)):
        raise ValueError("Intent source-group balance changed")
    if manifest["intent_group_counts"] != dict(sorted(intent_counts.items())):
        raise ValueError("Manifest intent counts changed")
    return {
        "status": manifest["pilot_status"],
        "training_approved": False,
        "source_groups": len(groups),
        "rows": len(rows),
        "locale_counts": dict(
            sorted(Counter(item["locale"] for item in local).items())
        ),
        "intent_group_counts": dict(sorted(intent_counts.items())),
        "max_utterance_characters": max(len(row["state"]) for row in rows),
        "manifest_sha256": pilot.sha_file(directory / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--legacy-v1-train", required=True, type=Path)
    args = parser.parse_args()
    print(
        pilot.canonical(
            verify(
                args.directory,
                args.archive,
                args.source_directory,
                args.legacy_v1_train,
            )
        )
    )


if __name__ == "__main__":
    main()
