"""Freeze a TRAIN-only MASSIVE stable-intent Choice pilot for blind review.

This builder does not approve the data for training. It never uses MASSIVE
TEST or JevArena gold and it preserves whole seven-locale source groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.data import build_massive_multilingual as massive
from training.data import build_pilot as pilot
from training.data import build_score_curriculum_v6 as score_v6
from training.data import build_targeted_candidate as targeted
from training.model.data import validate_row

SEED = "decision2-massive-v5-stable-intent-choice-v1"
INTENTS = {
    "alarm_set": "Set an alarm",
    "weather_query": "Ask about weather",
    "play_music": "Play music",
    "general_joke": "Ask for a joke",
}
SOURCE_INTENT_ORDER = tuple(sorted(INTENTS))
GROUPS_PER_INTENT = 3
LEGACY_TRAIN_SHA = "4c57dac9d5dd39cf3e0920cda7c1e975f5e2dacf46379e7232797efb3ae2d9ba"
REFERENCE_SHA = {
    "base": massive.REFERENCE_FILES["clean_train"][1],
    "select": massive.REFERENCE_FILES["clean_select"][1],
    "cal": massive.REFERENCE_FILES["clean_cal"][1],
}
INSTRUCTION = (
    "Choose the one official intent ID that exactly describes the speaker's "
    "request. Answer with the ID only."
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank line at {path.name}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object row at {path.name}:{number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        for row in rows:
            stream.write(pilot.canonical(row) + "\n")


def _rank(*parts: str) -> str:
    return hashlib.sha256("\0".join((SEED, *parts)).encode()).hexdigest()


def legacy_ids(path: Path) -> set[str]:
    if pilot.sha_file(path) != LEGACY_TRAIN_SHA:
        raise ValueError("Frozen MASSIVE v1 TRAIN SHA mismatch")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in _jsonl(path):
        metadata = row.get("audit_metadata", {})
        source_id, locale = metadata.get("source_id"), metadata.get("source_locale")
        if (
            row.get("split") != "train"
            or metadata.get("source_partition") != "train"
            or not isinstance(source_id, str)
            or locale not in massive.LOCALES
        ):
            raise ValueError("MASSIVE v1 source lineage changed")
        groups[source_id].add(locale)
    if len(groups) != 600 or any(
        locales != set(massive.LOCALES) for locales in groups.values()
    ):
        raise ValueError("MASSIVE v1 group inventory changed")
    return set(groups)


def _leaf_texts(value: Any) -> list[str]:
    """Collect literal utterances embedded in audit-only structured states."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [text for child in value.values() for text in _leaf_texts(child)]
    if isinstance(value, list):
        return [text for child in value for text in _leaf_texts(child)]
    return []


def _context_rows(texts: list[str], prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{prefix}:{number}",
            "state": text,
            "instructions": "",
            "options": [],
            "task_type": "context",
        }
        for number, text in enumerate(texts)
    ]


def protected_inputs(
    protected_list: Path,
    base: Path,
    select: Path,
    cal: Path,
    source: dict[str, dict[str, dict]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references, receipts = score_v6._load_protected(protected_list)
    for role, path in (("base", base), ("select", select), ("cal", cal)):
        if pilot.sha_file(path) != REFERENCE_SHA[role]:
            raise ValueError(f"Frozen {role} SHA mismatch")
        rows = _jsonl(path)
        references[role] = rows
        receipts.append(
            {"role": role, "sha256": REFERENCE_SHA[role], "rows": len(rows)}
        )
    # Only utterance fields are used for official DEV overlap; no DEV intent
    # enters selection or output. MASSIVE TEST rows are never materialized here.
    dev_text = [
        source[locale][identifier]["utt"]
        for identifier, row in source["en-US"].items()
        if row["partition"] == "dev"
        for locale in massive.LOCALES
    ]
    references["official_dev_utterances"] = [
        {"id": f"dev:{number}", "state": text} for number, text in enumerate(dev_text)
    ]
    receipts.append({"role": "official_dev_utterances", "rows": len(dev_text)})
    texts: list[str] = []
    for role in sorted(references):
        for row in references[role]:
            texts.extend(_leaf_texts(row["state"]))
    if not texts:
        raise ValueError("Protected utterance index is empty")
    return _context_rows(texts, "protected"), sorted(receipts, key=lambda r: r["role"])


def candidates(
    source: dict[str, dict[str, dict]], excluded: set[str]
) -> dict[str, list[str]]:
    eligible = set(massive.qualify(source)["train"]) - excluded
    by_intent: dict[str, list[str]] = {intent: [] for intent in SOURCE_INTENT_ORDER}
    for identifier in eligible:
        intent = source["en-US"][identifier]["intent"]
        if intent in by_intent:
            by_intent[intent].append(identifier)
    for intent, identifiers in by_intent.items():
        identifiers.sort(
            key=lambda identifier: (_rank("select", intent, identifier), identifier)
        )
        if len(identifiers) < GROUPS_PER_INTENT:
            raise ValueError(f"Insufficient qualified TRAIN supply for {intent}")
    return by_intent


def overlap_filter(
    source: dict[str, dict[str, dict]],
    by_intent: dict[str, list[str]],
    protected: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    pool_ids = [identifier for ids in by_intent.values() for identifier in ids]
    pool = [
        {
            "id": f"candidate:{identifier}:{locale}",
            "state": source[locale][identifier]["utt"],
            "instructions": "",
            "options": [],
            "task_type": "context",
        }
        for identifier in pool_ids
        for locale in massive.LOCALES
    ]
    protected_hashes = {targeted.text_hashes(row["state"]) for row in protected}
    raw = {value[0] for value in protected_hashes}
    normalized = {value[1] for value in protected_hashes}
    exact_ids = {
        row["id"]
        for row in pool
        if (pair := targeted.text_hashes(row["state"]))[0] in raw
        or pair[1] in normalized
    }
    near = pilot.near_duplicates(pool, protected, collect_left_ids=True)
    near_ids = set(near.pop("left_ids"))
    blocked = {
        identifier
        for identifier in pool_ids
        if any(
            f"candidate:{identifier}:{locale}" in exact_ids | near_ids
            for locale in massive.LOCALES
        )
    }
    selected: list[str] = []
    chosen_context: list[dict[str, Any]] = []
    clone_quarantine = Counter()
    for intent in SOURCE_INTENT_ORDER:
        count = 0
        for identifier in by_intent[intent]:
            if identifier in blocked:
                continue
            current = [
                {
                    "id": f"candidate:{identifier}:{locale}",
                    "state": source[locale][identifier]["utt"],
                    "instructions": "",
                    "options": [],
                    "task_type": "context",
                }
                for locale in massive.LOCALES
            ]
            if chosen_context:
                same = {targeted.text_hashes(row["state"]) for row in chosen_context}
                near_chosen = pilot.near_duplicates(current, chosen_context)
                if near_chosen["count"] or any(
                    targeted.text_hashes(row["state"]) in same for row in current
                ):
                    clone_quarantine[intent] += 1
                    continue
            selected.append(identifier)
            chosen_context.extend(current)
            count += 1
            if count == GROUPS_PER_INTENT:
                break
        if count != GROUPS_PER_INTENT:
            raise ValueError(f"Only {count} non-overlap groups remain for {intent}")
    audit = {
        "qualified_pool_groups": len(pool_ids),
        "protected_exact_rows": len(exact_ids),
        "protected_near_rows": len(near_ids),
        "quarantined_source_groups": len(blocked),
        "cross_selected_clone_skips": dict(sorted(clone_quarantine.items())),
        "near_method": near["method"],
        "near_retrieval_is_approximate": True,
    }
    return selected, audit


def make_rows(
    source: dict[str, dict[str, dict]], identifiers: list[str]
) -> list[dict[str, Any]]:
    options = [
        {"key": intent, "description": INTENTS[intent]}
        for intent in SOURCE_INTENT_ORDER
    ]
    rows = []
    for identifier in identifiers:
        english = source["en-US"][identifier]
        group_id = f"massive-v5:{_rank('group', identifier)[:24]}"
        label = SOURCE_INTENT_ORDER.index(english["intent"])
        for locale in massive.LOCALES:
            item = source[locale][identifier]
            row = {
                "id": f"massive-v5:{_rank('row', identifier, locale)[:24]}",
                "group_id": group_id,
                "source": "amazon_massive_1.1_official",
                "state": item["utt"],
                "instructions": INSTRUCTION,
                "options": options,
                "label": label,
                "task_type": "choice",
                "family": "massive_v5_stable_intent_choice",
                "language": locale.split("-")[0],
                "split": "train",
                "evaluation_role": "train",
                "render_template": "massive_v5_four_stable_intent_ids_v1",
                "audit_metadata": {
                    "source_id": identifier,
                    "source_partition": "train",
                    "source_locale": locale,
                    "intent": english["intent"],
                    "scenario": english["scenario"],
                    "passing_localization_votes": item.get("passing_votes"),
                    "upstream_license": "CC-BY-4.0",
                    "training_approved": False,
                },
            }
            row["input_sha256"] = pilot.input_sha256(row)
            validate_row(row, "train")
            rows.append(row)
    if len(rows) != 84 or len({row["group_id"] for row in rows}) != 12:
        raise ValueError("Pilot group or row count changed")
    return rows


def make_review_packets(
    rows: list[dict[str, Any]], salt: bytes
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    local, parallel, key = [], [], []
    for group_id, group_rows in by_group.items():
        token = hashlib.sha256(salt + b"\0group\0" + group_id.encode()).hexdigest()[:32]
        localized = []
        english = None
        for row in group_rows:
            locale = row["audit_metadata"]["source_locale"]
            row_token = hashlib.sha256(
                salt + b"\0row\0" + row["id"].encode()
            ).hexdigest()[:32]
            local.append(
                {
                    "review_id": row_token,
                    "locale": locale,
                    "state": row["state"],
                    "instructions": row["instructions"],
                    "options": row["options"],
                }
            )
            key.append(
                {
                    "review_id": row_token,
                    "group_token": token,
                    "source_id": row["audit_metadata"]["source_id"],
                    "locale": locale,
                    "intent": row["audit_metadata"]["intent"],
                    "gold_option_key": row["options"][row["label"]]["key"],
                    "candidate_id": row["id"],
                }
            )
            if locale == "en-US":
                english = row["state"]
            else:
                localized.append({"locale": locale, "utterance": row["state"]})
        if english is None or len(localized) != 6:
            raise ValueError("Incomplete parallel source group")
        parallel.append(
            {
                "group_token": token,
                "english_anchor": english,
                "localized": sorted(localized, key=lambda x: x["locale"]),
            }
        )
    local.sort(
        key=lambda row: hashlib.sha256(salt + row["review_id"].encode()).hexdigest()
    )
    parallel.sort(
        key=lambda row: hashlib.sha256(salt + row["group_token"].encode()).hexdigest()
    )
    key.sort(key=lambda row: row["review_id"])
    if len(local) != 84 or len(parallel) != 12 or len(key) != 84:
        raise ValueError("Blind packet count changed")
    if any("source_id" in row or "intent" in row or "label" in row for row in local):
        raise ValueError("Gold leaked into local packet")
    return local, parallel, key


def build(args: argparse.Namespace) -> dict[str, Any]:
    source, source_counts = massive.load_source(args.source_directory, args.archive)
    excluded = legacy_ids(args.legacy_v1_train)
    protected, receipts = protected_inputs(
        args.protected_list, args.base_train, args.select_file, args.cal_file, source
    )
    selected, overlap = overlap_filter(source, candidates(source, excluded), protected)
    rows = make_rows(source, selected)
    # Full prompt check is separate from the utterance-only selection gate.
    references, _ = score_v6._load_protected(args.protected_list)
    full_refs = [row for role in references for row in references[role]]
    if pilot.near_duplicates(rows, full_refs)["count"]:
        raise ValueError("Full prompt overlaps protected gold-free reference")
    salt = secrets.token_bytes(32)
    local, parallel, key = make_review_packets(rows, salt)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(args.output, 0o700)
    output = {
        "candidate.private.jsonl": rows,
        "blind-local.jsonl": local,
        "blind-parallel.jsonl": parallel,
        "private-key.jsonl": key,
    }
    for filename, content in output.items():
        _write_jsonl(args.output / filename, content)
    shutil.copyfile(args.source_directory / "LICENSE", args.output / "LICENSE")
    shutil.copyfile(args.source_directory / "NOTICE.md", args.output / "NOTICE.md")
    os.chmod(args.output / "LICENSE", 0o600)
    os.chmod(args.output / "NOTICE.md", 0o600)
    manifest = {
        "schema_version": "decision2-massive-v5-stable-intent-pilot/1",
        "training_approved": False,
        "pilot_status": "AWAIT_INDEPENDENT_BLIND_REVIEW",
        "source_partition": "train",
        "source_archive_sha256": massive.ARCHIVE_SHA,
        "source_locale_sha256": massive.SOURCE_SHA,
        "source_counts": source_counts,
        "source_license_sha256": massive.LICENSE_SHA,
        "source_notice_sha256": massive.NOTICE_SHA,
        "source_attribution": ["Amazon MASSIVE 1.1", "SLURP"],
        "license": "CC-BY-4.0",
        "builder_sha256": pilot.sha_file(Path(__file__)),
        "legacy_v1_train_sha256": LEGACY_TRAIN_SHA,
        "protected_inventory_sha256": pilot.sha_file(args.protected_list),
        "protected_sources": receipts,
        "selected_group_count": len(selected),
        "selected_row_count": len(rows),
        "intent_group_counts": dict(
            sorted(Counter(source["en-US"][x]["intent"] for x in selected).items())
        ),
        "locale_row_counts": dict(
            sorted(
                Counter(row["audit_metadata"]["source_locale"] for row in rows).items()
            )
        ),
        "overlap": overlap,
        "review_salt_sha256": hashlib.sha256(salt).hexdigest(),
        "outputs": {
            name: pilot.sha_file(args.output / name)
            for name in (*output, "LICENSE", "NOTICE.md")
        },
        "limitations": [
            "Intent labels and human localization votes are provisional; independent review is required.",
            "Near-overlap retrieval is approximate and cannot prove semantic independence.",
            "Choice-only taxonomy mapping; no Noul, Score, compositional reasoning or multilingual gain claim.",
        ],
    }
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        os.chmod(args.output / "manifest.json", 0o600)
        stream.write(pilot.canonical(manifest) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--legacy-v1-train", required=True, type=Path)
    parser.add_argument("--protected-list", required=True, type=Path)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--select-file", required=True, type=Path)
    parser.add_argument("--cal-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "training_approved": manifest["training_approved"],
                "groups": manifest["selected_group_count"],
                "rows": manifest["selected_row_count"],
                "overlap": manifest["overlap"],
                "output_sha256": manifest["outputs"],
            }
        )
    )


if __name__ == "__main__":
    main()
