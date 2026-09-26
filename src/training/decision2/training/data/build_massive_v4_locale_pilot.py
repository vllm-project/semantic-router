"""Freeze the preregistered MASSIVE v4 exact-option locale audit packet.

The output is a private research packet and separate key, never approved TRAIN.
No source utterance or answer key is changed by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SEED = "decision2-massive-v4-locale-pilot-v1"
LOCALES = ("ar-SA", "de-DE", "es-ES", "fr-FR", "ja-JP", "zh-CN")
INTENTS = (
    "alarm_query",
    "alarm_remove",
    "audio_volume_down",
    "audio_volume_mute",
    "audio_volume_up",
    "email_addcontact",
    "email_query",
    "email_querycontact",
    "general_joke",
    "iot_hue_lightoff",
    "qa_currency",
    "qa_maths",
)
CANDIDATE_MANIFEST_SHA = (
    "193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10"
)
CANDIDATE_TRAIN_SHA = "55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80"
V3_KEY_SHA = "db9e53762879a6ebc604927125836a4e8b19982984b19a600675a63642710e25"
TRANSLATIONS_SHA = "8b71e471bef575b15c3e0745ac8c81a031b88e048f5694a9d5fbdac8b09d4a43"
LICENSE_SHA = "c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435"
NOTICE_SHA = "b90534ccd20c6f0e1e5239567af0d150496339542b75a15bfbc3e1e737593ddb"
ARCHIVE_SHA = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
PREREG_GIST_COMMIT = "df95a13"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Frozen input digest changed: {path.name}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )


def opaque(*parts: str) -> str:
    return hashlib.sha256("\0".join((SEED, *parts)).encode()).hexdigest()[:24]


def rank(intent: str, source_id: str) -> tuple[str, str]:
    digest = hashlib.sha256("\0".join((SEED, intent, source_id)).encode()).hexdigest()
    return digest, source_id


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def excluded_v3_ids(key_rows: list[dict]) -> set[str]:
    if len(key_rows) != 60 or len({row.get("review_id") for row in key_rows}) != 60:
        raise ValueError("V3 key must contain 60 unique review rows")
    groups = defaultdict(list)
    for row in key_rows:
        if (
            not isinstance(row.get("source_id"), str)
            or not row["source_id"]
            or row.get("gold_option_key") not in "ABCDEF"
        ):
            raise ValueError("V3 key lineage is invalid")
        groups[row["source_id"]].append(row)
    if len(groups) != 10 or any(
        len(rows) != 6
        or {row.get("locale") for row in rows} != set(LOCALES)
        or len({row.get("parallel_group") for row in rows}) != 1
        for rows in groups.values()
    ):
        raise ValueError("V3 source-group exclusion is incomplete")
    return set(groups)


def select_groups(
    source_rows: list[dict], translations: dict, excluded: set[str]
) -> tuple[list[tuple[str, str, list[dict]]], dict]:
    groups = defaultdict(list)
    for row in source_rows:
        source_id = row.get("audit_metadata", {}).get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("MASSIVE source row lacks a source ID")
        groups[source_id].append(row)
    if (
        len(source_rows) != 3367
        or len(groups) != 481
        or any(len(rows) != 7 for rows in groups.values())
    ):
        raise ValueError("Frozen v2 seven-locale inventory changed")
    if not excluded.issubset(groups):
        raise ValueError("V3 exclusion refers to a missing v2 group")

    by_intent = defaultdict(list)
    eligible = 0
    remaining = 0
    for source_id, rows in groups.items():
        english = [
            r
            for r in rows
            if r.get("audit_metadata", {}).get("source_locale") == "en-US"
        ]
        if len(english) != 1:
            raise ValueError("Source group lacks exactly one English row")
        anchor = english[0]
        options = anchor.get("options")
        if not isinstance(options, list) or len(options) != 6:
            raise ValueError("English source group lacks six options")
        if not all(option.get("description") in translations for option in options):
            continue
        eligible += 1
        if source_id in excluded:
            continue
        remaining += 1
        by_intent[anchor["audit_metadata"]["intent"]].append(source_id)
    if (eligible, remaining) != (93, 83):
        raise ValueError("Preregistered option-coverage inventory changed")

    selected = []
    for intent in INTENTS:
        candidates = by_intent[intent]
        if len(candidates) < 2:
            raise ValueError(f"Preregistered stratum lacks two candidates: {intent}")
        source_id = min(candidates, key=lambda item: rank(intent, item))
        selected.append((intent, source_id, groups[source_id]))
    if len({source_id for _, source_id, _ in selected}) != 12:
        raise ValueError("V4 selected source groups are not independent")
    return selected, {
        "translatable": eligible,
        "remaining_after_v3_exclusion": remaining,
    }


def localize_options(
    options: list[dict], locale: str, translations: dict
) -> list[dict]:
    index = LOCALES.index(locale)
    localized = []
    for option in options:
        english = option["description"]
        text = translations[english][index]
        if (
            not isinstance(text, str)
            or not text.strip()
            or normalized(text) == normalized(english)
        ):
            raise ValueError("Option description was not localized")
        localized.append({"key": option["key"], "description": text})
    if len({normalized(option["description"]) for option in localized}) != 6:
        raise ValueError("Localized options collapse distinct choices")
    return localized


def make_rows(
    selected: list[tuple[str, str, list[dict]]], translations: dict
) -> tuple[list[dict], list[dict]]:
    packet, key = [], []
    for intent, source_id, rows in selected:
        by_locale = {r["audit_metadata"]["source_locale"]: r for r in rows}
        if len(by_locale) != 7 or set(by_locale) != {"en-US", *LOCALES}:
            raise ValueError("Selected group lacks a unique seven-locale inventory")
        english = by_locale["en-US"]
        options = english["options"]
        if (
            english["audit_metadata"]["intent"] != intent
            or len(options) != 6
            or [o.get("key") for o in options] != list("ABCDEF")
            or len({normalized(o["description"]) for o in options}) != 6
            or not isinstance(english.get("label"), int)
            or not 0 <= english["label"] < 6
        ):
            raise ValueError("English label or option contract changed")
        for locale in LOCALES:
            row = by_locale[locale]
            if (
                row["options"] != options
                or row["label"] != english["label"]
                or row["group_id"] != english["group_id"]
                or row["audit_metadata"]["intent"] != intent
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in (
                        english["state"],
                        row["state"],
                        english["instructions"],
                        row["instructions"],
                    )
                )
            ):
                raise ValueError("Selected parallel source group changed")
            review_id = opaque("review", source_id, locale)
            group_id = opaque("group", source_id)
            packet.append(
                {
                    "review_id": review_id,
                    "parallel_group": group_id,
                    "locale": locale,
                    "english_utterance": english["state"],
                    "localized_utterance": row["state"],
                    "english_instruction": english["instructions"],
                    "localized_instruction": row["instructions"],
                    "english_options": options,
                    "localized_options": localize_options(
                        options, locale, translations
                    ),
                }
            )
            key.append(
                {
                    "review_id": review_id,
                    "parallel_group": group_id,
                    "locale": locale,
                    "source_id": source_id,
                    "source_intent": intent,
                    "gold_option_key": options[english["label"]]["key"],
                }
            )
    if (
        len(packet) != 72
        or len({r["review_id"] for r in packet}) != 72
        or len({r["parallel_group"] for r in packet}) != 12
        or Counter(r["locale"] for r in packet) != dict.fromkeys(LOCALES, 12)
    ):
        raise ValueError("Preregistered 12-by-six packet quota not met")
    return (
        sorted(packet, key=lambda row: row["review_id"]),
        sorted(key, key=lambda row: row["review_id"]),
    )


def build(candidate: Path, v3_key: Path, output: Path) -> dict:
    stage = output.with_name(output.name + ".pending")
    if any(path.exists() or path.is_symlink() for path in (output, stage)):
        raise FileExistsError(output)
    sources = {
        candidate / "manifest.json": CANDIDATE_MANIFEST_SHA,
        candidate / "train.private.jsonl": CANDIDATE_TRAIN_SHA,
        candidate / "LICENSE": LICENSE_SHA,
        candidate / "NOTICE.md": NOTICE_SHA,
        v3_key: V3_KEY_SHA,
    }
    for path, expected in sources.items():
        verify(path, expected)
    manifest = json.loads((candidate / "manifest.json").read_text())
    if (
        manifest.get("training_approved") is not False
        or manifest.get("source_groups", {}).get("kept_train") != 481
        or manifest.get("rows", {}).get("train") != 3367
        or manifest.get("outputs", {}).get("train.private.jsonl") != CANDIDATE_TRAIN_SHA
        or manifest.get("rights", {}).get("license") != "CC-BY-4.0"
        or manifest.get("rights", {}).get("original_license_sha256") != LICENSE_SHA
        or manifest.get("rights", {}).get("original_notice_sha256") != NOTICE_SHA
    ):
        raise ValueError("MASSIVE v2 rights or candidate lineage changed")
    translations_path = Path(__file__).with_name("massive_v3_option_descriptions.json")
    verify(translations_path, TRANSLATIONS_SHA)
    translations = json.loads(translations_path.read_text(encoding="utf-8"))
    if len(translations) != 41 or any(
        not isinstance(values, list)
        or len(values) != len(LOCALES)
        or any(not isinstance(value, str) for value in values)
        for values in translations.values()
    ):
        raise ValueError("Pinned six-language option inventory changed")
    excluded = excluded_v3_ids(read_jsonl(v3_key))
    selected, eligibility = select_groups(
        read_jsonl(candidate / "train.private.jsonl"), translations, excluded
    )
    packet, key = make_rows(selected, translations)

    stage.mkdir(parents=True, mode=0o700)
    try:
        write_jsonl(stage / "locale-v4.blind.private.jsonl", packet)
        write_jsonl(stage / "locale-v4.key.private.jsonl", key)
        shutil.copy2(candidate / "LICENSE", stage / "LICENSE")
        shutil.copy2(candidate / "NOTICE.md", stage / "NOTICE.md")
        receipt = {
            "schema_version": "decision2-massive-v4-locale-exact-option-pilot/1",
            "research_only": True,
            "training_approved": False,
            "review_status": "pending_independent_gold_blind_review",
            "packet_gold_blind": True,
            "key_separate": True,
            "source_archive_sha256": ARCHIVE_SHA,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA,
            "candidate_train_sha256": CANDIDATE_TRAIN_SHA,
            "v3_exclusion_key_sha256": V3_KEY_SHA,
            "translations_sha256": TRANSLATIONS_SHA,
            "builder_sha256": sha(Path(__file__)),
            "prereg_gist_commit": PREREG_GIST_COMMIT,
            "seed": SEED,
            "selection_rule": "One whole group per fixed intent by minimum SHA256 of NUL-separated seed,intent,source ID; exclude all v3 groups; no refills",
            "intents": INTENTS,
            "eligible_groups": eligibility,
            "excluded_v3_groups": len(excluded),
            "selected_groups": 12,
            "translated_pairs": 72,
            "per_locale": dict.fromkeys(LOCALES, 12),
            "rights": {
                "license": "CC-BY-4.0",
                "attribution": [
                    "Amazon MASSIVE 1.1",
                    "MASSIVE ACL 2023",
                    "SLURP EMNLP 2020",
                ],
            },
            "known_limits": [
                "English screening and mechanical translation checks do not establish exact semantic answerability.",
                "Only independent sealed row and all-six group review can advance this pilot to a larger audit.",
                "No full-corpus approval, GPU training, held-out TEST or FINAL evaluation follows from this packet.",
            ],
            "outputs": {
                name: sha(stage / name)
                for name in (
                    "locale-v4.blind.private.jsonl",
                    "locale-v4.key.private.jsonl",
                    "LICENSE",
                    "NOTICE.md",
                )
            },
        }
        (stage / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        for path in stage.iterdir():
            os.chmod(path, 0o600)
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--v3-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.candidate, args.v3_key, args.output)
    print(
        json.dumps(
            {
                "receipt_sha256": sha(args.output / "receipt.json"),
                "outputs": receipt["outputs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
