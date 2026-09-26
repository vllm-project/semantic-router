"""Freeze a gold-blind six-language semantic review pilot from MASSIVE v2.

The source v2 candidate is unapproved. This packet is for independent review
only; it never changes labels or grants training approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_MANIFEST_SHA = (
    "193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10"
)
EXPECTED_TRAIN_SHA = "55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80"
SEED = "decision2-massive-v2-locale-pilot-18-groups-v1"
LOCALES = ("ar-SA", "de-DE", "es-ES", "fr-FR", "ja-JP", "zh-CN")

# Fixed before viewing any cross-locale semantic verdict. One independent
# source group is selected per intent; each contributes all six translations.
INTENT_STRATA = {
    "sparse": (
        "audio_volume_other",
        "general_quirky",
        "transport_query",
        "music_dislikeness",
        "recommendation_movies",
        "iot_hue_lighton",
    ),
    "prior_risk": (
        "datetime_convert",
        "qa_definition",
        "play_podcasts",
        "social_post",
        "transport_ticket",
        "recommendation_events",
    ),
    "broad_scenarios": (
        "alarm_set",
        "email_sendemail",
        "weather_query",
        "lists_query",
        "takeaway_order",
        "iot_hue_lightoff",
    ),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque(*parts: str) -> str:
    return hashlib.sha256("\0".join((SEED, *parts)).encode()).hexdigest()[:24]


def verified(path: Path, digest: str) -> None:
    if sha(path) != digest:
        raise ValueError(f"Frozen MASSIVE v2 input changed: {path.name}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )


def select_source_groups(groups: dict[str, list[dict]]) -> list[tuple[str, str, str]]:
    by_intent = defaultdict(list)
    for source_id, rows in groups.items():
        english = [
            row for row in rows if row["audit_metadata"]["source_locale"] == "en-US"
        ]
        if len(english) != 1:
            raise ValueError("MASSIVE source group lacks exactly one English row")
        by_intent[english[0]["audit_metadata"]["intent"]].append(source_id)
    intents = [intent for names in INTENT_STRATA.values() for intent in names]
    if len(intents) != 18 or len(set(intents)) != 18:
        raise ValueError("Preregistered locale pilot strata changed")
    selected = []
    for tier, names in INTENT_STRATA.items():
        if len(names) != 6:
            raise ValueError("Each locale review stratum needs six intents")
        for intent in names:
            candidates = by_intent[intent]
            if not candidates:
                raise ValueError(f"No retained MASSIVE v2 source group for {intent}")
            source_id = min(
                candidates, key=lambda item: (opaque("sample", intent, item), item)
            )
            selected.append((tier, intent, source_id))
    if len({item[2] for item in selected}) != 18:
        raise ValueError("Locale pilot source groups are not independent")
    return selected


def build(
    candidate: Path,
    output: Path,
    *,
    expected_manifest_sha: str = EXPECTED_MANIFEST_SHA,
    expected_train_sha: str = EXPECTED_TRAIN_SHA,
) -> dict:
    stage = output.with_name(output.name + ".pending")
    if output.exists() or output.is_symlink() or stage.exists() or stage.is_symlink():
        raise FileExistsError(output)
    verified(candidate / "manifest.json", expected_manifest_sha)
    manifest = json.loads((candidate / "manifest.json").read_text())
    if (
        manifest.get("training_approved") is not False
        or manifest.get("cross_locale_semantic_review_pending") is not True
        or manifest.get("source_groups", {}).get("kept_train") != 481
        or manifest.get("rows", {}).get("train") != 3367
        or manifest.get("train_intent_coverage") != 59
        or manifest.get("outputs", {}).get("train.private.jsonl") != expected_train_sha
    ):
        raise ValueError("Expected the exact unapproved English-filtered MASSIVE v2")
    verified(candidate / "train.private.jsonl", expected_train_sha)
    for name, field in (
        ("LICENSE", "original_license_sha256"),
        ("NOTICE.md", "original_notice_sha256"),
    ):
        verified(candidate / name, manifest["rights"][field])

    groups = defaultdict(list)
    for row in read_jsonl(candidate / "train.private.jsonl"):
        groups[row["audit_metadata"]["source_id"]].append(row)
    if len(groups) != 481 or any(len(rows) != 7 for rows in groups.values()):
        raise ValueError("MASSIVE v2 seven-locale group inventory changed")
    selected = select_source_groups(groups)
    packet, key = [], []
    for tier, intent, source_id in selected:
        rows = groups[source_id]
        by_locale = {row["audit_metadata"]["source_locale"]: row for row in rows}
        if set(by_locale) != {"en-US", *LOCALES}:
            raise ValueError("Selected MASSIVE v2 group lost a locale")
        english = by_locale["en-US"]
        options = english["options"]
        if (
            english["audit_metadata"]["intent"] != intent
            or not 0 <= english["label"] < len(options)
            or len(options) != 6
            or [option["key"] for option in options] != list("ABCDEF")
        ):
            raise ValueError("Selected MASSIVE v2 intent or option contract changed")
        group_token = opaque("parallel-group", source_id)
        for locale in LOCALES:
            row = by_locale[locale]
            if (
                row["options"] != options
                or row["label"] != english["label"]
                or row["audit_metadata"]["intent"] != intent
                or row["group_id"] != english["group_id"]
            ):
                raise ValueError("Parallel group option, label or intent drift")
            review_id = opaque("review-row", source_id, locale)
            packet.append(
                {
                    "review_id": review_id,
                    "parallel_group": group_token,
                    "locale": locale,
                    "english_utterance": english["state"],
                    "localized_utterance": row["state"],
                    "english_instruction": english["instructions"],
                    "localized_instruction": row["instructions"],
                    "options": options,
                }
            )
            key.append(
                {
                    "review_id": review_id,
                    "parallel_group": group_token,
                    "source_id": source_id,
                    "locale": locale,
                    "source_intent": intent,
                    "gold_option_key": options[english["label"]]["key"],
                    "selection_stratum": tier,
                }
            )
    if (
        len(packet) != 108
        or len(key) != 108
        or len({row["review_id"] for row in packet}) != 108
        or len({row["parallel_group"] for row in packet}) != 18
        or Counter(row["locale"] for row in packet) != dict.fromkeys(LOCALES, 18)
    ):
        raise ValueError("Preregistered 18-by-six locale review quota not met")
    packet.sort(key=lambda row: row["review_id"])
    key.sort(key=lambda row: row["review_id"])
    stage.mkdir(parents=True, mode=0o700)
    write_jsonl(stage / "locale-pilot.blind.private.jsonl", packet)
    write_jsonl(stage / "locale-pilot.key.private.jsonl", key)
    for name in ("LICENSE", "NOTICE.md"):
        shutil.copy2(candidate / name, stage / name)
    receipt = {
        "schema_version": "decision2-massive-v2-locale-blind-pilot/1",
        "training_approved": False,
        "review_status": "pending_independent_gold_blind_adjudication",
        "candidate_manifest_sha256": expected_manifest_sha,
        "candidate_train_sha256": expected_train_sha,
        "builder_sha256": sha(Path(__file__)),
        "seed": SEED,
        "sample_rule": "one source ID per fixed intent, minimum SHA256(seed,intent,source_id)",
        "intent_strata": INTENT_STRATA,
        "sampled_source_groups": 18,
        "sampled_translated_pairs": 108,
        "pairs_per_locale": dict.fromkeys(LOCALES, 18),
        "packet_gold_blind": True,
        "key_separate": True,
        "known_limits": [
            "A pilot review of 18 of 481 retained source groups cannot certify all translations.",
            "The 59-intent TRAIN lacks cooking_query; the unchanged DEV covers only 52 intents.",
            "No training or release approval follows from packet construction.",
        ],
        "outputs": {
            name: sha(stage / name)
            for name in (
                "locale-pilot.blind.private.jsonl",
                "locale-pilot.key.private.jsonl",
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
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.candidate, args.output)
    print(
        json.dumps(
            {
                "sampled_source_groups": receipt["sampled_source_groups"],
                "sampled_translated_pairs": receipt["sampled_translated_pairs"],
                "training_approved": receipt["training_approved"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
