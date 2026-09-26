"""Freeze a small, gold-blind MASSIVE pilot with six localized option sets.

Only ten source groups that passed every locale in the independently sealed v2
review can enter this packet. It is an audit artifact, never approved TRAIN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

LOCALES = ("ar-SA", "de-DE", "es-ES", "fr-FR", "ja-JP", "zh-CN")
SEED = "decision2-massive-v3-localized-options-10-groups-v1"
CANDIDATE_MANIFEST_SHA = (
    "193b68b92f588f1d1e13352ff4752eab9048fbdd37e1f0ea88d1b514b5431a10"
)
CANDIDATE_TRAIN_SHA = "55d7d711a130165be5a1d43930c836c7356938c33577e50fa22fdc937d976b80"
PRIOR_RECEIPT_SHA = "a996345ec056491ce2c33c9c2f20a3b63e2e24fad1d0705904efe2c3a7c4ef25"
PRIOR_PACKET_SHA = "5ed2c30dde9ac2def316c070b021b19fed1b0858cbb46abc8a3607638830abb5"
PRIOR_KEY_SHA = "9d67a07587bf28e5bc17ca097d5bce64926523ca15d0d71a97142f303af7379f"
PRIOR_BLIND_VERDICT_SHA = (
    "a11e8add7ff18f9960c9735e9eab25bcd4c8436dde3b8fd65fc0a021b2658f1c"
)
PRIOR_BLIND_MANIFEST_SHA = (
    "a9be0cec6295f75e3315707561927d450cab55e6bfab54520dd4174b8d8900ba"
)
PRIOR_POSTKEY_SHA = "f2888be0091041db9fa7029f3f5b768507c2bc309c555f7da6d2d98045d889a8"
PRIOR_POSTKEY_MANIFEST_SHA = (
    "915e800bdf804d92bb0a046264fc9262a8fe8b5de91ef8929d3e1cd13afbeff1"
)
LICENSE_SHA = "c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435"
NOTICE_SHA = "b90534ccd20c6f0e1e5239567af0d150496339542b75a15bfbc3e1e737593ddb"
ARCHIVE_SHA = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Frozen input digest changed: {path.name}")


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


def opaque(*parts: str) -> str:
    return hashlib.sha256("\0".join((SEED, *parts)).encode()).hexdigest()[:24]


def index_unique(rows: list[dict], field: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row.get(field)
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Duplicate or invalid {field}")
        result[key] = row
    return result


def choose_groups(
    packet: list[dict], key: list[dict], verdict: list[dict], postkey: list[dict]
) -> tuple[list[dict], dict]:
    """Apply sealed v2 verdicts to whole seven-locale source groups only."""
    if any(len(rows) != 108 for rows in (packet, key, verdict, postkey)):
        raise ValueError("Prior independent review must cover all 108 rows")
    by_packet = index_unique(packet, "review_id")
    by_key = index_unique(key, "review_id")
    by_verdict = index_unique(verdict, "review_id")
    by_postkey = index_unique(postkey, "review_id")
    ids = set(by_packet)
    if any(set(rows) != ids for rows in (by_key, by_verdict, by_postkey)):
        raise ValueError("Prior packet, key, verdict and post-key IDs differ")
    groups = defaultdict(list)
    for review_id in ids:
        p, k, v, post = (
            rows[review_id] for rows in (by_packet, by_key, by_verdict, by_postkey)
        )
        if (
            len(p.get("options", [])) != 6
            or [option.get("key") for option in p["options"]] != list("ABCDEF")
            or not isinstance(p.get("english_utterance"), str)
            or not isinstance(p.get("localized_utterance"), str)
            or p["locale"] not in LOCALES
            or any(
                row.get("parallel_group") != p["parallel_group"] for row in (k, v, post)
            )
            or any(row.get("locale") != p["locale"] for row in (k, v, post))
            or post.get("key_option") != k.get("gold_option_key")
            or post.get("source_id") != k.get("source_id")
            or post.get("source_intent") != k.get("source_intent")
            or post.get("blind_label_validity") != v.get("label_validity")
            or post.get("blind_strict_parallel_pass") != v.get("strict_parallel_pass")
            or not isinstance(k.get("source_id"), str)
            or not isinstance(k.get("source_intent"), str)
            or k.get("gold_option_key") not in "ABCDEF"
        ):
            raise ValueError(
                "Prior independent review lineage or option contract changed"
            )
        groups[p["parallel_group"]].append((p, k, v))
    if len(groups) != 18:
        raise ValueError("Expected 18 independently sampled source groups")
    if len({rows[0][1]["source_id"] for rows in groups.values()}) != 18:
        raise ValueError("Prior source groups are not independent")

    selected, quarantined = [], Counter()
    for group, rows in sorted(groups.items()):
        locales = {p["locale"] for p, _, _ in rows}
        ids = {k["source_id"] for _, k, _ in rows}
        intents = {k["source_intent"] for _, k, _ in rows}
        english = {p["english_utterance"] for p, _, _ in rows}
        options = {json.dumps(p["options"], sort_keys=True) for p, _, _ in rows}
        golds = {k["gold_option_key"] for _, k, _ in rows}
        if (
            len(rows) != 6
            or locales != set(LOCALES)
            or any(
                len(values) != 1 for values in (ids, intents, english, options, golds)
            )
        ):
            raise ValueError("Incomplete or inconsistent seven-locale source group")
        for _, _, v in rows:
            if v.get("strict_parallel_pass") is not True:
                quarantined["semantic_or_label_failure"] += 1
                break
            if (
                v.get("label_validity") is not True
                or v.get("source_intent_preserved") is not True
                or v.get("semantic_tier") not in {"exact", "minor"}
                or v.get("localized_utterance_tier") != "target_language"
            ):
                quarantined["policy_failure"] += 1
                break
        else:
            selected.extend(rows)
    if len(selected) != 60 or len({p["parallel_group"] for p, _, _ in selected}) != 10:
        raise ValueError("Expected exactly ten all-six strict and label-valid groups")
    if sum(quarantined.values()) != 8:
        raise ValueError("Expected eight quarantined source groups")
    if len({k["source_id"] for _, k, _ in selected}) != 10:
        raise ValueError("Selected source IDs are not independent")
    return selected, dict(sorted(quarantined.items()))


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
            or text.strip().casefold() == english.casefold()
        ):
            raise ValueError("Option description was not localized")
        localized.append({"key": option["key"], "description": text})
    if len({option["description"] for option in localized}) != len(localized):
        raise ValueError("Localized options collapse distinct choices")
    return localized


def build(candidate: Path, prior: Path, output: Path) -> dict:
    stage = output.with_name(output.name + ".pending")
    if any(path.exists() or path.is_symlink() for path in (output, stage)):
        raise FileExistsError(output)
    sources = {
        candidate / "manifest.json": CANDIDATE_MANIFEST_SHA,
        candidate / "train.private.jsonl": CANDIDATE_TRAIN_SHA,
        candidate / "LICENSE": LICENSE_SHA,
        candidate / "NOTICE.md": NOTICE_SHA,
        prior / "receipt.json": PRIOR_RECEIPT_SHA,
        prior / "locale-pilot.blind.private.jsonl": PRIOR_PACKET_SHA,
        prior / "locale-pilot.key.private.jsonl": PRIOR_KEY_SHA,
        prior / "locale-pilot.blind-verdict.private.jsonl": PRIOR_BLIND_VERDICT_SHA,
        prior
        / "locale-pilot.blind-verdict.manifest.private.json": PRIOR_BLIND_MANIFEST_SHA,
        prior / "locale-pilot.postkey.private.jsonl": PRIOR_POSTKEY_SHA,
        prior
        / "locale-pilot.postkey.manifest.private.json": PRIOR_POSTKEY_MANIFEST_SHA,
    }
    for path, digest in sources.items():
        verified(path, digest)
    candidate_manifest = json.loads((candidate / "manifest.json").read_text())
    blind_manifest = json.loads(
        (prior / "locale-pilot.blind-verdict.manifest.private.json").read_text()
    )
    postkey_manifest = json.loads(
        (prior / "locale-pilot.postkey.manifest.private.json").read_text()
    )
    prior_receipt = json.loads((prior / "receipt.json").read_text())
    if (
        candidate_manifest.get("training_approved") is not False
        or candidate_manifest.get("outputs", {}).get("train.private.jsonl")
        != CANDIDATE_TRAIN_SHA
        or candidate_manifest.get("source_groups", {}).get("kept_train") != 481
        or candidate_manifest.get("rights", {}).get("license") != "CC-BY-4.0"
        or blind_manifest.get("status") != "SEALED_BEFORE_KEY_EXPOSURE"
        or blind_manifest.get("answer_key_or_gold_accessed") is not False
        or blind_manifest.get("verdict_sha256") != PRIOR_BLIND_VERDICT_SHA
        or blind_manifest.get("packet_sha256") != PRIOR_PACKET_SHA
        or postkey_manifest.get("status")
        != "POST_KEY_JOIN_AFTER_IMMUTABLE_BLIND_REVIEW"
        or postkey_manifest.get("training_approved") is not False
        or postkey_manifest.get("inputs_sha256", {}).get("blind_verdict")
        != PRIOR_BLIND_VERDICT_SHA
        or postkey_manifest.get("postkey_sha256") != PRIOR_POSTKEY_SHA
        or prior_receipt.get("training_approved") is not False
        or prior_receipt.get("outputs", {}).get("locale-pilot.key.private.jsonl")
        != PRIOR_KEY_SHA
    ):
        raise ValueError("MASSIVE v2 rights or sealed review evidence is inconsistent")

    selected, quarantined = choose_groups(
        read_jsonl(prior / "locale-pilot.blind.private.jsonl"),
        read_jsonl(prior / "locale-pilot.key.private.jsonl"),
        read_jsonl(prior / "locale-pilot.blind-verdict.private.jsonl"),
        read_jsonl(prior / "locale-pilot.postkey.private.jsonl"),
    )
    translations_path = Path(__file__).with_name("massive_v3_option_descriptions.json")
    translations = json.loads(translations_path.read_text(encoding="utf-8"))
    expected_descriptions = {
        option["description"]
        for packet, _, _ in selected
        for option in packet["options"]
    }
    if set(translations) != expected_descriptions or any(
        not isinstance(value, list) or len(value) != len(LOCALES)
        for value in translations.values()
    ):
        raise ValueError("Localized option inventory differs from selected pilot")
    packet_rows, key_rows = [], []
    for old_packet, old_key, _ in selected:
        locale = old_packet["locale"]
        source_id = old_key["source_id"]
        review_id = opaque("review", source_id, locale)
        group_id = opaque("group", source_id)
        packet_rows.append(
            {
                "review_id": review_id,
                "parallel_group": group_id,
                "locale": locale,
                "english_utterance": old_packet["english_utterance"],
                "localized_utterance": old_packet["localized_utterance"],
                "english_instruction": old_packet["english_instruction"],
                "localized_instruction": old_packet["localized_instruction"],
                "english_options": old_packet["options"],
                "localized_options": localize_options(
                    old_packet["options"], locale, translations
                ),
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "parallel_group": group_id,
                "locale": locale,
                "source_id": source_id,
                "source_intent": old_key["source_intent"],
                "gold_option_key": old_key["gold_option_key"],
                "prior_review_id": old_packet["review_id"],
            }
        )
    if (
        len(packet_rows) != 60
        or len(index_unique(packet_rows, "review_id")) != 60
        or Counter(row["locale"] for row in packet_rows) != dict.fromkeys(LOCALES, 10)
    ):
        raise ValueError("Pilot size or locale balance drift")
    packet_rows.sort(key=lambda row: row["review_id"])
    key_rows.sort(key=lambda row: row["review_id"])
    stage.mkdir(parents=True, mode=0o700)
    try:
        write_jsonl(stage / "locale-v3.blind.private.jsonl", packet_rows)
        write_jsonl(stage / "locale-v3.key.private.jsonl", key_rows)
        shutil.copy2(candidate / "LICENSE", stage / "LICENSE")
        shutil.copy2(candidate / "NOTICE.md", stage / "NOTICE.md")
        receipt = {
            "schema_version": "decision2-massive-v3-locale-options-pilot/1",
            "training_approved": False,
            "research_only": True,
            "no_public_raw_text": True,
            "review_status": "pending_independent_gold_blind_adjudication",
            "source_archive_sha256": ARCHIVE_SHA,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA,
            "candidate_train_sha256": CANDIDATE_TRAIN_SHA,
            "prior_review_sha256": {
                "receipt": PRIOR_RECEIPT_SHA,
                "packet": PRIOR_PACKET_SHA,
                "key": PRIOR_KEY_SHA,
                "blind_verdict": PRIOR_BLIND_VERDICT_SHA,
                "blind_manifest": PRIOR_BLIND_MANIFEST_SHA,
                "postkey": PRIOR_POSTKEY_SHA,
                "postkey_manifest": PRIOR_POSTKEY_MANIFEST_SHA,
            },
            "builder_sha256": sha(Path(__file__)),
            "translations_sha256": sha(translations_path),
            "selection_rule": "Whole groups with six independent strict passes, exact/minor semantics, valid unique option, preserved source intent, and target-language utterance; no refills",
            "prior_sampled_groups": 18,
            "quarantined_groups": 8,
            "quarantine_reason_counts": quarantined,
            "selected_groups": 10,
            "translated_pairs": 60,
            "per_locale": dict.fromkeys(LOCALES, 10),
            "rights": {
                "license": "CC-BY-4.0",
                "attribution": [
                    "Amazon MASSIVE 1.1",
                    "MASSIVE ACL 2023",
                    "SLURP EMNLP 2020",
                ],
            },
            "known_limits": [
                "The independent v2 review sampled 18 of 481 English-approved TRAIN groups; only ten enter this pilot.",
                "A prior strict pass does not validate the newly localized option descriptions.",
                "This packet does not approve full multilingual TRAIN or model publication.",
                "No heldout DEV, TEST or JevArena FINAL labels were used to select or relabel rows.",
            ],
            "outputs": {
                name: sha(stage / name)
                for name in (
                    "locale-v3.blind.private.jsonl",
                    "locale-v3.key.private.jsonl",
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
    parser.add_argument("--prior-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.candidate, args.prior_review, args.output)
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
