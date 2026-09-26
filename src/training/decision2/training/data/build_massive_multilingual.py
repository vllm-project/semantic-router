"""Build an audited, private MASSIVE 1.1 intent Choice feasibility set.

Only official TRAIN and DEV IDs can enter the output. Official TEST labels and
JevArena final gold are never read by this builder. Every source ID is a single
seven-locale lineage group. The output is a *review candidate*, not approved
training data until a separate semantic sign-off is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, validate_row

SEED = "decision2-massive-1.1-seven-locale-v1"
LOCALES = ("en-US", "ar-SA", "de-DE", "es-ES", "fr-FR", "ja-JP", "zh-CN")
SOURCE_SHA = {
    "en-US": "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e",
    "ar-SA": "b604b44d3bb94e4f71f64d041229e16ceb42d7b89385df7c82873d2514737186",
    "de-DE": "5e09cc550b38d37faf002e0ff42103acd330906e23032689e8f071e1cf3ff621",
    "es-ES": "310462a79fa181ff83c643a8d356c7b8155fd37a25e80a77ba3ca9b29305c4a5",
    "fr-FR": "f9bf3db170ad415b389e4c9594dd0f8f80c38188143e05cc4459a6fa7df7cf49",
    "ja-JP": "c22df382db6aa4a23dd1e7f62a2ac8f01c6158865771ad25201696be7201ab79",
    "zh-CN": "992bf0bef3d678f08c27e514739bc851163e8f40f530bfb4d5970a2c24408ace",
}
ARCHIVE_SHA = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
LICENSE_SHA = "c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435"
NOTICE_SHA = "b90534ccd20c6f0e1e5239567af0d150496339542b75a15bfbc3e1e737593ddb"

# All references are input-only. None is a FINAL gold or target file.
REFERENCE_FILES = {
    "human_train": (
        "data/balanced_human_5824_v1/balanced_human_5824.train.jsonl",
        "e83fb07021b779bb86d6b1d773b007c2dda9d91052aedf1f72f89bebbfef50e2",
    ),
    "human_select": (
        "data/balanced_human_5824_v1/select.jsonl",
        "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38",
    ),
    "human_cal": (
        "data/balanced_human_5824_v1/cal.jsonl",
        "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf",
    ),
    "structured_train": (
        "data/nox4b_structured_mix_v1/nox4b_structured.train.jsonl",
        "773cd53d21663095a4208e6e35de6654bdca4af5d314e23b03582dbe70eae87f",
    ),
    "structured_select": (
        "data/nox4b_structured_mix_v1/select.jsonl",
        "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38",
    ),
    "structured_cal": (
        "data/nox4b_structured_mix_v1/cal.jsonl",
        "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf",
    ),
    "clean_train": (
        "data/rights_clean_goemotions_v2/rights_clean.train.jsonl",
        "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    ),
    "clean_select": (
        "data/rights_clean_goemotions_v2/select.jsonl",
        "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    ),
    "clean_cal": (
        "data/rights_clean_goemotions_v2/cal.jsonl",
        "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    ),
    "dev": (
        "runs/dev.prompts.jsonl",
        "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    ),
    "css_pilot": (
        "runs/css-transfer-v1/css-pilot.prompts.jsonl",
        "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
    ),
    "css_evaluation": (
        "data/gold-free-references/css-evaluation.prompts.jsonl",
        "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    ),
    "rq1": (
        "runs/pressure-dev-v1/rq1_extended_capacity.prompts.jsonl",
        "f1584742583dacab334e0f5ca7702e7248991d7f58f41176e87e14b14446aa19",
    ),
    "rq2": (
        "runs/pressure-dev-v1/rq2_shared_order.prompts.jsonl",
        "e094f222e01240f22b2cc508893ce554a3850961958f72ddb6d20551572b9df8",
    ),
    "rq3": (
        "runs/pressure-dev-v1/rq3_shared_hardness.prompts.jsonl",
        "be73de25f7fd3202d2a7dfd8831c338c70cce344a01551313eab066670055b32",
    ),
    "jevbench_public": (
        "bench/jevbench-public-231/prompts.jsonl",
        "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
    ),
    "decisionbench_public": (
        "bench/decision-bench-v4-text-readable/prompts.jsonl",
        "41c0e4728202d800972edb31375e92e92148906856b2f20396154c8d0a9c80da",
    ),
    "authored_dev": (
        "bench/jev-arena-authored-v4-candidate/dev-240-v1/prompts.jsonl",
        "9b704aa3dbeaf58199fb6fc9ec1d4b2f7cce21c27dae3279e34654dc7789b9fc",
    ),
    "multilingual_dev": (
        "bench/jev-arena-multilingual-dev-v1/prompts.jsonl",
        "1f1882e545cdc4189834c9c055f458e3cc02acf662e0f294e3c8076a3977de92",
    ),
    "multilingual_parallel_dev": (
        "bench/jev-arena-multilingual-parallel-dev-v1/prompts.jsonl",
        "d4f83ba17030ddb1591b1ec7da22ae851859fd2d0e2ce6020997a8ba96ffad39",
    ),
}

INSTRUCTIONS = {
    "en-US": "Which action is the speaker asking the assistant to perform? Choose one of the six options.",
    "ar-SA": "ما الإجراء الذي يطلبه المتحدث من المساعد؟ اختر خيارًا واحدًا من الخيارات الستة.",
    "de-DE": "Welche Aktion soll der Assistent für diese Anfrage ausführen? Wähle eine der sechs Optionen.",  # codespell:ignore aktion,assistent
    "es-ES": "¿Qué acción le pide esta solicitud al asistente? Elige una de las seis opciones.",
    "fr-FR": "Quelle action cette demande veut-elle que l’assistant effectue ? Choisissez l’une des six options.",
    "ja-JP": "この依頼でアシスタントに求められている操作は何ですか。6つの選択肢から1つ選んでください。",
    "zh-CN": "这句话要求助手执行什么操作？请从六个选项中选择一个。",
}

# Manually checked against the official intent taxonomy and English examples.
# These English descriptions are deliberately shared across all seven locales.
INTENT_DESCRIPTIONS = {
    "alarm_query": "Check existing alarms",
    "alarm_remove": "Cancel an alarm",
    "alarm_set": "Set an alarm",
    "audio_volume_down": "Lower audio volume",
    "audio_volume_mute": "Mute audio",
    "audio_volume_other": "Change audio volume without a specified direction",
    "audio_volume_up": "Raise audio volume",
    "calendar_query": "Check calendar events",
    "calendar_remove": "Remove a calendar event",
    "calendar_set": "Add a calendar event or reminder",
    "cooking_query": "Ask what to cook",
    "cooking_recipe": "Ask for recipe steps or ingredients",
    "datetime_convert": "Convert time between zones",
    "datetime_query": "Ask the date or time",
    "email_addcontact": "Add an email contact",
    "email_query": "Read or check email",
    "email_querycontact": "Find a contact's details",
    "email_sendemail": "Send an email",
    "general_greet": "Greet the assistant",
    "general_joke": "Ask for a joke",
    "general_quirky": "Make a general unusual assistant request",
    "iot_cleaning": "Start household cleaning",
    "iot_coffee": "Make coffee",
    "iot_hue_lightchange": "Change light color or scene",
    "iot_hue_lightdim": "Dim the lights",
    "iot_hue_lightoff": "Turn the lights off",
    "iot_hue_lighton": "Turn the lights on",
    "iot_hue_lightup": "Brighten the lights",
    "iot_wemo_off": "Turn a smart plug off",
    "iot_wemo_on": "Turn a smart plug on",
    "lists_createoradd": "Create a list or add an item",
    "lists_query": "Read a list",
    "lists_remove": "Remove an item from a list",
    "music_dislikeness": "Express dislike of music",
    "music_likeness": "Express liking of music",
    "music_query": "Ask about music",
    "music_settings": "Change music playback settings",
    "news_query": "Ask for news",
    "play_audiobook": "Play or control an audiobook",
    "play_game": "Start a game",
    "play_music": "Play music",
    "play_podcasts": "Play or control a podcast",
    "play_radio": "Play radio",
    "qa_currency": "Ask about currency exchange",
    "qa_definition": "Ask for a definition",
    "qa_factoid": "Ask a factual question",
    "qa_maths": "Ask a math question",
    "qa_stock": "Ask about stock prices",
    "recommendation_events": "Ask for event recommendations",
    "recommendation_locations": "Ask for place recommendations",
    "recommendation_movies": "Ask for movie recommendations",
    "social_post": "Post to social media",
    "social_query": "Check social media",
    "takeaway_order": "Order takeaway food",
    "takeaway_query": "Ask about takeaway food options",
    "transport_query": "Ask for travel directions",
    "transport_taxi": "Book a taxi",
    "transport_ticket": "Find or book travel tickets",
    "transport_traffic": "Ask about traffic",
    "weather_query": "Ask about weather",
}
AMBIGUOUS_NEIGHBORS = (
    ("cooking_query", "cooking_recipe"),
    ("qa_definition", "qa_factoid"),
    ("iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightup"),
    ("audio_volume_other", "audio_volume_down", "audio_volume_up"),
)


def sha(path: Path) -> str:
    return pilot.sha_file(path)


def rank(*parts: str) -> str:
    return hashlib.sha256("\0".join((SEED, *parts)).encode()).hexdigest()


def quality_pass(judgments: Any) -> bool:
    if not isinstance(judgments, list) or len(judgments) > 3:
        raise ValueError("MASSIVE human judgment field changed")
    if len(judgments) != 3:
        return False  # The preregistered rule is two *of three* actual votes.

    def passed(j: dict) -> bool:
        return (
            j.get("intent_score") in (1, 2)
            and type(j.get("grammar_score")) is int
            and j["grammar_score"] >= 3
            and "target" in str(j.get("language_identification", "")).split("|")
        )

    return sum(passed(j) for j in judgments) >= 2


def load_source(
    directory: Path, archive: Path
) -> tuple[dict[str, dict[str, dict]], dict]:
    if (
        sha(archive) != ARCHIVE_SHA
        or sha(directory / "LICENSE") != LICENSE_SHA
        or sha(directory / "NOTICE.md") != NOTICE_SHA
    ):
        raise ValueError("Official MASSIVE archive, license or notice changed")
    source: dict[str, dict[str, dict]] = {}
    counts: dict[str, dict[str, int]] = {}
    for locale in LOCALES:
        path = directory / "data" / f"{locale}.jsonl"
        if sha(path) != SOURCE_SHA[locale]:
            raise ValueError(f"Official MASSIVE {locale} source changed")
        records: dict[str, dict] = {}
        stats = Counter()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("locale") != locale or not isinstance(row.get("id"), str):
                    raise ValueError(f"MASSIVE {locale} identity changed")
                stats[row["partition"]] += 1
                if row["partition"] == "test":
                    continue  # Official TEST intent/utterance/labels are not used.
                if row["partition"] not in ("train", "dev") or row["id"] in records:
                    raise ValueError(f"MASSIVE {locale} split/ID changed")
                if (
                    not isinstance(row.get("utt"), str)
                    or not row["utt"].strip()
                    or not isinstance(row.get("intent"), str)
                    or not isinstance(row.get("scenario"), str)
                    or row["intent"] not in INTENT_DESCRIPTIONS
                ):
                    raise ValueError(f"MASSIVE {locale} row malformed")
                row = {
                    key: row[key]
                    for key in (
                        "id",
                        "locale",
                        "partition",
                        "scenario",
                        "intent",
                        "utt",
                    )
                }
                if locale != "en-US":
                    original = json.loads(line)
                    judgments = original.get("judgments")
                    row["quality_pass"] = quality_pass(judgments)
                    if len(judgments) != 3:
                        stats[f"{row['partition']}_incomplete_judgments"] += 1
                    row["passing_votes"] = sum(
                        j.get("intent_score") in (1, 2)
                        and type(j.get("grammar_score")) is int
                        and j["grammar_score"] >= 3
                        and "target"
                        in str(j.get("language_identification", "")).split("|")
                        for j in judgments
                    )
                    if row["quality_pass"]:
                        stats[f"{row['partition']}_quality_pass"] += 1
                records[row["id"]] = row
        if (
            stats["train"] != 11514
            or stats["dev"] != 2033
            or stats["test"] != 2974
            or len(records) != 13547
        ):
            raise ValueError(f"MASSIVE {locale} split inventory changed")
        source[locale] = records
        counts[locale] = dict(sorted(stats.items()))
    english = source["en-US"]
    if len({row["intent"] for row in english.values()}) != 60 or set(
        INTENT_DESCRIPTIONS
    ) != {row["intent"] for row in english.values()}:
        raise ValueError("MASSIVE intent taxonomy changed")
    for locale in LOCALES[1:]:
        if set(source[locale]) != set(english):
            raise ValueError(f"MASSIVE {locale} source ID alignment changed")
        for identifier, row in source[locale].items():
            baseline = english[identifier]
            if any(
                row[key] != baseline[key] for key in ("partition", "scenario", "intent")
            ):
                raise ValueError(f"MASSIVE {locale} partition/intent alignment changed")
    return source, counts


def qualify(source: dict[str, dict[str, dict]]) -> dict[str, list[str]]:
    english = source["en-US"]
    groups = {split: [] for split in ("train", "dev")}
    for identifier, row in english.items():
        if all(source[locale][identifier]["quality_pass"] for locale in LOCALES[1:]):
            groups[row["partition"]].append(identifier)
    groups = {
        key: sorted(
            value, key=lambda identifier: (rank("pool", identifier), identifier)
        )
        for key, value in groups.items()
    }
    if {key: len(value) for key, value in groups.items()} != {
        "train": 9191,
        "dev": 1626,
    }:
        raise ValueError("Frozen six-locale consensus inventory changed")
    return groups


def balanced_quotas(
    source: dict[str, dict[str, dict]], eligible: list[str], total: int, role: str
) -> dict[str, int]:
    available = Counter(
        source["en-US"][identifier]["intent"] for identifier in eligible
    )
    intents = sorted(intent for intent in INTENT_DESCRIPTIONS if available[intent])
    if (
        (role == "train" and len(intents) != len(INTENT_DESCRIPTIONS))
        or total < len(intents)
        or total > len(eligible)
    ):
        raise ValueError(
            f"MASSIVE {role} cannot cover its source intents at requested total"
        )
    quotas = dict.fromkeys(intents, 1)
    while sum(quotas.values()) < total:
        candidates = [
            intent for intent in intents if quotas[intent] < available[intent]
        ]
        if not candidates:
            raise ValueError(f"MASSIVE {role} group supply exhausted")
        intent = min(
            candidates, key=lambda name: (quotas[name], rank("quota", role, name), name)
        )
        quotas[intent] += 1
    return quotas


def shortlist(
    source: dict[str, dict[str, dict]], groups: dict[str, list[str]]
) -> tuple[dict[str, list[str]], dict]:
    targets = {"train": 600, "dev": 200}
    selected = {}
    counts = {}
    for role, identifiers in groups.items():
        quotas = balanced_quotas(source, identifiers, targets[role], role)
        by_intent = defaultdict(list)
        for identifier in identifiers:
            by_intent[source["en-US"][identifier]["intent"]].append(identifier)
        picked = []
        for intent in sorted(quotas):
            ordered = sorted(
                by_intent[intent],
                key=lambda identifier: (rank("select", intent, identifier), identifier),
            )
            limit = min(len(ordered), max(quotas[intent] + 8, 2 * quotas[intent]))
            picked.extend(ordered[:limit])
        selected[role] = picked
        counts[role] = {
            "quality_pool_groups": len(identifiers),
            "shortlist_groups": len(picked),
            "quality_supply_per_intent": dict(
                sorted(
                    Counter(
                        source["en-US"][identifier]["intent"]
                        for identifier in identifiers
                    ).items()
                )
            ),
            "prequarantine_quotas": quotas,
            "shortlist_rule": "per-intent frozen hash order, min(supply,max(quota+8,2*quota))",
        }
    return selected, counts


def load_references(root: Path) -> tuple[list[dict], dict]:
    references = []
    receipt = {}
    seen_files = {}
    for role, (relative, digest) in REFERENCE_FILES.items():
        path = root / relative
        if sha(path) != digest:
            raise ValueError(f"Frozen protected {role} changed")
        if digest not in seen_files:
            seen_files[digest], _ = targeted.load_context_reference(path)
            references.extend(seen_files[digest])
        rows = seen_files[digest]
        receipt[role] = {"sha256": digest, "rows": len(rows), "name": path.name}
    return references, receipt


def context(row: dict) -> dict:
    return {
        "id": f"massive-1.1:{row['id']}:{row['locale']}",
        "group_id": f"massive-1.1:{row['id']}",
        "state": row["utt"],
        "input_sha256": targeted.text_hashes(row["utt"])[1],
    }


def quarantine(
    source: dict[str, dict[str, dict]],
    groups: dict[str, list[str]],
    references: list[dict],
) -> tuple[dict[str, list[str]], dict]:
    protected_ids = {row["id"] for row in references}
    protected_groups = {row.get("group_id") for row in references}
    hashes = {targeted.text_hashes(row["state"]) for row in references}
    raw, normalized = {p[0] for p in hashes}, {p[1] for p in hashes}
    candidate = [
        context(source[locale][identifier])
        for split in ("train", "dev")
        for identifier in groups[split]
        for locale in LOCALES
    ]
    flagged = set()
    reasons = Counter()
    remaining = []
    for row in candidate:
        a, b = targeted.text_hashes(row["state"])
        hits = []
        if row["id"] in protected_ids or row["group_id"] in protected_groups:
            hits.append("id_or_group")
        if a in raw or b in normalized:
            hits.append("exact_or_normalized_context")
        if hits:
            flagged.add(row["group_id"])
            reasons.update(hits)
        else:
            remaining.append(row)
    near = pilot.near_duplicates(
        targeted.context_rows(remaining),
        targeted.context_rows(references),
        collect_left_ids=True,
    )
    near_ids = set(near.pop("left_ids"))
    flagged.update(row["group_id"] for row in remaining if row["id"] in near_ids)
    reasons["near_context_rows"] = len(near_ids)
    eligible = {
        split: [
            identifier
            for identifier in groups[split]
            if f"massive-1.1:{identifier}" not in flagged
        ]
        for split in ("train", "dev")
    }
    return eligible, {
        "candidate_groups": {k: len(v) for k, v in groups.items()},
        "eligible_groups": {k: len(v) for k, v in eligible.items()},
        "quarantined_groups": len(flagged),
        "trigger_rows": dict(sorted(reasons.items())),
        "near_method": near["method"],
        "approximate_near_is_not_exhaustive": True,
    }


def select_ids(
    source: dict[str, dict[str, dict]],
    eligible: list[str],
    total: int,
    *,
    reserved_contexts: set[tuple[str, str]] | None = None,
    require_all_intents: bool = False,
) -> list[str]:
    by_intent = defaultdict(list)
    for identifier in eligible:
        by_intent[source["en-US"][identifier]["intent"]].append(identifier)
    intents = sorted(by_intent)
    if require_all_intents and len(intents) != len(INTENT_DESCRIPTIONS):
        raise ValueError("MASSIVE TRAIN lacks a required intent")
    for intent in intents:
        by_intent[intent].sort(
            key=lambda identifier: (rank("select", intent, identifier), identifier)
        )
    used_contexts = set() if reserved_contexts is None else set(reserved_contexts)
    pointers = Counter()
    counts = Counter()
    exhausted = set()
    chosen = []
    while len(chosen) < total:
        candidates = [intent for intent in intents if intent not in exhausted]
        if not candidates:
            raise ValueError("Insufficient nonduplicate MASSIVE source groups")
        intent = min(
            candidates,
            key=lambda name: (counts[name], rank("select-intent", name), name),
        )
        accepted = False
        while pointers[intent] < len(by_intent[intent]):
            identifier = by_intent[intent][pointers[intent]]
            pointers[intent] += 1
            fingerprints = {
                (locale, targeted.text_hashes(source[locale][identifier]["utt"])[1])
                for locale in LOCALES
            }
            if fingerprints & used_contexts:
                continue
            chosen.append(identifier)
            used_contexts.update(fingerprints)
            counts[intent] += 1
            accepted = True
            break
        if not accepted:
            exhausted.add(intent)
            if counts[intent] == 0:
                raise ValueError(f"No nonduplicate MASSIVE {intent} group remains")
    if len(counts) != len(intents):
        raise ValueError("MASSIVE selection lost an intent")
    return sorted(
        chosen, key=lambda identifier: (rank("output", identifier), identifier)
    )


def option_intents(
    intent: str,
    scenario: str,
    all_intents: list[str],
    scenarios: dict[str, str],
    identifier: str,
) -> list[str]:
    same = sorted(
        (
            name
            for name in all_intents
            if name != intent and scenarios[name] == scenario
        ),
        key=lambda name: (rank("same-scenario", identifier, name), name),
    )
    other = sorted(
        (name for name in all_intents if scenarios[name] != scenario),
        key=lambda name: (rank("other-scenario", identifier, name), name),
    )
    picked = [intent, *same[:3]]
    picked.extend(other[: 6 - len(picked)])
    if (
        len(picked) != 6
        or len(set(picked)) != 6
        or (len(same) >= 2 and len(set(picked) & set(same)) < 2)
    ):
        raise ValueError("Dynamic option construction lost hard negatives")
    return sorted(
        picked, key=lambda name: (rank("option-position", identifier, name), name)
    )


def make_rows(
    source: dict[str, dict[str, dict]], identifiers: list[str], role: str
) -> list[dict]:
    all_intents = sorted(INTENT_DESCRIPTIONS)
    scenarios = {row["intent"]: row["scenario"] for row in source["en-US"].values()}
    rows = []
    for identifier in identifiers:
        english = source["en-US"][identifier]
        choices = option_intents(
            english["intent"], english["scenario"], all_intents, scenarios, identifier
        )
        options = [
            {"key": chr(65 + i), "description": INTENT_DESCRIPTIONS[name]}
            for i, name in enumerate(choices)
        ]
        gold = choices.index(english["intent"])
        for locale in LOCALES:
            item = source[locale][identifier]
            row = {
                "id": f"massive-1.1:{identifier}:{locale}",
                "group_id": f"massive-1.1:{identifier}",
                "source": "amazon_massive_1.1_official",
                "state": item["utt"],
                "instructions": INSTRUCTIONS[locale],
                "options": options,
                "label": gold,
                "task_type": "choice",
                "family": "massive_localized_intent_choice",
                "language": locale.split("-")[0],
                "split": role,
                "evaluation_role": "train" if role == "train" else "select",
                "render_template": "massive_1.1_intent_six_option_v1",
                "audit_metadata": {
                    "source_id": identifier,
                    "source_partition": english["partition"],
                    "source_locale": locale,
                    "intent": english["intent"],
                    "scenario": english["scenario"],
                    "passing_localization_votes": item.get("passing_votes"),
                    "dynamic_option_intents": choices,
                    "upstream_license": "CC-BY-4.0",
                },
            }
            row["input_sha256"] = pilot.input_sha256(row)
            validate_row(row, role)
            rows.append(row)
    return rows


def cross_split_audit(train: list[dict], dev: list[dict]) -> dict:
    check_partition_isolation({"train": train, "select": dev})
    train_hashes = {targeted.text_hashes(row["state"])[1] for row in train}
    duplicate = [
        row["id"]
        for row in dev
        if targeted.text_hashes(row["state"])[1] in train_hashes
    ]
    near = pilot.near_duplicates(
        targeted.context_rows(dev), targeted.context_rows(train), collect_left_ids=True
    )
    near_ids = near.pop("left_ids")
    if duplicate or near_ids:
        raise ValueError(
            f"MASSIVE train/DEV context collision: exact={len(duplicate)} near={len(near_ids)}"
        )
    return {
        "exact_context_rows": 0,
        "near_context_rows": 0,
        "near_method": near["method"],
        "group_isolation": True,
    }


def quarantine_train_neighbors(
    source: dict[str, dict[str, dict]], train_ids: list[str], dev_eligible: list[str]
) -> tuple[list[str], dict]:
    """Remove whole DEV source groups resembling any selected TRAIN locale."""
    train = [
        context(source[locale][identifier])
        for identifier in train_ids
        for locale in LOCALES
    ]
    dev = [
        context(source[locale][identifier])
        for identifier in dev_eligible
        for locale in LOCALES
    ]
    train_hashes = {targeted.text_hashes(row["state"])[1] for row in train}
    exact_rows = {
        row["id"]
        for row in dev
        if targeted.text_hashes(row["state"])[1] in train_hashes
    }
    near = pilot.near_duplicates(
        targeted.context_rows(dev), targeted.context_rows(train), collect_left_ids=True
    )
    near_rows = set(near.pop("left_ids"))
    flagged_rows = exact_rows | near_rows
    blocked_groups = {row["group_id"] for row in dev if row["id"] in flagged_rows}
    kept = [
        identifier
        for identifier in dev_eligible
        if f"massive-1.1:{identifier}" not in blocked_groups
    ]
    return kept, {
        "candidate_groups": len(dev_eligible),
        "quarantined_groups": len(blocked_groups),
        "eligible_groups": len(kept),
        "exact_context_rows": len(exact_rows),
        "near_context_rows": len(near_rows),
        "near_method": near["method"],
    }


def make_blind_review(
    source: dict[str, dict[str, dict]], train_ids: list[str]
) -> tuple[list[dict], list[dict]]:
    intents = sorted(INTENT_DESCRIPTIONS)
    scenarios = {row["intent"]: row["scenario"] for row in source["en-US"].values()}
    selected = [
        next(
            identifier
            for identifier in train_ids
            if source["en-US"][identifier]["intent"] == intent
        )
        for intent in intents
    ]
    packet, key = [], []
    for identifier in selected:
        english = source["en-US"][identifier]
        choices = option_intents(
            english["intent"], english["scenario"], intents, scenarios, identifier
        )
        options = [
            {"key": chr(65 + i), "description": INTENT_DESCRIPTIONS[name]}
            for i, name in enumerate(choices)
        ]
        for locale in LOCALES:
            review_id = rank("blind-review", identifier, locale)[:20]
            packet.append(
                {
                    "review_id": review_id,
                    "parallel_group": rank("parallel-group", identifier)[:20],
                    "locale": locale,
                    "english_reference": english["utt"],
                    "localized_utterance": source[locale][identifier]["utt"],
                    "instruction": INSTRUCTIONS[locale],
                    "options": options,
                }
            )
            key.append(
                {
                    "review_id": review_id,
                    "source_id": identifier,
                    "locale": locale,
                    "intent": english["intent"],
                    "gold_option_key": chr(65 + choices.index(english["intent"])),
                    "passing_votes": source[locale][identifier].get("passing_votes"),
                }
            )
    return packet, key


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def build(
    archive: Path, source_directory: Path, workspace_root: Path, output: Path
) -> dict[str, Any]:
    if (
        output.exists()
        or output.is_symlink()
        or output.with_name(output.name + ".pending").exists()
    ):
        raise FileExistsError(output)
    source, source_counts = load_source(source_directory, archive)
    qualified = qualify(source)
    candidates, shortlist_receipt = shortlist(source, qualified)
    references, reference_receipt = load_references(workspace_root)
    eligible, quarantine_receipt = quarantine(source, candidates, references)
    train_ids = select_ids(source, eligible["train"], 600, require_all_intents=True)
    safe_dev, cross_quarantine = quarantine_train_neighbors(
        source, train_ids, eligible["dev"]
    )
    train_contexts = {
        (locale, targeted.text_hashes(source[locale][identifier]["utt"])[1])
        for identifier in train_ids
        for locale in LOCALES
    }
    dev_ids = select_ids(source, safe_dev, 200, reserved_contexts=train_contexts)
    train, dev = make_rows(source, train_ids, "train"), make_rows(
        source, dev_ids, "select"
    )
    cross = cross_split_audit(train, dev)
    if (
        len(train_ids) != 600
        or len(dev_ids) != 200
        or len(train) != 4200
        or len(dev) != 1400
    ):
        raise ValueError("MASSIVE preregistered group/row quota was not met")
    # A reviewer sees paired utterances/options without the intended label.
    review, review_key = make_blind_review(source, train_ids)
    stage = output.with_name(output.name + ".pending")
    stage.mkdir(parents=True, mode=0o700)
    write_jsonl(stage / "train.private.jsonl", train)
    write_jsonl(stage / "dev.private.jsonl", dev)
    write_jsonl(stage / "semantic-review.private.jsonl", review)
    write_jsonl(stage / "semantic-review-key.private.jsonl", review_key)
    manifest = {
        "schema_version": "decision2-massive-multilingual-feasibility/1",
        "training_approved": False,
        "research_only": True,
        "no_public_raw_text": True,
        "builder_sha256": sha(Path(__file__)),
        "seed": SEED,
        "source": {
            "release": "amazon-massive-dataset-1.1",
            "archive_sha256": ARCHIVE_SHA,
            "locale_sha256": SOURCE_SHA,
            "license_sha256": LICENSE_SHA,
            "notice_sha256": NOTICE_SHA,
            "license": "CC-BY-4.0",
            "attribution": [
                "Amazon MASSIVE 1.1",
                "MASSIVE ACL 2023 paper",
                "SLURP EMNLP 2020 paper",
            ],
        },
        "source_counts": source_counts,
        "quality_groups": {k: len(v) for k, v in qualified.items()},
        "shortlist": shortlist_receipt,
        "protected_references": reference_receipt,
        "quarantine": quarantine_receipt,
        "selected_source_groups": {"train": len(train_ids), "dev": len(dev_ids)},
        "selected_rows": {"train": len(train), "dev": len(dev)},
        "selected_intents": {
            role: dict(
                sorted(Counter(row["audit_metadata"]["intent"] for row in rows).items())
            )
            for role, rows in (("train", train), ("dev", dev))
        },
        "intent_coverage": {
            "train": len({row["audit_metadata"]["intent"] for row in train}),
            "dev": len({row["audit_metadata"]["intent"] for row in dev}),
            "dev_missing": sorted(
                set(INTENT_DESCRIPTIONS)
                - {row["audit_metadata"]["intent"] for row in dev}
            ),
        },
        "selected_locale": {
            role: dict(sorted(Counter(row["language"] for row in rows).items()))
            for role, rows in (("train", train), ("dev", dev))
        },
        "cross_split": cross,
        "cross_split_candidate_quarantine": cross_quarantine,
        "semantics": {
            "review_packet_rows": len(review),
            "review_packet_gold_blind": True,
            "option_descriptions": "English, manually reviewed against official intent examples",
            "localized_instruction_languages": list(LOCALES),
            "known_ambiguous_neighbor_sets": AMBIGUOUS_NEIGHBORS,
            "manual_localized_semantic_signoff_pending": True,
        },
        "outputs": {
            name: {
                "sha256": sha(stage / name),
                "rows": sum(1 for _ in (stage / name).open()),
            }
            for name in (
                "train.private.jsonl",
                "dev.private.jsonl",
                "semantic-review.private.jsonl",
                "semantic-review-key.private.jsonl",
            )
        },
        "limitations": [
            "Only six-option intent Choice; no Score, Noul, or long-context claim.",
            "English option descriptions create mixed-language prompts.",
            "Official quality-filtered DEV has no audio_volume_other source ID.",
            "Quality consensus is annotation evidence, not a complete semantic proof.",
            "Approximate near-context retrieval is not a completeness guarantee.",
            "Official TEST IDs and all FINAL gold remain unused.",
        ],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(stage, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.archive, args.source_directory, args.workspace_root, args.output
    )
    print(
        json.dumps(
            {
                "quality_groups": result["quality_groups"],
                "selected_source_groups": result["selected_source_groups"],
                "selected_rows": result["selected_rows"],
                "training_approved": result["training_approved"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
