"""Schema-Guided Dialogue (DSTC8) TRAIN as A1 rows: the goal that a user turn
activates (Choice over the service's schema intents) and whether the latest
user message requests a named slot (Noul).

States are the last <= 8 turns ending at the chosen user turn. Search intents
dominate activations and most slots are never requested, so both families are
balanced before capping or their answers would be predictable without the
state: each service keeps the same number of rows per intent (a service with an
intent that is never chosen is dropped), and each (service, asked slot) keeps
as many false rows as true ones. False rows ask about a slot the service
requests somewhere in TRAIN but the turn does not.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

from v2.data.sources.common import choice_options, make_row, noul_options, rotate, sha

ARM = "a1"
SOURCE = "sgd_dstc8_train"
CHOICE_FAMILY = "sgd_intent"
NOUL_FAMILY = "sgd_requested_slot"
SEED = "a1-sgd-v1"
CAPS = {CHOICE_FAMILY: 1000, NOUL_FAMILY: 1000}
SCHEMA = "train/schema.json"
WINDOW = 8
CHOICE_INSTRUCTIONS = "Which goal is the user pursuing in their latest message?"
SPEAKERS = {"USER": "User", "SYSTEM": "System"}
BOOLEAN_SLOT = re.compile(
    r"boolean flag (?:indicating )?(?:if|whether) (.+)", re.IGNORECASE
)
ARTICLE = re.compile(r"(?:the|a|an) ")

Event = tuple[int, str, Any]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lower_first(text: str) -> str:
    if text[:1].isupper() and text[1:2].islower():
        return text[0].lower() + text[1:]
    return text


def slot_question(description: str) -> str:
    text = " ".join(description.split()).rstrip(".")
    boolean = BOOLEAN_SLOT.fullmatch(text)
    if boolean:
        return f"In the latest message, does the user ask whether {boolean.group(1)}?"
    text = _lower_first(text)
    if text.startswith("whether "):
        return f"In the latest message, does the user ask {text}?"
    if not ARTICLE.match(text):
        text = "the " + text
    return f"In the latest message, does the user ask for {text}?"


def window(turns: list[dict[str, Any]], index: int) -> str:
    return "\n".join(
        f"{SPEAKERS[turn['speaker']]}: {' '.join(turn['utterance'].split())}"
        for turn in turns[max(0, index - WINDOW + 1) : index + 1]
    )


def _user_frames(dialogue: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    return [
        (index, frame["service"], frame["state"])
        for index, turn in enumerate(dialogue["turns"])
        if turn["speaker"] == "USER"
        for frame in turn["frames"]
    ]


def activations(dialogue: dict[str, Any]) -> list[tuple[int, str, str]]:
    """(turn, service, intent) where the service's last known active intent
    changes to a non-NONE intent; a frame absent from a turn keeps its intent."""
    active: dict[str, str] = {}
    found = []
    for index, service, state in _user_frames(dialogue):
        intent = state["active_intent"]
        if intent != "NONE" and intent != active.get(service, "NONE"):
            found.append((index, service, intent))
        active[service] = intent
    return found


def requests(dialogue: dict[str, Any]) -> list[tuple[int, str, tuple[str, ...]]]:
    return [
        (index, service, tuple(sorted(set(state["requested_slots"]))))
        for index, service, state in _user_frames(dialogue)
        if state["requested_slots"]
    ]


def _pick_event(events: list[Event], seed: str) -> Event:
    return min(events, key=lambda event: sha(f"{seed}:{event[0]}:{event[1]}"))


def _pick(names: list[str], seed: str) -> str:
    return min(names, key=lambda name: sha(f"{seed}:{name}"))


def _balanced(
    by_label: dict[Any, list[dict[str, Any]]], labels: list[Any]
) -> list[dict[str, Any]]:
    quota = min(len(by_label.get(label, [])) for label in labels)
    kept = []
    for label in labels:
        members = sorted(
            by_label.get(label, []), key=lambda row: sha(f"{SEED}:balance:{row['id']}")
        )
        kept.extend(members[:quota])
    return kept


def _choice_row(descriptions: list[str], gold: int, **fields: Any) -> dict[str, Any]:
    probe = make_row(options=choice_options(descriptions), label=gold, **fields)
    options, label = rotate(probe["options"], gold, f"{fields['arm']}-v1:{probe['id']}")
    return make_row(options=options, label=label, **fields)


def _choice_rows(
    dialogues: list[dict[str, Any]], services: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], collections.Counter[str], dict[str, Any]]:
    dropped: collections.Counter[str] = collections.Counter()
    usable: dict[str, list[str]] = {}
    ineligible = []
    for name, service in sorted(services.items()):
        descriptions = [intent["description"].lower() for intent in service["intents"]]
        if len(descriptions) >= 2 and len(set(descriptions)) == len(descriptions):
            usable[name] = [intent["name"] for intent in service["intents"]]
        else:
            ineligible.append(name)
    picked: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for dialogue in dialogues:
        events = [event for event in activations(dialogue) if event[1] in usable]
        if not events:
            dropped["dialogues_without_activation"] += 1
            continue
        dialogue_id = dialogue["dialogue_id"]
        index, service, intent = _pick_event(events, f"{SEED}:choice:{dialogue_id}")
        picked[service][intent].append(
            _choice_row(
                [item["description"] for item in services[service]["intents"]],
                usable[service].index(intent),
                arm=ARM,
                source=SOURCE,
                family=CHOICE_FAMILY,
                task_type="choice",
                language="en",
                group_key=dialogue_id,
                local_id=f"{dialogue_id}:{index}:{service}",
                state=window(dialogue["turns"], index),
                instructions=CHOICE_INSTRUCTIONS,
                render_template=f"{CHOICE_FAMILY}/v1",
                audit={
                    "dialogue_id": dialogue_id,
                    "turn_index": index,
                    "service": service,
                    "intent": intent,
                },
            )
        )
    rows: list[dict[str, Any]] = []
    per_intent, missing = {}, []
    for service in sorted(picked):
        total = sum(len(members) for members in picked[service].values())
        kept = _balanced(picked[service], usable[service])
        if not kept:
            missing.append(service)
            dropped["service_with_unchosen_intent"] += total
            continue
        per_intent[service] = len(kept) // len(usable[service])
        dropped["intent_balance"] += total - len(kept)
        rows.extend(kept)
    return (
        rows,
        dropped,
        {
            "services_ineligible": ineligible,
            "services_with_unchosen_intent": missing,
            "rows_per_intent": per_intent,
        },
    )


def _noul_rows(
    dialogues: list[dict[str, Any]], services: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], collections.Counter[str], dict[str, Any]]:
    requestable: dict[str, set[str]] = collections.defaultdict(set)
    for dialogue in dialogues:
        for _, service, requested in requests(dialogue):
            requestable[service].update(requested)
    dropped: collections.Counter[str] = collections.Counter()
    strata: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for dialogue in dialogues:
        events = requests(dialogue)
        if not events:
            dropped["dialogues_without_request"] += 1
            continue
        dialogue_id = dialogue["dialogue_id"]
        index, service, requested = _pick_event(events, f"{SEED}:noul:{dialogue_id}")
        slots = {
            slot["name"]: slot["description"] for slot in services[service]["slots"]
        }
        unknown = sorted(set(requested) - slots.keys())
        if unknown:
            raise ValueError(
                f"SGD {dialogue_id}: requested slots {unknown} not in schema"
            )
        local_id = f"{dialogue_id}:{index}:{service}"
        truth = int(sha(f"{SEED}:truth:{dialogue_id}"), 16) % 2
        taken = {slots[name].lower() for name in requested}
        pool = (
            list(requested)
            if truth
            else sorted(
                name
                for name in requestable[service] - set(requested)
                if slots[name].lower() not in taken
            )
        )
        if not pool:
            dropped["no_false_slot"] += 1
            continue
        asked = _pick(pool, f"{SEED}:slot:{local_id}")
        strata[(service, asked)][truth].append(
            make_row(
                arm=ARM,
                source=SOURCE,
                family=NOUL_FAMILY,
                task_type="noul",
                language="en",
                group_key=dialogue_id,
                local_id=local_id,
                state=window(dialogue["turns"], index),
                instructions=slot_question(slots[asked]),
                options=noul_options("en"),
                label=truth,
                render_template=f"{NOUL_FAMILY}/v1",
                audit={
                    "dialogue_id": dialogue_id,
                    "turn_index": index,
                    "service": service,
                    "asked_slot": asked,
                    "requested_slots": list(requested),
                },
            )
        )
    rows: list[dict[str, Any]] = []
    for key in sorted(strata):
        kept = _balanced(strata[key], [0, 1])
        dropped["slot_balance"] += sum(len(v) for v in strata[key].values()) - len(kept)
        rows.extend(kept)
    return (
        rows,
        dropped,
        {
            "requestable_slots": {
                name: len(requestable[name]) for name in sorted(requestable)
            }
        },
    )


def build(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    names = [
        SCHEMA,
        *sorted(
            path.relative_to(root).as_posix()
            for path in (root / "train").glob("dialogues_*.json")
        ),
    ]
    if len(names) == 1:
        raise ValueError(f"{root}: no train/dialogues_*.json files")
    inputs = {name: file_sha256(root / name) for name in names}
    services = {service["service_name"]: service for service in _json(root / SCHEMA)}
    dialogues = [dialogue for name in names[1:] for dialogue in _json(root / name)]
    if len({dialogue["dialogue_id"] for dialogue in dialogues}) != len(dialogues):
        raise ValueError("duplicate SGD dialogue_id")
    for dialogue in dialogues:
        unknown = {
            service for _, service, _ in _user_frames(dialogue)
        } - services.keys()
        if unknown:
            raise ValueError(
                f"SGD {dialogue['dialogue_id']}: services {sorted(unknown)} not in schema"
            )
    choice, choice_dropped, choice_notes = _choice_rows(dialogues, services)
    noul, noul_dropped, noul_notes = _noul_rows(dialogues, services)
    return {CHOICE_FAMILY: choice, NOUL_FAMILY: noul}, {
        "inputs": inputs,
        "dropped": {
            CHOICE_FAMILY: dict(sorted(choice_dropped.items())),
            NOUL_FAMILY: dict(sorted(noul_dropped.items())),
        },
        CHOICE_FAMILY: choice_notes,
        NOUL_FAMILY: noul_notes,
    }
