"""Tests for v2.data.m2.src_short on small synthetic fixtures."""

from __future__ import annotations

import collections
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

from training.model.data import canonical, file_sha256, validate_row
from v2.data.m2 import build, common, spec, src_short
from v2.data.sources import sgd
from v2.data.textnorm import normalize

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
FAMILIES = {item.family: item for item in src_short.FAMILIES}
POISON = "POISONSPLIT"
EXPECTED = {
    "multiwoz_intent": ("h1", "multiwoz22_train", 4000, "h1-multiwoz-intent-v1"),
    "multiwoz_slot": ("h1", "multiwoz22_train", 4000, "h1-multiwoz-slot-v1"),
    "taskmaster2_slot": ("h1", "taskmaster2_train", 4000, "h1-taskmaster2-slot-v1"),
    "dbpedia14": ("h1", "dbpedia14_train", 4200, "h1-dbpedia14-v1"),
    "winogrande_twins": ("h1", "winogrande_xl_train", 20000, "h1-winogrande-v1"),
    "csqa": ("h1", "csqa_train", 9000, "h1-csqa-v1"),
    "arc": ("h1", "arc_train", 3500, "h1-arc-v1"),
    "obqa": ("h1", "obqa_main_train", 5000, "h1-obqa-v1"),
    "aquarat": ("h1", "aquarat_train", 10000, "h1-aquarat-v1"),
    "quartz": ("h1", "quartz_train", 2700, "h1-quartz-v1"),
    "ropes": ("h1", "ropes_train", 8000, "h1-ropes-v1"),
    "scitail": ("h1", "scitail_train", 10000, "h1-scitail-v1"),
    "gsm8k_twins": ("h1", "gsm8k_train", 16000, "h1-gsm8k-v1"),
    **{
        f"mtop_intent_{language}": ("h5", "mtop_train", 2000, f"h5-mtop-{language}-v1")
        for language in ("de", "es", "fr", "hi", "th")
    },
}
NAMESPACES = {
    "multiwoz_intent": "multiwoz",
    "multiwoz_slot": "multiwoz",
    "taskmaster2_slot": "taskmaster",
    "winogrande_twins": "winogrande",
    "gsm8k_twins": "gsm8k",
    **{
        f"mtop_intent_{language}": "mtop" for language in ("de", "es", "fr", "hi", "th")
    },
}
DIGEST_SCRIPT = """
import sys
from pathlib import Path
from training.model.data import canonical
from v2.data.m2 import common, spec, src_short
dirs = spec.resolve(Path(sys.argv[1]))
built = [canonical(item.build(dirs)) for item in src_short.FAMILIES]
print(common.sha("\\n".join(built)))
"""


def _source(root: Path, name: str) -> Path:
    return root / spec.SOURCE_DIRS[name]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in items),
        encoding="utf-8",
    )


def _gold(row: dict[str, Any]) -> Any:
    return row["options"][row["label"]]["description"]


# MultiWOZ 2.2 (SGD format)

MULTIWOZ_SCHEMA = [
    {
        "service_name": "hotel",
        "description": "hotel reservations",
        "intents": [
            {"name": "find_hotel", "description": "search for a hotel to stay in"},
            {"name": "book_hotel", "description": "book a hotel  to stay in"},
        ],
        "slots": [
            {"name": "hotel-phone", "description": "phone number of the hotel"},
            {"name": "hotel-type", "description": "what is the type of the hotel"},
            {"name": "hotel-parking", "description": "whether the hotel has parking"},
            {"name": "hotel-area", "description": "area or place of the hotel"},
        ],
    },
    {
        "service_name": "police",
        "description": "police station",
        "intents": [{"name": "police", "description": "search for police station"}],
        "slots": [
            {"name": "police-phone", "description": "phone number of the police"},
            {"name": "police-address", "description": "address of the police"},
        ],
    },
]
HOTEL_REQUESTS = [
    ["hotel-phone"],
    ["hotel-type"],
    ["hotel-parking"],
    ["hotel-phone", "hotel-area"],
]


def _dialogue(dialogue_id: str, user_turns: list[list[tuple[str, str, list[str]]]]):
    turns = []
    for number, frames in enumerate(user_turns):
        turns.append(
            {
                "speaker": "USER",
                "turn_id": str(2 * number),
                "utterance": f"user  message {number} of {dialogue_id}",
                "frames": [
                    {
                        "service": service,
                        "slots": [],
                        "actions": [],
                        "state": {
                            "active_intent": intent,
                            "requested_slots": requested,
                            "slot_values": {},
                        },
                    }
                    for service, intent, requested in frames
                ],
            }
        )
        turns.append(
            {
                "speaker": "SYSTEM",
                "turn_id": str(2 * number + 1),
                "utterance": f"system reply {number}",
                "frames": [],
            }
        )
    return {"dialogue_id": dialogue_id, "services": ["hotel"], "turns": turns}


def multiwoz_dialogues() -> list[dict[str, Any]]:
    dialogues = []
    for index in range(48):
        turns = [[("hotel", "find_hotel", []), ("police", "NONE", [])]]
        turns.append(
            [
                ("hotel", "find_hotel", HOTEL_REQUESTS[index % 4]),
                ("police", "NONE", []),
            ]
        )
        if index % 4 == 3:
            turns += [[("hotel", "find_hotel", [])] for _ in range(3)]
        turns.append([("hotel", "book_hotel" if index % 2 else "find_hotel", [])])
        dialogues.append(_dialogue(f"PMUL{index:04d}.json", turns))
    for index in range(12):
        requested = [["police-phone"], ["police-address"]][index % 2]
        dialogues.append(
            _dialogue(f"SNG{index:04d}.json", [[("police", "police", requested)]])
        )
    dialogues.append(
        _dialogue(
            "BOGUS1.json",
            [[("hotel", "hotel_bogus", [])], [("hotel", "find_hotel", [])]],
        )
    )
    dialogues.append(
        _dialogue("BOGUS2.json", [[("hotel", "find_hotel", ["hotel-fax"])]])
    )
    return dialogues


def write_multiwoz(root: Path) -> None:
    base = _source(root, "multiwoz") / "data/MultiWOZ_2.2"
    dialogues = multiwoz_dialogues()
    _write_json(base / "schema.json", MULTIWOZ_SCHEMA)
    _write_json(base / "train/dialogues_001.json", dialogues[:30])
    _write_json(base / "train/dialogues_002.json", dialogues[30:])
    poison = _dialogue(POISON, [[("hotel", "book_hotel", ["hotel-phone"])]])
    _write_json(base / "dev/dialogues_001.json", [poison])


# Taskmaster-2

ORIGIN = "flight_search.origin"
DESTINATION = "flight_search.destination1"
RETURN = "flight_search.date.return"
AIRLINE = "flight_search.airline"
STOPS = "flight_search.stops"
FLIGHT_SETS = [
    [ORIGIN, DESTINATION],
    [DESTINATION, RETURN + ".accept"],
    [ORIGIN, RETURN],
    [ORIGIN, DESTINATION, AIRLINE],
]


def _utterance(index: int, speaker: str, slots: list[str]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "index": index,
        "speaker": speaker,
        "text": f"{speaker.lower()}  says {index}",
    }
    if slots:
        item["segments"] = [
            {
                "start_index": 0,
                "end_index": 4,
                "text": "says",
                "annotations": [{"name": slot} for slot in slots],
            }
        ]
    return item


def _conversation(ident: str, slots: dict[int, list[str]], speakers=None):
    speakers = speakers or {}
    return {
        "conversation_id": ident,
        "instruction_id": "synthetic",
        "utterances": [
            _utterance(
                index,
                speakers.get(index, "USER" if index % 2 == 0 else "ASSISTANT"),
                slots.get(index, []),
            )
            for index in range(14)
        ],
    }


def taskmaster_flights() -> list[dict[str, Any]]:
    conversations = []
    for index in range(30):
        present = FLIGHT_SETS[index % 4]
        slots = {2 * position: [slot] for position, slot in enumerate(present)}
        slots[1] = ["flight1_detail.fare"] + ([DESTINATION] if index % 5 == 0 else [])
        slots[12] = [STOPS]
        if index % 6 == 0:
            slots[6] = [STOPS]
        conversations.append(_conversation(f"dlg-flight-{index:03d}", slots))
    return conversations


def taskmaster_hotels() -> list[dict[str, Any]]:
    sets = [
        ["hotel_search.name.accept", "hotel_search.location"],
        ["hotel_search.location", "hotel_search.price_range"],
        ["hotel_search.price_range.reject"],
    ]
    conversations = [
        _conversation(
            f"dlg-hotel-{index:03d}",
            {2 * position: [slot] for position, slot in enumerate(sets[index % 3])},
        )
        for index in range(6)
    ]
    conversations.append(
        _conversation("dlg-hotel-assistant-only", {1: ["hotel_search.location"]})
    )
    conversations.append(
        _conversation(
            "dlg-hotel-speaker",
            {0: ["hotel_search.location"]},
            speakers={3: "SYSTEM"},
        )
    )
    conversations.append(
        _conversation("dlg-flight-000", {0: ["hotel_search.location"]})
    )
    conversations.append(
        _conversation(
            "dlg-hotel-everything",
            {
                0: ["hotel_search.name"],
                2: ["hotel_search.location"],
                4: ["hotel_search.price_range"],
            },
        )
    )
    return conversations


def write_taskmaster(root: Path) -> None:
    base = _source(root, "taskmaster") / "TM-2-2020/data"
    _write_json(base / "flights.json", taskmaster_flights())
    _write_json(base / "hotels.json", taskmaster_hotels())


# DBpedia-14


def dbpedia_items(sizes: list[int]) -> list[dict[str, Any]]:
    items = []
    for label, size in enumerate(sizes):
        for number in range(size):
            items.append(
                {
                    "label": label,
                    "title": f"Entity {label}-{number}",
                    "content": f"  Synthetic   abstract {number} of class {label}. ",
                }
            )
    return items


def write_dbpedia(root: Path, sizes: list[int]) -> None:
    base = _source(root, "dbpedia14") / "dbpedia_14"
    items = dbpedia_items(sizes)
    items[1]["title"] = items[-1]["title"] = "Shared  Title"
    items += [
        {"label": 14, "title": "Out of range", "content": "text"},
        {"label": "3", "title": "String label", "content": "text"},
        {"label": 2, "title": "", "content": "no title"},
    ]
    half = len(items) // 2
    _write_jsonl(base / "train-00000-of-00002.jsonl", items[:half])
    _write_jsonl(base / "train-00001-of-00002.jsonl", items[half:])
    _write_jsonl(
        base / "test-00000-of-00001.jsonl",
        [{"label": 0, "title": POISON, "content": POISON}],
    )


# WinoGrande

WINOGRANDE = [
    ("Ann helped Bob because _ was kind.", "Ann", "Bob", "1"),
    ("Ann helped Bob because _ was in need.", "Ann", "Bob", "2"),
    ("The cup fit in the bag because the _ was small.", "cup", "bag", "1"),
    ("The cup fit in the bag because the _ was large.", "cup", "bag", "2"),
    ("Sue beat Tim because _ trained.", "Sue", "Tim", "1"),
    ("Eve thanked Fay since _ got a gift.", "Eve", "Fay", "1"),
    ("Eve thanked Fay since _ gave a gift.", "Eve", "Fay", "2"),
    ("Gus called Hal as _ was bored.", "Gus", "Hal", "1"),
    ("Gus called Hal as _ was lonely.", "Gus", "Hal", "1"),
    ("Ian paid Joe since _ owed money.", "Ian", "Joe", "1"),
    ("Ian paid Joe since _ lent money.", "Ian", "Joe", "2"),
]


def write_winogrande(root: Path) -> None:
    items = [
        {"sentence": sentence, "option1": first, "option2": second, "answer": answer}
        for sentence, first, second, answer in WINOGRANDE
    ]
    base = _source(root, "winogrande") / "winogrande_xl"
    _write_jsonl(base / "train-00000-of-00001.jsonl", items)
    _write_jsonl(
        base / "validation-00000-of-00001.jsonl", [dict(items[0], sentence=POISON)]
    )


# Generic multiple choice


def _choices(texts: list[str], labels: str = "ABCDE") -> dict[str, list[str]]:
    return {"label": list(labels[: len(texts)]), "text": texts}


def write_csqa(root: Path) -> None:
    items = [
        {
            "id": f"csqa-{index}",
            "question": f"Where would you  find thing {index}?",
            "question_concept": "thing",
            "choices": _choices([f"place {index}-{c}" for c in range(5)]),
            "answerKey": "ABCDE"[index % 5],
        }
        for index in range(8)
    ]
    items.append(dict(items[0]))
    items.append(dict(items[1], id="csqa-bad-key", answerKey="F"))
    items.append(
        dict(
            items[2], id="csqa-dup-options", choices=_choices(["a", "b", "A", "c", "d"])
        )
    )
    base = _source(root, "csqa") / "data"
    _write_jsonl(base / "train-00000-of-00001.jsonl", items)
    _write_jsonl(
        base / "validation-00000-of-00001.jsonl", [dict(items[0], question=POISON)]
    )


def write_arc(root: Path) -> None:
    challenge = [
        {
            "id": f"Mercury_{index}",
            "question": f"Which factor explains effect {index}?",
            "choices": _choices([f"factor {index}-{c}" for c in range(4)]),
            "answerKey": "ABCD"[index % 4],
        }
        for index in range(4)
    ]
    challenge.append(
        {
            "id": "NYSEDREGENTS_1",
            "question": "Which number item is right?",
            "choices": _choices(["one", "two", "three", "four"], "1234"),
            "answerKey": "3",
        }
    )
    challenge.append(dict(challenge[0], id="Mercury_bad", answerKey="E"))
    easy = [
        {
            "id": f"MCAS_{index}",
            "question": f"What is easy fact {index}?",
            "choices": _choices([f"fact {index}-{c}" for c in range(3)]),
            "answerKey": "ABC"[index % 3],
        }
        for index in range(3)
    ]
    easy.append(dict(challenge[1]))
    base = _source(root, "arc")
    _write_jsonl(base / "ARC-Challenge/train-00000-of-00001.jsonl", challenge)
    _write_jsonl(base / "ARC-Easy/train-00000-of-00001.jsonl", easy)
    _write_jsonl(
        base / "ARC-Easy/test-00000-of-00001.jsonl", [dict(easy[0], question=POISON)]
    )


def write_obqa(root: Path) -> None:
    items = [
        {
            "id": f"7-{index}",
            "question_stem": f"The sun is responsible for thing {index}",
            "choices": _choices([f"effect {index}-{c}" for c in range(4)]),
            "answerKey": "ABCD"[(index + 1) % 4],
        }
        for index in range(5)
    ]
    _write_jsonl(_source(root, "obqa") / "main/train-00000-of-00001.jsonl", items)


def write_quartz(root: Path) -> None:
    paragraphs = [
        "More heat makes  water evaporate faster.",
        "Larger objects have more gravity.",
    ]
    items = [
        {
            "id": f"QRQA-{index}",
            "para": paragraphs[index % 2],
            "para_id": f"QRSent-{index % 2}",
            "question": f"Question {index} is _____ than before.",
            "choices": _choices(["higher", "lower"], "AB"),
            "answerKey": "AB"[index % 2],
        }
        for index in range(5)
    ]
    items.append(dict(items[0], id="QRQA-empty", para="  "))
    _write_jsonl(_source(root, "quartz") / "data/train-00000-of-00001.jsonl", items)


AQUA = [
    ("What is 2 + 2?", ["A)3", "B)4", "C)5", "D)6", "E)7"], "B"),
    (
        "How much is the tip?",
        ["A)A)$60.00", "B)B)$35.42", "C)C)$60.60", "D)D)$21.56", "E)E)$78.45"],
        "A",
    ),
    ("What is  2 + 2?", ["A) 30", "B) 40", "C) 50", "D) 60", "E) 70"], "D"),
    ("Four options only?", ["A)1", "B)2", "C)3", "D)4"], "A"),
    ("Duplicate texts?", ["A)9", "B)10", "C)9", "D)12", "E)10"], "A"),
    ("Wrong letter?", ["A)1", "B)2", "C)3", "D)4", "E)5"], "F"),
    ("Missing prefix?", ["A)1", "2", "C)3", "D)4", "E)5"], "A"),
    (
        "Speed of the train?",
        ["a)10 km", "b)20 km", "c)30 km", "d)40 km", "e)50 km"],
        "c",
    ),
]


def write_aquarat(root: Path) -> None:
    items = [
        {"question": question, "options": options, "correct": correct, "rationale": "x"}
        for question, options, correct in AQUA
    ]
    items.append(dict(items[0]))
    _write_jsonl(_source(root, "aquarat") / "raw/train-00000-of-00001.jsonl", items)


ROPES = [
    ("Will cell Z have a larger or smaller surface area?", ["smaller"]),
    ("Will cell X have a larger or smaller surface area?", ["larger"]),
    ("Which city had more pollution, city A or city B?", ["city B"]),
    ("Does Cell A or Cell B have more folds?", ["Cell A"]),
    ("Which cup has a higher concentration of sugar?", ["cup B"]),
    ("Is it A or B or C?", ["A"]),
    ("Is the air in the city A or the city B cleaner?", ["city A"]),
    ("Would the plant grow faster or more slowly?", ["more slowly"]),
    ("Is city A or city B warmer?", ["city A", "city B"]),
]


def write_ropes(root: Path) -> None:
    items = [
        {
            "id": str(1000 + index),
            "background": "Background  text about heat.",
            "situation": f"Situation {index // 3} with city A and city B.",
            "question": question,
            "answers": {"text": answers},
        }
        for index, (question, answers) in enumerate(ROPES)
    ]
    items.append(dict(items[0], id="2000", background=" "))
    _write_jsonl(
        _source(root, "ropes") / "plain_text/train-00000-of-00001.jsonl", items
    )


# SciTail

SCITAIL = [
    ("Premise one says the earth spins.", "The earth rotates.", "entailment"),
    ("Premise two says the earth turns daily.", "The earth rotates.", "entailment"),
    ("Premise three says the earth spins fast.", "The earth rotates.", "entails"),
    ("Premise extra is about the earth.", "The earth rotates.", "entailment"),
    ("Premise four is about mars.", "The earth rotates.", "neutral"),
    ("Premise five is about venus.", "The earth rotates.", "neutral"),
    ("Premise six is about jupiter.", "The earth rotates.", "neutral"),
    ("Leaves are green.", "Plants photosynthesize.", "entailment"),
    ("Roots grow down.", "Plants photosynthesize.", "entailment"),
    ("Premise four is about mars.", "Mars is red.", "entailment"),
    ("Mars has two moons.", "Mars is red.", "neutral"),
    ("Water boils at 100 degrees.", "Water is wet.", "contradiction"),
    ("Premise nine is  about ice.", "Ice is cold.", "entailment"),
    ("Premise nine is about ice.", "Ice is cold.", "entailment"),
    ("Snow is white.", "Ice is cold.", "neutral"),
    ("Fire is hot.", "Fire burns.", "entailment"),
    ("Fire is hot.", "Fire burns.", "neutral"),
]


def write_scitail(root: Path) -> None:
    items = [
        {"sentence1": premise, "sentence2": hypothesis, "gold_label": label}
        for premise, hypothesis, label in SCITAIL
    ]
    _write_jsonl(
        _source(root, "scitail") / "snli_format/train-00000-of-00001.jsonl", items
    )


# GSM8K

GSM8K = [
    ("12", "Problem zero?"),
    ("34", "Problem one?"),
    ("56", "Problem two?"),
    ("7", "Problem three?"),
    ("8", "Problem four?"),
    ("1,200", "Problem five?"),
    ("-3", "Problem six?"),
    ("9", "Problem seven?"),
    ("4,500", "Problem eight?"),
    ("2.5", "Problem nine?"),
    (None, "Problem ten?"),
    ("12345", "Problem eleven?"),
]


def write_gsm8k(root: Path) -> None:
    items = [
        {
            "question": question,
            "answer": "Work <<1+1=2>>.\n" + (f"#### {answer}" if answer else "done"),
        }
        for answer, question in GSM8K
    ]
    _write_jsonl(_source(root, "gsm8k") / "main/train-00000-of-00001.jsonl", items)


# MTOP


def mtop_line(ident: str, intent: str, utterance: str, domain: str, locale: str) -> str:
    tokens = json.dumps({"tokens": utterance.split(), "tokenSpans": []})
    return "\t".join(
        [ident, intent, "", utterance, domain, locale, f"[{intent} ]", tokens]
    )


def mtop_members() -> dict[str, str]:
    english = [
        ("e1", "IN:CREATE_ALARM", "alarm"),
        ("e2", "IN:CREATE_ALARM", "alarm"),
        ("e3", "IN:GET_ALARM", "alarm"),
        ("e4", "IN:DELETE_ALARM", "alarm"),
        ("e5", "IN:PLAY_MUSIC", "music"),
        ("e6", "IN:PAUSE_MUSIC", "music"),
        ("e7", "IN:SKIP_TRACK_MUSIC", "music"),
        ("e8", "IN:GET_WEATHER", "weather"),
        ("e9", "IN:GET_SUNRISE", "weather"),
    ]
    members = {
        "mtop/en/train.txt": "\n".join(
            mtop_line(ident, intent, f"english {ident}", domain, "en_XX")
            for ident, intent, domain in english
        )
        + "\n"
    }
    german = [(f"a{n:02d}", "IN:CREATE_ALARM", "alarm") for n in range(9)]
    german += [(f"g{n:02d}", "IN:GET_ALARM", "alarm") for n in range(5)]
    german += [(f"d{n:02d}", "IN:DELETE_ALARM", "alarm") for n in range(4)]
    german += [(f"m{n:02d}", "IN:PLAY_MUSIC", "music") for n in range(3)]
    german += [(f"w{n:02d}", "IN:GET_WEATHER", "weather") for n in range(2)]
    german += [
        ("s00", "IN:SNOOZE_ALARM", "alarm"),
        ("r00", "IN:CREATE_REMINDER", "reminder"),
    ]
    lines = [mtop_line(i, t, f"Wecker  {i} bitte", d, "de_XX") for i, t, d in german]
    lines += [
        mtop_line("a00", "IN:CREATE_ALARM", "doppelt", "alarm", "de_XX"),
        "too\tfew\tcolumns",
        mtop_line("x00", "IN:GET_ALARM", "   ", "alarm", "de_XX"),
    ]
    members["mtop/de/train.txt"] = "\r\n".join(lines) + "\r\n"
    members["mtop/de/eval.txt"] = mtop_line(
        "z00", "IN:GET_ALARM", POISON, "alarm", "de_XX"
    )
    members["mtop/de/test.txt"] = mtop_line(
        "z01", "IN:GET_ALARM", POISON, "alarm", "de_XX"
    )
    french = [(f"a{n:02d}", "IN:CREATE_ALARM") for n in range(5)]
    french += [(f"g{n:02d}", "IN:GET_ALARM") for n in range(5)]
    french += [(f"d{n:02d}", "IN:DELETE_ALARM") for n in range(5)]
    members["mtop/fr/train.txt"] = "\n".join(
        mtop_line(i, t, f"réveil {i}", "alarm", "fr_XX") for i, t in french
    )
    for language, word in (("es", "alarma"), ("hi", "अलार्म"), ("th", "นาฬิกาปลุก")):
        rows = [(f"g{n:02d}", "IN:GET_ALARM") for n in range(2)]
        rows += [(f"d{n:02d}", "IN:DELETE_ALARM") for n in range(2)]
        rows += [(f"a{n:02d}", "IN:CREATE_ALARM") for n in range(2)]
        members[f"mtop/{language}/train.txt"] = "\n".join(
            mtop_line(i, t, f"{word} {i}", "alarm", f"{language}_XX") for i, t in rows
        )
    members["mtop/LICENSE"] = "Synthetic licence text.\n"
    return members


def write_mtop(root: Path) -> None:
    base = _source(root, "mtop")
    base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base / "mtop.zip", "w") as archive:
        for name, text in mtop_members().items():
            archive.writestr(name, text.encode("utf-8"))


def write_all(root: Path) -> None:
    write_multiwoz(root)
    write_taskmaster(root)
    write_dbpedia(root, [5] * 14)
    write_winogrande(root)
    write_csqa(root)
    write_arc(root)
    write_obqa(root)
    write_quartz(root)
    write_aquarat(root)
    write_ropes(root)
    write_scitail(root)
    write_gsm8k(root)
    write_mtop(root)


_TMP: tempfile.TemporaryDirectory[str] | None = None
ROOT = Path()
DIRS: dict[str, Path] = {}
BUILT: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}


def setUpModule() -> None:
    global _TMP, ROOT
    _TMP = tempfile.TemporaryDirectory()
    ROOT = Path(_TMP.name)
    write_all(ROOT)
    DIRS.update(spec.resolve(ROOT))
    BUILT.update({name: item.build(DIRS) for name, item in FAMILIES.items()})


def tearDownModule() -> None:
    if _TMP is not None:
        _TMP.cleanup()


def rows_of(family: str) -> list[dict[str, Any]]:
    return BUILT[family][0]


def report_of(family: str) -> dict[str, Any]:
    return BUILT[family][1]


class RegistryTest(unittest.TestCase):
    def test_families_match_the_assignment(self) -> None:
        self.assertEqual(
            {
                item.family: (item.arm, item.source, item.cap_rows, item.seed)
                for item in src_short.FAMILIES
            },
            EXPECTED,
        )
        self.assertEqual(len(src_short.FAMILIES), len(EXPECTED))

    def test_every_family_builds_valid_train_rows(self) -> None:
        for family, (_, source, _, _) in EXPECTED.items():
            with self.subTest(family=family):
                rows, report = BUILT[family]
                self.assertTrue(rows)
                namespace = NAMESPACES.get(family, family)
                language = (
                    family.rsplit("_", 1)[1] if family.startswith("mtop") else "en"
                )
                for row in rows:
                    validate_row(row, "train")
                    self.assertEqual((row["family"], row["source"]), (family, source))
                    self.assertEqual(row["language"], language)
                    self.assertEqual(row["render_template"], f"m2/{family}/v1")
                    self.assertTrue(row["group_id"].startswith(f"m2:{namespace}:"))
                    self.assertNotIn(POISON, canonical(row))
                ids = [row["id"] for row in rows]
                self.assertEqual(len(set(ids)), len(ids))
                self.assertEqual(report["candidates"], len(rows))
                self.assertIsInstance(report["items"], int)
                self.assertTrue(all(count > 0 for count in report["drops"].values()))
                json.dumps(report)

    def test_inputs_are_train_files_with_their_hashes(self) -> None:
        for item in src_short.FAMILIES:
            with self.subTest(family=item.family):
                inputs = report_of(item.family)["inputs"]
                source_dir = DIRS[item.build.keywords["job"].name]
                self.assertTrue(inputs)
                for relative, digest in inputs.items():
                    self.assertNotRegex(relative, "dev|test|validation|eval")
                    self.assertEqual(file_sha256(source_dir / relative), digest)

    def test_missing_sources_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = spec.resolve(Path(tmp))
            for item in src_short.FAMILIES:
                with self.subTest(family=item.family):
                    with self.assertRaises(FileNotFoundError):
                        item.build(empty)

    def test_orchestrator_runs_both_arms(self) -> None:
        for arm in ("h1", "h5"):
            with self.subTest(arm=arm):
                rows, report = build.build(arm, DIRS, workers=1, modules=("src_short",))
                families = {name for name, item in EXPECTED.items() if item[0] == arm}
                self.assertEqual(set(report["families"]), families)
                self.assertEqual(
                    len(rows),
                    sum(report["families"][name]["after_cap"] for name in families),
                )
                for row in rows:
                    validate_row(row, "train")

    def test_builds_are_deterministic_across_hash_seeds(self) -> None:
        again = {name: item.build(DIRS) for name, item in FAMILIES.items()}
        for name in FAMILIES:
            self.assertEqual(canonical(again[name]), canonical(BUILT[name]))
        expected = common.sha(
            "\n".join(canonical(BUILT[item.family]) for item in src_short.FAMILIES)
        )
        for seed in ("1", "4242"):
            result = subprocess.run(
                [sys.executable, "-c", DIGEST_SCRIPT, str(ROOT)],
                cwd=PACKAGE_ROOT,
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), expected)


class MultiwozTest(unittest.TestCase):
    def test_choice_is_intent_balanced_over_services_with_two_intents(self) -> None:
        rows, report = rows_of("multiwoz_intent"), report_of("multiwoz_intent")
        self.assertEqual(report["services_eligible"], ["hotel"])
        self.assertEqual(report["services_ineligible"], ["police"])
        intents = collections.Counter(row["audit_metadata"]["intent"] for row in rows)
        self.assertEqual(set(intents), {"find_hotel", "book_hotel"})
        self.assertEqual(len(set(intents.values())), 1)
        self.assertEqual(report["rows_per_intent"], {"hotel": intents["find_hotel"]})
        self.assertEqual(report["drops"]["activation_intent_not_in_schema"], 1)
        self.assertGreater(report["drops"]["intent_balance"], 0)
        descriptions = {
            "find_hotel": "search for a hotel to stay in",
            "book_hotel": "book a hotel to stay in",
        }
        for row in rows:
            audit = row["audit_metadata"]
            self.assertEqual(_gold(row), descriptions[audit["intent"]])
            self.assertEqual(len(row["options"]), 2)
            self.assertEqual(row["instructions"], sgd.CHOICE_INSTRUCTIONS)
            self.assertEqual(
                row["group_id"],
                common.group_id("multiwoz", audit["source_local_id"].split(":")[0]),
            )
            lines = row["state"].split("\n")
            self.assertLessEqual(len(lines), sgd.WINDOW)
            self.assertTrue(lines[-1].startswith("User: user message"))

    def test_window_keeps_the_last_eight_turns(self) -> None:
        long = [
            row
            for row in rows_of("multiwoz_intent")
            if row["audit_metadata"]["turn_index"] == 10
        ]
        self.assertTrue(long)
        for row in long:
            lines = row["state"].split("\n")
            self.assertEqual(len(lines), 8)
            self.assertTrue(lines[0].startswith("System: system reply 1"))

    def test_slot_noul_is_balanced_per_service_and_asked_slot(self) -> None:
        rows, report = rows_of("multiwoz_slot"), report_of("multiwoz_slot")
        strata = collections.defaultdict(collections.Counter)
        for row in rows:
            audit = row["audit_metadata"]
            strata[(audit["service"], audit["asked_slot"])][row["label"]] += 1
            if row["label"]:
                self.assertIn(audit["asked_slot"], audit["requested_slots"])
            else:
                self.assertNotIn(audit["asked_slot"], audit["requested_slots"])
        self.assertTrue(strata)
        for counts in strata.values():
            self.assertEqual(counts[0], counts[1])
        services = {service for service, _ in strata}
        self.assertEqual(services, {"hotel", "police"})
        self.assertEqual(report["drops"]["requested_slot_not_in_schema"], 1)
        self.assertEqual(report["drops"]["dialogue_without_request"], 2)
        self.assertEqual(report["requestable_slots"], {"hotel": 4, "police": 2})

    def test_slot_questions_read_naturally(self) -> None:
        self.assertEqual(
            src_short.slot_question("what is the type of the hotel"),
            "In the latest message, does the user ask what is the type of the hotel?",
        )
        self.assertEqual(
            src_short.slot_question("whether the hotel has parking"),
            "In the latest message, does the user ask whether the hotel has parking?",
        )
        self.assertEqual(
            src_short.slot_question("phone number of the hotel"),
            "In the latest message, does the user ask for "
            "the phone number of the hotel?",
        )
        self.assertEqual(
            src_short.slot_question("How many  tickets you need"),
            "In the latest message, does the user ask how many tickets you need?",
        )


class TaskmasterTest(unittest.TestCase):
    def conversations(self) -> dict[str, dict[str, Any]]:
        return {
            item["conversation_id"]: item
            for item in taskmaster_flights() + taskmaster_hotels()[:6]
        }

    def test_each_conversation_gives_one_true_and_one_false_row(self) -> None:
        rows, report = rows_of("taskmaster2_slot"), report_of("taskmaster2_slot")
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row["group_id"]].append(row)
            ident = row["audit_metadata"]["source_local_id"].rsplit(":", 1)[0]
            self.assertEqual(row["group_id"], common.group_id("taskmaster", ident))
        for members in groups.values():
            self.assertEqual(sorted(row["label"] for row in members), [0, 1])
            self.assertEqual(members[0]["state"], members[1]["state"])
        drops = dict(report["drops"])
        self.assertEqual(len(groups) + drops.pop("slot_balance", 0), 36)
        self.assertGreaterEqual(len(groups), 24)
        self.assertEqual(
            drops,
            {
                "duplicate_conversation": 1,
                "no_absent_slot": 1,
                "no_user_slot": 1,
                "unknown_speaker": 1,
            },
        )
        by_domain = collections.defaultdict(collections.Counter)
        for row in rows:
            by_domain[row["audit_metadata"]["domain"]][row["label"]] += 1
        self.assertEqual(set(by_domain), {"flights", "hotels"})
        for counts in by_domain.values():
            self.assertEqual(counts[0], counts[1])
        for domain, counts in by_domain.items():
            self.assertEqual(report["domains"][domain]["rows"], sum(counts.values()))

    def test_true_slots_are_user_slots_and_false_slots_unmentioned(self) -> None:
        source = self.conversations()
        for row in rows_of("taskmaster2_slot"):
            audit = row["audit_metadata"]
            ident = audit["source_local_id"].rsplit(":", 1)[0]
            shown = source[ident]["utterances"][:12]
            names = [
                (
                    utterance["speaker"],
                    src_short.TASKMASTER_SUFFIX.sub("", annotation["name"]),
                )
                for utterance in shown
                for segment in utterance.get("segments", [])
                for annotation in segment["annotations"]
            ]
            user = {name for speaker, name in names if speaker == "USER"}
            mentioned = {name for _, name in names}
            self.assertEqual(len(row["state"].split("\n")), 12)
            self.assertNotIn(".accept", audit["slot"])
            self.assertNotIn(".reject", audit["slot"])
            self.assertEqual(
                row["instructions"],
                "In this conversation, does the user specify the "
                f"{src_short.readable_slot(audit['slot'])}?",
            )
            if row["label"]:
                self.assertIn(audit["slot"], user)
            else:
                self.assertNotIn(audit["slot"], mentioned)
            self.assertNotEqual(audit["slot"], "flight1_detail.fare")

    def test_slot_names_carry_no_label(self) -> None:
        counts = collections.defaultdict(collections.Counter)
        for row in rows_of("taskmaster2_slot"):
            audit = row["audit_metadata"]
            counts[(audit["domain"], audit["slot"])][row["label"]] += 1
        for slot in (ORIGIN, DESTINATION, RETURN, STOPS):
            self.assertGreater(counts[("flights", slot)][0], 0, slot)
            self.assertGreater(counts[("flights", slot)][1], 0, slot)
        for key, labels in counts.items():
            self.assertLessEqual(abs(labels[1] - labels[0]), 1, key)
        domains = report_of("taskmaster2_slot")["domains"]
        self.assertEqual(domains["flights"]["inventory"], 5)
        self.assertTrue(all(item["slot_gap_max"] <= 1 for item in domains.values()))

    def test_slot_gap_bound_drops_conversations_that_would_exceed_it(self) -> None:
        gap = collections.Counter({"always": 1, "rare": -1})
        rng = random.Random("seed")
        self.assertIsNone(src_short._least(["always"], gap, 1, rng))
        self.assertIsNone(src_short._least(["rare"], gap, -1, rng))
        self.assertEqual(src_short._least(["always", "rare"], gap, 1, rng), "rare")
        self.assertEqual(src_short._least(["always", "other"], gap, -1, rng), "always")

    def test_readable_slot(self) -> None:
        self.assertEqual(
            src_short.readable_slot("flight_search.origin"), "flight search origin"
        )
        self.assertEqual(
            src_short.readable_slot("flight_search.date.depart_origin"),
            "flight search date depart origin",
        )


class DbpediaTest(unittest.TestCase):
    def test_equal_rows_per_class_and_title_groups(self) -> None:
        rows, report = rows_of("dbpedia14"), report_of("dbpedia14")
        classes = collections.Counter(_gold(row) for row in rows)
        self.assertEqual(classes, {name: 5 for name in src_short.DBPEDIA_CLASSES})
        self.assertEqual(report["per_class"], 5)
        self.assertEqual(report["drops"], {"empty_text": 1, "unknown_label": 2})
        for row in rows:
            self.assertEqual(
                sorted(option["description"] for option in row["options"]),
                sorted(src_short.DBPEDIA_CLASSES),
            )
            self.assertEqual(row["instructions"], src_short.DBPEDIA_INSTRUCTIONS)
            title, content = row["state"].split("\n\n")
            self.assertEqual(
                row["group_id"], common.group_id("dbpedia14", normalize(title))
            )
            self.assertTrue(content.startswith("Synthetic abstract"))
        shared = [row for row in rows if row["state"].startswith("Shared Title")]
        self.assertEqual(len(shared), 2)
        self.assertEqual(shared[0]["group_id"], shared[1]["group_id"])

    def test_quota_takes_the_first_rows_in_seed_hash_order(self) -> None:
        with mock.patch.object(src_short, "DBPEDIA_PER_CLASS", 3):
            rows, report = FAMILIES["dbpedia14"].build(DIRS)
        self.assertEqual(report["drops"]["class_quota"], 28)
        by_class = collections.defaultdict(list)
        for row in rows:
            by_class[_gold(row)].append(row["audit_metadata"]["source_local_id"])
        for label, name in enumerate(src_short.DBPEDIA_CLASSES):
            local = [f"db-{5 * label + number}" for number in range(5)]
            expected = sorted(
                local, key=lambda ident: common.sha(f"h1-dbpedia14-v1:{ident}")
            )[:3]
            self.assertEqual(sorted(by_class[name]), sorted(expected))

    def test_full_quota_gives_4200_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dbpedia(root, [300 + label % 2 for label in range(14)])
            rows, report = FAMILIES["dbpedia14"].build(spec.resolve(root))
        self.assertEqual(len(rows), 4200)
        self.assertEqual(
            collections.Counter(_gold(row) for row in rows),
            {name: 300 for name in src_short.DBPEDIA_CLASSES},
        )
        self.assertEqual(report["per_class"], 300)


class WinograndeTest(unittest.TestCase):
    def test_twins_share_options_flip_gold_and_keep_source_order(self) -> None:
        rows, report = rows_of("winogrande_twins"), report_of("winogrande_twins")
        self.assertEqual(report["pairs"], 4)
        self.assertEqual(report["drops"], {"unpaired_row": 3})
        self.assertEqual(
            sorted(row["audit_metadata"]["source_local_id"] for row in rows),
            sorted(f"wg-{index}" for index in (0, 1, 2, 3, 5, 6, 9, 10)),
        )
        by_group = collections.defaultdict(list)
        for row in rows:
            by_group[row["group_id"]].append(row)
            index = int(row["audit_metadata"]["source_local_id"][3:])
            sentence, first, second, answer = WINOGRANDE[index]
            self.assertEqual(
                [o["description"] for o in row["options"]], [first, second]
            )
            self.assertEqual(row["label"], int(answer) - 1)
            self.assertEqual(row["state"], sentence)
            self.assertEqual(row["instructions"], src_short.WINOGRANDE_INSTRUCTIONS)
        self.assertEqual(len(by_group), 4)
        for members in by_group.values():
            self.assertEqual(sorted(row["label"] for row in members), [0, 1])
            self.assertEqual(members[0]["options"], members[1]["options"])
            pair = members[0]["audit_metadata"]["pair"]
            self.assertEqual(
                members[0]["group_id"], common.group_id("winogrande", pair)
            )
        self.assertEqual(
            collections.Counter(row["label"] for row in rows), {0: 4, 1: 4}
        )


class MultipleChoiceTest(unittest.TestCase):
    def assert_rotated_gold(self, family: str, answers: dict[str, str]) -> None:
        rows = rows_of(family)
        self.assertEqual(
            {row["audit_metadata"]["source_local_id"]: _gold(row) for row in rows},
            answers,
        )
        for row in rows:
            self.assertEqual(row["instructions"], src_short.CHOICE_INSTRUCTIONS)

    def test_csqa(self) -> None:
        self.assert_rotated_gold(
            "csqa", {f"csqa-{i}": f"place {i}-{i % 5}" for i in range(8)}
        )
        self.assertEqual(
            report_of("csqa")["drops"],
            {"answer_not_in_labels": 1, "duplicate_id": 1, "duplicate_options": 1},
        )
        row = rows_of("csqa")[0]
        self.assertEqual(row["state"], "Where would you find thing 0?")
        self.assertEqual(row["group_id"], common.group_id("csqa", "csqa-0"))

    def test_arc_maps_letter_and_digit_keys(self) -> None:
        answers = {f"Mercury_{i}": f"factor {i}-{i % 4}" for i in range(4)}
        answers["NYSEDREGENTS_1"] = "three"
        answers.update({f"MCAS_{i}": f"fact {i}-{i % 3}" for i in range(3)})
        self.assert_rotated_gold("arc", answers)
        self.assertEqual(
            report_of("arc")["drops"], {"answer_not_in_labels": 1, "duplicate_id": 1}
        )
        subsets = {row["audit_metadata"]["subset"] for row in rows_of("arc")}
        self.assertEqual(subsets, {"ARC-Challenge", "ARC-Easy"})

    def test_obqa_uses_the_question_stem(self) -> None:
        self.assert_rotated_gold(
            "obqa", {f"7-{i}": f"effect {i}-{(i + 1) % 4}" for i in range(5)}
        )
        self.assertTrue(
            rows_of("obqa")[0]["state"].startswith("The sun is responsible")
        )

    def test_quartz_state_and_paragraph_groups(self) -> None:
        self.assert_rotated_gold(
            "quartz", {f"QRQA-{i}": ["higher", "lower"][i % 2] for i in range(5)}
        )
        self.assertEqual(report_of("quartz")["drops"], {"empty_context": 1})
        groups = collections.defaultdict(set)
        for row in rows_of("quartz"):
            paragraph, question = row["state"].split("\n\n")
            self.assertTrue(question.startswith("Question"))
            groups[paragraph].add(row["group_id"])
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(ids) == 1 for ids in groups.values()))

    def test_aquarat_strips_prefixes_and_groups_questions(self) -> None:
        rows, report = rows_of("aquarat"), report_of("aquarat")
        self.assertEqual(
            sorted((row["state"], _gold(row)) for row in rows),
            [
                ("How much is the tip?", "$60.00"),
                ("Speed of the train?", "30 km"),
                ("What is 2 + 2?", "4"),
                ("What is 2 + 2?", "60"),
            ],
        )
        tip = next(row for row in rows if row["state"] == "How much is the tip?")
        self.assertEqual(
            sorted(option["description"] for option in tip["options"]),
            sorted(["$60.00", "$35.42", "$60.60", "$21.56", "$78.45"]),
        )
        self.assertEqual(
            report["drops"],
            {
                "answer_not_in_labels": 1,
                "duplicate_item": 1,
                "duplicate_options": 1,
                "malformed_options": 2,
            },
        )
        same = [row for row in rows if normalize(row["state"]) == "what is 2 + 2?"]
        self.assertEqual(len(same), 2)
        self.assertEqual(same[0]["group_id"], same[1]["group_id"])
        self.assertEqual({_gold(row) for row in same}, {"4", "60"})

    def test_aqua_options(self) -> None:
        self.assertEqual(
            src_short.aqua_options(["A)A)$1", "B) 2", "C)3", "D)4", "E)E) 5"]),
            ["$1", "2", "3", "4", "5"],
        )
        self.assertIsNone(src_short.aqua_options(["B)1", "A)2", "C)3", "D)4", "E)5"]))
        self.assertIsNone(src_short.aqua_options(["A)1", "B)2"]))


class RopesTest(unittest.TestCase):
    def test_or_candidates(self) -> None:
        cases = [
            (
                "Will cell Z have a larger or smaller surface area?",
                "smaller",
                (["larger", "smaller"], 1),
            ),
            (
                "Does Cell A or Cell B have more folds?",
                "cell a",
                (["Cell A", "Cell B"], 0),
            ),
            (
                "Which city had more pollution, city A or city B?",
                "city B",
                (["city A", "city B"], 1),
            ),
            (
                "Which one is heavier: the bowling ball or the feather?",
                "Bowling ball",
                (["the bowling ball", "the feather"], 0),
            ),
            (
                "Will the ice melt faster in the sun or the shade?",
                "sun",
                (["the sun", "the shade"], 0),
            ),
            ("Which substance, A or B, is more acidic?", "B", (["A", "B"], 1)),
            ("Is Tom or Bob taller?", "Tom", (["Tom", "Bob"], 0)),
            ("Yesterday, did the dog run or walk?", "walk", (["run", "walk"], 1)),
            ("Which cup has more sugar?", "cup B", "no_or"),
            ("Is it A or B or C?", "A", "multiple_or"),
            (
                "Is the air in the city A or the city B cleaner?",
                "city A",
                "ambiguous_candidates",
            ),
            (
                "Would the plant grow faster or more slowly?",
                "more slowly",
                "gold_not_a_candidate",
            ),
            ("Did the dog bark or not?", "yes", "gold_not_a_candidate"),
            ("Did John or his brother eat more?", "John", "gold_not_a_candidate"),
            ("Is it red and/or blue?", "red", "or_not_between_words"),
        ]
        for question, answer, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(src_short.or_candidates(question, answer), expected)

    def test_rows_use_question_spans_and_situation_groups(self) -> None:
        rows, report = rows_of("ropes"), report_of("ropes")
        self.assertEqual(
            {row["instructions"]: _gold(row) for row in rows},
            {
                ROPES[0][0]: "smaller",
                ROPES[1][0]: "larger",
                ROPES[2][0]: "city B",
                ROPES[3][0]: "Cell A",
            },
        )
        self.assertEqual(
            report["drops"],
            {
                "ambiguous_candidates": 1,
                "answer_count": 1,
                "empty_context": 1,
                "gold_not_a_candidate": 1,
                "multiple_or": 1,
                "no_or": 1,
            },
        )
        for row in rows:
            background, situation = row["state"].split("\n\n")
            self.assertEqual(background, "Background text about heat.")
            self.assertEqual(
                row["group_id"],
                common.group_id(
                    "ropes", f"{normalize(background)}\n\n{normalize(situation)}"
                ),
            )
        self.assertEqual(len({row["group_id"] for row in rows}), 2)


class ScitailTest(unittest.TestCase):
    def test_balanced_within_every_hypothesis(self) -> None:
        rows, report = rows_of("scitail"), report_of("scitail")
        per_hypothesis = collections.defaultdict(collections.Counter)
        for row in rows:
            self.assertEqual(set(row["state"]), {"premise", "hypothesis"})
            self.assertEqual(row["instructions"], src_short.SCITAIL_INSTRUCTIONS)
            per_hypothesis[row["state"]["hypothesis"]][row["label"]] += 1
        self.assertEqual(
            dict(per_hypothesis),
            {
                "The earth rotates.": {0: 3, 1: 3},
                "Mars is red.": {0: 1, 1: 1},
                "Ice is cold.": {0: 1, 1: 1},
            },
        )
        self.assertEqual(
            report["drops"],
            {
                "conflicting_duplicate": 2,
                "duplicate_pair": 1,
                "hypothesis_balance": 3,
                "unknown_label": 1,
            },
        )
        self.assertEqual(report["hypotheses"], 3)
        self.assertEqual(
            collections.Counter(row["label"] for row in rows), {0: 5, 1: 5}
        )

    def test_groups_join_shared_premises_and_hypotheses(self) -> None:
        rows = rows_of("scitail")
        components = {
            "earth_and_mars": [
                row
                for row in rows
                if row["state"]["hypothesis"] in ("The earth rotates.", "Mars is red.")
            ],
            "ice": [
                row for row in rows if row["state"]["hypothesis"] == "Ice is cold."
            ],
        }
        self.assertEqual(len(components["earth_and_mars"]), 8)
        for members in components.values():
            expected = min(normalize(row["state"]["premise"]) for row in members)
            self.assertEqual(
                {row["group_id"] for row in members},
                {common.group_id("scitail", expected)},
            )
        self.assertEqual(len({row["group_id"] for row in rows}), 2)
        self.assertEqual(report_of("scitail")["largest_component"], 8)

    def test_whole_group_caps_stay_exactly_balanced(self) -> None:
        for cap in (2, 8, 9):
            capped = common.cap_groups(rows_of("scitail"), cap, "h1-scitail-v1")
            self.assertTrue(capped)
            labels = collections.Counter(row["label"] for row in capped)
            self.assertEqual(labels[0], labels[1])


class Gsm8kTest(unittest.TestCase):
    def test_numeric_twins(self) -> None:
        rows, report = rows_of("gsm8k_twins"), report_of("gsm8k_twins")
        self.assertEqual(len(rows), 18)
        self.assertEqual(
            report["drops"],
            {"no_final_answer": 1, "non_integer_answer": 1, "singleton_bucket": 1},
        )
        truth, false = {}, collections.Counter()
        for row in rows:
            value = (
                row["instructions"]
                .removeprefix("Is the final answer to this problem ")
                .removesuffix("?")
            )
            if row["label"]:
                truth[row["group_id"]] = value
            else:
                false[value] += 1
        self.assertEqual(
            sorted(truth.values()),
            sorted(["12", "34", "56", "7", "8", "1200", "-3", "9", "4500"]),
        )
        self.assertEqual(false, collections.Counter(truth.values()))
        by_group = collections.defaultdict(list)
        for row in rows:
            by_group[row["group_id"]].append(row["label"])
        self.assertTrue(all(sorted(labels) == [0, 1] for labels in by_group.values()))

    def test_final_answers(self) -> None:
        self.assertEqual(src_short.gsm8k_answer("x\n#### 1,234"), ("1234", None))
        self.assertEqual(src_short.gsm8k_answer("#### -07"), ("-7", None))
        self.assertEqual(
            src_short.gsm8k_answer("#### 2.5"), (None, "non_integer_answer")
        )
        self.assertEqual(src_short.gsm8k_answer("no marker"), (None, "no_final_answer"))


class MtopTest(unittest.TestCase):
    def test_options_are_the_english_domain_intents(self) -> None:
        expected = sorted(["create alarm", "delete alarm", "get alarm"])
        for language in ("de", "es", "fr", "hi", "th"):
            for row in rows_of(f"mtop_intent_{language}"):
                self.assertEqual(
                    sorted(option["description"] for option in row["options"]), expected
                )
                self.assertEqual(row["instructions"], src_short.MTOP_INSTRUCTIONS)
                self.assertEqual(
                    _gold(row),
                    src_short.intent_description(row["audit_metadata"]["intent"]),
                )

    def test_german_balance_and_drops(self) -> None:
        rows, report = rows_of("mtop_intent_de"), report_of("mtop_intent_de")
        self.assertEqual(
            collections.Counter(row["audit_metadata"]["intent"] for row in rows),
            {"IN:CREATE_ALARM": 4, "IN:GET_ALARM": 4, "IN:DELETE_ALARM": 4},
        )
        self.assertEqual(
            report["drops"],
            {
                "domain_not_in_english": 1,
                "domain_too_few_intents": 2,
                "duplicate_id": 1,
                "empty_utterance": 1,
                "intent_balance": 6,
                "intent_not_in_english_domain": 1,
                "malformed_line": 1,
                "single_intent_domain": 3,
            },
        )
        self.assertEqual(report["domains"]["alarm"]["before"], [9, 4, 5])
        self.assertEqual(report["domains"]["alarm"]["limit"], 4)
        self.assertEqual(report["licence_members"].keys(), {"mtop/LICENSE"})
        self.assertEqual(
            set(report["members"]), {"mtop/en/train.txt", "mtop/de/train.txt"}
        )
        for row in rows:
            ident = row["audit_metadata"]["source_local_id"].removeprefix("de:")
            self.assertEqual(row["state"], f"Wecker {ident} bitte")

    def test_limit_is_six_fifths_of_the_rarest_present_intent(self) -> None:
        rows = rows_of("mtop_intent_fr")
        self.assertEqual(len(rows), 15)
        self.assertEqual(report_of("mtop_intent_fr")["domains"]["alarm"]["limit"], 6)

    def test_parallel_ids_share_groups_across_languages(self) -> None:
        german = {
            row["audit_metadata"]["source_local_id"][3:]: row
            for row in rows_of("mtop_intent_de")
        }
        french = {
            row["audit_metadata"]["source_local_id"][3:]: row
            for row in rows_of("mtop_intent_fr")
        }
        shared = german.keys() & french.keys()
        self.assertGreaterEqual(len(shared), 4)
        for ident in shared:
            self.assertEqual(german[ident]["group_id"], french[ident]["group_id"])
            self.assertEqual(german[ident]["group_id"], common.group_id("mtop", ident))
        thai = rows_of("mtop_intent_th")
        self.assertTrue(all(row["state"].startswith("นาฬิกาปลุก") for row in thai))


if __name__ == "__main__":
    unittest.main()
