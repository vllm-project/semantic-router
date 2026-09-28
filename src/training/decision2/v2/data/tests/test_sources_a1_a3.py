from __future__ import annotations

import collections
import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from training.model.data import validate_row

from v2.data import build_a1_a3
from v2.data.sources import abcd, musique, qasc, sgd
from v2.data.sources.common import choice_options, rotate


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _conversation(convo_id: int, flow: str, subflow: str, note: str = "") -> dict:
    return {
        "convo_id": convo_id,
        "scenario": {
            "personal": {},
            "order": {},
            "product": {},
            "flow": flow,
            "subflow": subflow,
        },
        "original": [
            ["agent", "Hello, how can I help you today?"],
            ["customer", f"I have a question about case {convo_id}. {note}"],
            ["action", f"ACTIONMARKER searching the knowledge base for {convo_id}"],
            ["agent", "  Sure,   give me one moment. "],
            ["customer", "Thanks a lot"],
        ],
        "delexed": [],
    }


def abcd_ontology() -> dict:
    return {
        "intents": {
            "flows": list(abcd.SUBFLOWS),
            "subflows": {
                flow: sorted({abcd.ontology_name(flow, label) for label in labels})
                for flow, labels in abcd.SUBFLOWS.items()
            },
        }
    }


def write_abcd(root: Path, count: int) -> None:
    labels = [(flow, label) for flow, table in abcd.SUBFLOWS.items() for label in table]
    train = [
        _conversation(1000 + index, *labels[(7 * index) % len(labels)])
        for index in range(count)
    ]
    train += [
        _conversation(5000, "subscription_inquiry", "status_questions"),
        _conversation(5001, "order_issue", "status_delivery_date"),
    ]
    splits = {
        "train": train,
        "dev": [_conversation(9000, "product_defect", "return_size", "DEVSPLIT")],
        "test": [_conversation(9001, "product_defect", "return_size", "TESTSPLIT")],
    }
    (root / abcd.DATA).parent.mkdir(parents=True, exist_ok=True)
    (root / abcd.DATA).write_bytes(gzip.compress(json.dumps(splits).encode(), mtime=0))
    _write(root / abcd.ONTOLOGY, abcd_ontology())
    guidelines = {
        name: {"description": f"synthetic topic number {index}", "subflows": {}}
        for index, name in enumerate(abcd.FLOW_NAMES.values())
    }
    guidelines["Shipping Issue"][
        "description"
    ] = "check our update a shipment of an item"
    _write(root / abcd.GUIDELINES, guidelines)


def _slot(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "is_categorical": False,
        "possible_values": [],
    }


def _intent(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "is_transactional": False,
        "required_slots": [],
        "optional_slots": {},
        "result_slots": [],
    }


SGD_SCHEMA = [
    {
        "service_name": "Venues_1",
        "description": "Synthetic venue service",
        "slots": [
            _slot("city", "City where the venue is located"),
            _slot("phone_number", "Phone number of the venue"),
            _slot("street_address", "Address of the venue"),
            _slot("has_wifi", "Boolean flag indicating if the venue has wifi"),
        ],
        "intents": [
            _intent("FindVenue", "Find a venue in a city"),
            _intent("BookVenue", "Book a table at a venue"),
        ],
    },
    {
        "service_name": "Tunes_1",
        "description": "Synthetic tune service",
        "slots": [
            _slot("title", "Title of the tune"),
            _slot("artist", "Artist who performed the tune"),
            _slot("album", "Album the tune belongs to"),
        ],
        "intents": [
            _intent("LookupTune", "Search for a tune"),
            _intent("PlayTune", "Play the selected tune"),
            _intent("ShareTune", "Share a tune with a friend"),
        ],
    },
    {
        "service_name": "Rides_1",
        "description": "Synthetic ride service",
        "slots": [
            _slot("destination", "Destination for the ride"),
            _slot("fare", "Total fare for the ride"),
        ],
        "intents": [_intent("GetRide", "Call a ride to a destination")],
    },
]


def _dialogue(dialogue_id: str, user_turns: list[list[tuple[str, str, list[str]]]]):
    turns = []
    for number, frames in enumerate(user_turns):
        turns.append(
            {
                "speaker": "USER",
                "utterance": f"user message {number} of {dialogue_id}",
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
                "utterance": f"system reply {number}",
                "frames": [{"service": frames[0][0], "slots": [], "actions": []}],
            }
        )
    services = sorted({frame[0] for frames in user_turns for frame in frames})
    return {"dialogue_id": dialogue_id, "services": services, "turns": turns}


def sgd_dialogues() -> list[dict]:
    dialogues = []
    for index in range(36):
        request = [["phone_number"], ["street_address"], ["phone_number", "has_wifi"]]
        dialogues.append(
            _dialogue(
                f"1_{index:05d}",
                [
                    [("Venues_1", "FindVenue", [])],
                    [("Venues_1", "FindVenue", request[index % 3])],
                    [("Venues_1", "BookVenue" if index % 2 else "FindVenue", [])],
                ],
            )
        )
    for index in range(36):
        dialogues.append(
            _dialogue(
                f"2_{index:05d}",
                [
                    [("Tunes_1", "LookupTune", [])],
                    [
                        (
                            "Tunes_1",
                            ("PlayTune", "ShareTune", "LookupTune")[index % 3],
                            [("artist", "album", "title")[index % 3]],
                        )
                    ],
                ],
            )
        )
    for index in range(8):
        dialogues.append(
            _dialogue(f"3_{index:05d}", [[("Rides_1", "GetRide", ["fare"])]])
        )
    return dialogues


def write_sgd(root: Path) -> None:
    dialogues = sgd_dialogues()
    _write(root / sgd.SCHEMA, SGD_SCHEMA)
    _write(root / "train" / "dialogues_001.json", dialogues[:40])
    _write(root / "train" / "dialogues_002.json", dialogues[40:])


def qasc_items() -> list[dict]:
    items = []
    for index in range(24):
        pair = index // 2
        fact1 = (
            f"Synthetic fact {pair} says   something"
            if index % 2
            else f"synthetic FACT {pair} says something"
        )
        items.append(
            {
                "id": f"Q{index:03d}",
                "question": f"Which option fits synthetic question {index}?",
                "choices": {
                    "text": [f"option {index}-{choice}" for choice in range(8)],
                    "label": list("ABCDEFGH"),
                },
                "answerKey": "ABCDEFGH"[index % 8],
                "fact1": fact1,
                "fact2": f"Second fact {pair}.",
                "combinedfact": f"COMBINEDMARKER {index}",
                "formatted_question": f"Which option fits synthetic question {index}?",
            }
        )
    duplicated = dict(
        items[0],
        id="QDUP",
        choices={
            "text": ["same", "Same ", "b", "c", "d", "e", "f", "g"],
            "label": list("ABCDEFGH"),
        },
        answerKey="A",
    )
    return [*items, duplicated]


def write_qasc(root: Path) -> None:
    (root / qasc.DATA).parent.mkdir(parents=True, exist_ok=True)
    (root / qasc.DATA).write_text(
        "".join(json.dumps(item) + "\n" for item in qasc_items()), encoding="utf-8"
    )


def musique_items(count: int = 6) -> list[dict]:
    items = []
    for item in range(count):
        order = [(7 * position + item) % 20 for position in range(20)]
        support = (3 * item + 5) % 20
        paragraphs = [
            {
                "idx": idx,
                "title": f"Title {item}-{idx}",
                "paragraph_text": f"Text of paragraph {idx} for item {item}.",
                "is_supporting": idx in (support, (support + 1) % 20),
            }
            for idx in order
        ]
        steps = [
            {
                "id": 1,
                "question": "hop one",
                "answer": "bridge",
                "paragraph_support_idx": (support + 1) % 20,
            },
            {
                "id": 2,
                "question": "hop two",
                "answer": "final",
                "paragraph_support_idx": support,
            },
        ]
        answerable = {
            "id": f"2hop__{item}_{item + 100}",
            "paragraphs": paragraphs,
            "question": f"Synthetic question {item}?",
            "question_decomposition": steps,
            "answer": "final",
            "answer_aliases": [],
            "answerable": True,
        }
        unanswerable = dict(
            answerable,
            answerable=False,
            paragraphs=[
                dict(
                    paragraph,
                    is_supporting=False,
                    paragraph_text=f"Distractor {paragraph['idx']}.",
                )
                for paragraph in paragraphs
            ],
            question_decomposition=[
                dict(step, paragraph_support_idx=None) for step in steps
            ],
        )
        items += [answerable, unanswerable] if item % 2 else [unanswerable, answerable]
    return items


def write_musique(root: Path, items: list[dict] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / musique.DATA).write_text(
        "".join(json.dumps(item) + "\n" for item in items or musique_items()),
        encoding="utf-8",
    )


def _rerotated(descriptions: list[str], gold: int, row: dict) -> tuple[list, int]:
    return rotate(choice_options(descriptions), gold, f"a1-v1:{row['id']}")


class TempDirCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()


class AbcdTest(TempDirCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        write_abcd(cls.base / "abcd", 1000)
        cls.rows, cls.report = abcd.build(cls.base / "abcd")
        cls.choice = cls.rows[abcd.CHOICE_FAMILY]
        cls.noul = cls.rows[abcd.NOUL_FAMILY]

    def test_table_expands_the_55_ontology_subflows(self) -> None:
        names = abcd_ontology()["intents"]["subflows"]
        self.assertEqual(sum(len(labels) for labels in names.values()), 55)
        self.assertEqual(sum(len(table) for table in abcd.SUBFLOWS.values()), 95)
        self.assertEqual(abcd.check_ontology(abcd_ontology()), sorted(abcd.SUBFLOWS))
        self.assertEqual(abcd.choice_flows(), (sorted(abcd.SUBFLOWS), {}))

    def test_parses_train_only_and_records_drops(self) -> None:
        self.assertEqual((len(self.choice), len(self.noul)), (1001, 1002))
        self.assertEqual(self.report["splits_ignored"], ["dev", "test"])
        self.assertEqual(
            self.report["dropped"],
            {abcd.CHOICE_FAMILY: {"subflow_not_in_ontology": 1}, abcd.NOUL_FAMILY: {}},
        )
        self.assertEqual(
            self.report["label_aliases"], {"status_questions->status_active": 1}
        )
        self.assertEqual(self.report["flow_description_fixes"], ["shipping_issue"])
        self.assertEqual(
            set(self.report["inputs"]), {abcd.DATA, abcd.ONTOLOGY, abcd.GUIDELINES}
        )
        ids = {row["audit_metadata"]["convo_id"] for row in self.noul}
        self.assertFalse(ids & {9000, 9001})

    def test_state_is_the_transcript_without_action_turns(self) -> None:
        for row in self.choice + self.noul:
            self.assertNotIn("ACTIONMARKER", row["state"])
            lines = row["state"].split("\n")
            self.assertEqual(len(lines), 4)
            self.assertTrue(
                all(line.startswith(("Customer: ", "Agent: ")) for line in lines)
            )
            self.assertIn("Agent: Sure, give me one moment.", lines)
            meta = row["audit_metadata"]
            for field in ("flow", "subflow", "gold_flow"):
                if field in meta:
                    self.assertNotIn(meta[field], row["state"])

    def test_choice_gold_is_the_described_subflow_rotated_by_row_id(self) -> None:
        for row in self.choice:
            meta = row["audit_metadata"]
            table = abcd.SUBFLOWS[meta["flow"]]
            label = abcd.LABEL_ALIASES.get(meta["subflow"], meta["subflow"])
            self.assertEqual(row["options"][row["label"]]["description"], table[label])
            self.assertEqual(row["instructions"], abcd.CHOICE_INSTRUCTIONS)
            options, gold = _rerotated(
                list(table.values()), list(table).index(label), row
            )
            self.assertEqual((row["options"], row["label"]), (options, gold))

    def test_noul_is_balanced_and_false_asks_another_flow(self) -> None:
        share = sum(row["label"] for row in self.noul) / len(self.noul)
        self.assertTrue(0.45 <= share <= 0.55, share)
        for row in self.noul:
            meta = row["audit_metadata"]
            self.assertEqual(row["label"] == 1, meta["asked_flow"] == meta["gold_flow"])
            self.assertEqual([o["key"] for o in row["options"]], ["false", "true"])
        shipping = [
            r
            for r in self.noul
            if r["audit_metadata"]["asked_flow"] == "shipping_issue"
        ]
        self.assertTrue(shipping)
        self.assertTrue(
            all(
                row["instructions"]
                == "Is the customer's issue about checking or updating a shipment of an item?"
                for row in shipping
            )
        )

    def test_each_conversation_is_one_group(self) -> None:
        groups = collections.defaultdict(set)
        for row in self.choice + self.noul:
            groups[row["audit_metadata"]["convo_id"]].add(
                (row["group_id"], row["family"])
            )
        self.assertTrue(
            all(len({g for g, _ in pairs}) == 1 for pairs in groups.values())
        )
        self.assertEqual(max(len(pairs) for pairs in groups.values()), 2)

    def test_ontology_mismatch_and_indistinct_flows(self) -> None:
        ontology = abcd_ontology()
        ontology["intents"]["subflows"]["product_defect"].append("return_smell")
        with self.assertRaises(ValueError):
            abcd.check_ontology(ontology)
        eligible, skipped = abcd.choice_flows(
            {
                "a": {"x": "One", "y": "one", "z": "Two"},
                "b": {"x": "One", "y": "Two"},
                "c": {"x": "One", "y": "Two", "z": "Three"},
            }
        )
        self.assertEqual(eligible, ["c"])
        self.assertEqual(
            skipped, {"a": "indistinct_descriptions", "b": "fewer_than_two_siblings"}
        )


class SgdTest(TempDirCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        write_sgd(cls.base / "sgd")
        cls.rows, cls.report = sgd.build(cls.base / "sgd")
        cls.choice = cls.rows[sgd.CHOICE_FAMILY]
        cls.noul = cls.rows[sgd.NOUL_FAMILY]

    def test_intent_activation_detection_and_window(self) -> None:
        dialogue = _dialogue(
            "9_00000",
            [
                [("Venues_1", "FindVenue", [])],
                [("Venues_1", "FindVenue", [])],
                [("Tunes_1", "LookupTune", [])],
                [("Venues_1", "FindVenue", [])],
                [("Venues_1", "BookVenue", []), ("Tunes_1", "LookupTune", [])],
                [("Venues_1", "NONE", [])],
                [("Venues_1", "BookVenue", ["phone_number"])],
            ],
        )
        self.assertEqual(
            sgd.activations(dialogue),
            [
                (0, "Venues_1", "FindVenue"),
                (4, "Tunes_1", "LookupTune"),
                (8, "Venues_1", "BookVenue"),
                (12, "Venues_1", "BookVenue"),
            ],
        )
        self.assertEqual(sgd.requests(dialogue), [(12, "Venues_1", ("phone_number",))])
        lines = sgd.window(dialogue["turns"], 12).split("\n")
        self.assertEqual(len(lines), 8)
        self.assertEqual(lines[0], "System: system reply 2")
        self.assertEqual(lines[-1], "User: user message 6 of 9_00000")

    def test_slot_questions_read_naturally(self) -> None:
        self.assertEqual(
            sgd.slot_question("Boolean flag indicating if the venue has wifi"),
            "In the latest message, does the user ask whether the venue has wifi?",
        )
        self.assertEqual(
            sgd.slot_question("Phone number of the venue"),
            "In the latest message, does the user ask for the phone number of the venue?",
        )
        self.assertEqual(
            sgd.slot_question("Whether the booking is refundable or not"),
            "In the latest message, does the user ask whether the booking is refundable or not?",
        )
        self.assertEqual(
            sgd.slot_question("The account type of the user"),
            "In the latest message, does the user ask for the account type of the user?",
        )

    def test_choice_rows_are_intent_balanced_per_service(self) -> None:
        schema = {service["service_name"]: service for service in SGD_SCHEMA}
        counts = collections.Counter(
            (row["audit_metadata"]["service"], row["audit_metadata"]["intent"])
            for row in self.choice
        )
        per_service = collections.defaultdict(set)
        for (service, intent), count in counts.items():
            per_service[service].add(count)
        self.assertEqual(set(per_service), {"Venues_1", "Tunes_1"})
        self.assertTrue(all(len(sizes) == 1 for sizes in per_service.values()))
        self.assertEqual(
            {(s, i) for s, i in counts},
            {(s, i["name"]) for s in per_service for i in schema[s]["intents"]},
        )
        self.assertEqual(
            self.report[sgd.CHOICE_FAMILY]["services_ineligible"], ["Rides_1"]
        )
        for row in self.choice:
            meta = row["audit_metadata"]
            intents = schema[meta["service"]]["intents"]
            gold = [i["name"] for i in intents].index(meta["intent"])
            self.assertEqual(
                row["options"][row["label"]]["description"],
                intents[gold]["description"],
            )
            options, label = _rerotated([i["description"] for i in intents], gold, row)
            self.assertEqual((row["options"], row["label"]), (options, label))
            self.assertTrue(row["state"].split("\n")[-1].startswith("User: "))
        dialogue_ids = [row["audit_metadata"]["dialogue_id"] for row in self.choice]
        self.assertEqual(len(dialogue_ids), len(set(dialogue_ids)))

    def test_noul_rows_are_balanced_per_asked_slot(self) -> None:
        strata = collections.defaultdict(collections.Counter)
        for row in self.noul:
            meta = row["audit_metadata"]
            strata[(meta["service"], meta["asked_slot"])][row["label"]] += 1
            self.assertEqual(
                row["label"] == 1, meta["asked_slot"] in meta["requested_slots"]
            )
            self.assertTrue(
                row["instructions"].startswith(
                    "In the latest message, does the user ask"
                )
            )
        self.assertTrue(strata)
        self.assertTrue(all(c[0] == c[1] > 0 for c in strata.values()))
        self.assertEqual(sum(row["label"] for row in self.noul) * 2, len(self.noul))
        self.assertFalse(any(service == "Rides_1" for service, _ in strata))
        self.assertIn("slot_balance", self.report["dropped"][sgd.NOUL_FAMILY])
        dialogue_ids = [row["audit_metadata"]["dialogue_id"] for row in self.noul]
        self.assertEqual(len(dialogue_ids), len(set(dialogue_ids)))

    def test_state_never_names_labels(self) -> None:
        for row in self.choice + self.noul:
            for name in (
                "FindVenue",
                "BookVenue",
                "LookupTune",
                "PlayTune",
                "ShareTune",
            ):
                self.assertNotIn(name, row["state"])


class QascTest(TempDirCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        write_qasc(cls.base / "qasc")
        rows, cls.report = qasc.build(cls.base / "qasc")
        cls.rows = rows[qasc.FAMILY]

    def test_gold_mapping_state_and_groups(self) -> None:
        self.assertEqual(len(self.rows), 24)
        self.assertEqual(
            self.report["dropped"], {qasc.FAMILY: {"gold_text_duplicated": 1}}
        )
        items = {item["id"]: item for item in qasc_items()}
        for row in self.rows:
            item = items[row["audit_metadata"]["qasc_id"]]
            gold = item["choices"]["label"].index(item["answerKey"])
            self.assertEqual(
                row["options"][row["label"]]["description"],
                item["choices"]["text"][gold],
            )
            options, label = _rerotated(item["choices"]["text"], gold, row)
            self.assertEqual((row["options"], row["label"]), (options, label))
            self.assertEqual(row["instructions"], item["question"])
            self.assertTrue(row["state"].startswith("Fact 1: "))
            self.assertIn("\nFact 2: Second fact", row["state"])
            self.assertNotIn("COMBINEDMARKER", row["state"])
            self.assertNotIn("combinedfact", json.dumps(row["audit_metadata"]))
        self.assertEqual(len({row["group_id"] for row in self.rows}), 12)


class MusiqueTest(TempDirCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        write_musique(cls.base / "musique")
        cls.rows, cls.report = musique.build(cls.base / "musique")

    def test_twins_form_one_balanced_group_with_their_choice_row(self) -> None:
        noul, choice = self.rows[musique.NOUL_FAMILY], self.rows[musique.CHOICE_FAMILY]
        self.assertEqual(
            (len(noul), len(choice), self.report["twin_pairs"]), (12, 6, 6)
        )
        groups = collections.defaultdict(list)
        for row in noul + choice:
            groups[row["group_id"]].append(row)
        self.assertEqual(len(groups), 6)
        for members in groups.values():
            self.assertEqual(
                sorted(
                    (row["family"], row["label"])
                    for row in members
                    if row["task_type"] == "noul"
                ),
                [(musique.NOUL_FAMILY, 0), (musique.NOUL_FAMILY, 1)],
            )
            self.assertEqual(sum(row["task_type"] == "choice" for row in members), 1)
            self.assertEqual(
                len({row["audit_metadata"]["musique_id"] for row in members}), 1
            )
        instructions = {row["instructions"] for row in noul}
        self.assertEqual(len(instructions), 6)

    def test_final_support_maps_idx_to_state_position(self) -> None:
        items = {item["id"]: item for item in musique_items() if item["answerable"]}
        for row in self.rows[musique.CHOICE_FAMILY]:
            item = items[row["audit_metadata"]["musique_id"]]
            support = item["question_decomposition"][-1]["paragraph_support_idx"]
            position = [p["idx"] for p in item["paragraphs"]].index(support)
            self.assertNotEqual(position, support)
            self.assertEqual(row["label"], position)
            gold = f"Paragraph {position + 1} — {item['paragraphs'][position]['title']}"
            self.assertEqual(row["options"][row["label"]]["description"], gold)
            self.assertEqual(row["state"].split("\n\n")[position].split(":")[0], gold)
            self.assertEqual(
                [option["key"] for option in row["options"]],
                [f"o{k}" for k in range(1, 21)],
            )
            self.assertTrue(
                all(
                    option["description"].startswith(f"Paragraph {k} — ")
                    for k, option in enumerate(row["options"], 1)
                )
            )

    def test_select_keeps_whole_twin_groups(self) -> None:
        kept = musique.select(self.rows, pairs=2)
        self.assertEqual(len(kept[musique.NOUL_FAMILY]), 4)
        self.assertEqual(len(kept[musique.CHOICE_FAMILY]), 2)
        self.assertEqual(
            {row["group_id"] for row in kept[musique.NOUL_FAMILY]},
            {row["group_id"] for row in kept[musique.CHOICE_FAMILY]},
        )

    def test_twin_and_support_violations_raise(self) -> None:
        items = musique_items(2)
        cases = {
            "lonely": items[1:],
            "two_answerable": [items[0], dict(items[0], answerable=True), *items[2:]],
            "bad_support": [
                (
                    dict(
                        item,
                        question_decomposition=[
                            *item["question_decomposition"][:-1],
                            dict(
                                item["question_decomposition"][-1],
                                paragraph_support_idx=99,
                            ),
                        ],
                    )
                    if item["answerable"]
                    else item
                )
                for item in items
            ],
        }
        for name, case in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                write_musique(self.base / name, case)
                musique.build(self.base / name)


class BuildArmTest(TempDirCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.roots = {name: cls.base / name for name in build_a1_a3.SOURCE_DIRS}
        write_abcd(cls.roots["abcd"], 120)
        write_sgd(cls.roots["sgd"])
        write_qasc(cls.roots["qasc"])
        write_musique(cls.roots["musique"])

    def _outputs(self, out_dir: Path, arm: str) -> list[bytes]:
        return [path.read_bytes() for path in build_a1_a3.outputs(out_dir, arm)]

    def _check_rows(self, out_dir: Path, arm: str) -> int:
        count = 0
        for split, partition in (("train", "train"), ("aho", "select")):
            for line in (out_dir / f"{arm}.{split}.jsonl").read_text().splitlines():
                validate_row(json.loads(line), partition)
                count += 1
        return count

    def test_a1_is_valid_capped_and_byte_identical(self) -> None:
        caps = {abcd.CHOICE_FAMILY: 40, abcd.NOUL_FAMILY: 40, qasc.FAMILY: 10}
        first = build_a1_a3.build_arm("a1", self.roots, self.base / "a1-first", caps)
        build_a1_a3.build_arm("a1", self.roots, self.base / "a1-second", caps)
        self.assertEqual(
            self._outputs(self.base / "a1-first", "a1"),
            self._outputs(self.base / "a1-second", "a1"),
        )
        build = first["build"]
        self.assertEqual(
            self._check_rows(self.base / "a1-first", "a1"),
            sum(f["rows"] for f in build["families"].values()),
        )
        self.assertEqual(set(build["inputs"]), {abcd.SOURCE, sgd.SOURCE, qasc.SOURCE})
        self.assertEqual(
            set(build["inputs"][sgd.SOURCE]),
            {sgd.SCHEMA, "train/dialogues_001.json", "train/dialogues_002.json"},
        )
        self.assertEqual(
            set(build["code_sha256"]),
            {
                "v2.data.build_a1_a3",
                "v2.data.sources.common",
                "training.model.data",
                "v2.data.sources.abcd",
                "v2.data.sources.sgd",
                "v2.data.sources.qasc",
            },
        )
        self.assertEqual(
            build["seeds"],
            {abcd.SOURCE: abcd.SEED, sgd.SOURCE: sgd.SEED, qasc.SOURCE: qasc.SEED},
        )
        self.assertEqual(build["families"][abcd.CHOICE_FAMILY]["rows"], 40)
        self.assertEqual(build["families"][abcd.NOUL_FAMILY]["rows"], 40)
        self.assertLessEqual(build["families"][qasc.FAMILY]["rows"], 10)
        self.assertIn(sgd.CHOICE_FAMILY, build["shortfalls"])
        self.assertEqual(
            build["rotation"], {"seed": "a1-v1:<row id>", "unrotated_families": []}
        )
        self.assertIn("gold_position_by_option_count", build["families"][qasc.FAMILY])
        self.assertEqual(build["warnings"], [])

    def test_a3_reports_unrotated_gold_positions(self) -> None:
        manifest = build_a1_a3.build_arm(
            "a3",
            {"musique": self.roots["musique"]},
            self.base / "a3",
            {"musique_pairs": 4},
        )
        build = manifest["build"]
        self.assertEqual(build["families"][musique.NOUL_FAMILY]["rows"], 8)
        self.assertEqual(build["families"][musique.NOUL_FAMILY]["true_share"], 0.5)
        self.assertEqual(build["families"][musique.CHOICE_FAMILY]["rows"], 4)
        self.assertEqual(
            build["rotation"]["unrotated_families"], [musique.CHOICE_FAMILY]
        )
        self.assertTrue(build["warnings"])
        self.assertTrue(
            all(w.startswith(musique.CHOICE_FAMILY) for w in build["warnings"])
        )
        self.assertEqual(self._check_rows(self.base / "a3", "a3"), 12)
        self.assertEqual(
            build_a1_a3.position_warnings(
                "f", [{"options": [1, 2, 3, 4], "label": i % 4} for i in range(8)]
            ),
            [],
        )

    def test_cli_refuses_to_overwrite_and_rejects_foreign_flags(self) -> None:
        out_dir = self.base / "cli"
        argv = [
            "--arm",
            "a3",
            "--musique",
            str(self.roots["musique"]),
            "--out-dir",
            str(out_dir),
            "--musique-pairs",
            "3",
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(build_a1_a3.main(argv), 0)
            before = self._outputs(out_dir, "a3")
            self.assertEqual(build_a1_a3.main(argv), 2)
            self.assertEqual(self._outputs(out_dir, "a3"), before)
            with self.assertRaises(SystemExit):
                build_a1_a3.main([*argv[:-2], "--qasc-cap", "5"])
            with self.assertRaises(SystemExit):
                build_a1_a3.main(
                    [
                        "--arm",
                        "a1",
                        "--abcd",
                        str(self.roots["abcd"]),
                        "--out-dir",
                        str(out_dir),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
