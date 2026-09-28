from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import MAX_INPUT_CHARS, input_chars, validate
from v2.eval.sealed.sources import tutormoments as source

CHOICE = "tutormoments/moment_type"
NOUL = "tutormoments/is_rapport"
ROLES = ("Tutor", "Student")


def odd(item_id: str) -> bool:
    return int(hashlib.sha256(item_id.encode("utf-8")).hexdigest(), 16) % 2 == 1


def transcript(tid: str, texts: list[str]) -> dict:
    turns = [
        {"turn_number": n, "role": ROLES[(n - 1) % 2], "text": text}
        for n, text in enumerate(texts, 1)
    ]
    return {
        "transcript_id": tid,
        "turns": turns[::-1],
        "primary_language": "",
        "session_summary": f"LEAK summary of {tid}",
    }


def annotation(tid: str, kind: str, *ranges, **extra) -> dict:
    return {
        "transcript_id": tid,
        "annotation_type": kind,
        "summary": "LEAK pass summary",
        "turn_annotations": [
            {
                "turn_number_start": start,
                "turn_number_end": end,
                "situation": "LEAK situation",
                "action": "LEAK action",
                "result": "LEAK result",
                "caption": "LEAK caption",
            }
            for start, end in ranges
        ],
        **extra,
    }


def rendered(texts: list[str], *numbers: int) -> list[dict]:
    return [
        {"turn": n, "speaker": ROLES[(n - 1) % 2], "text": texts[n - 1]}
        for n in numbers
    ]


def snapshot(root: Path, transcripts: list[dict], annotations: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, records in (("transcripts", transcripts), ("annotations", annotations)):
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        (root / f"{name}.jsonl").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return root


def convert(transcripts: list[dict], annotations: list[dict]):
    with tempfile.TemporaryDirectory() as tmp:
        return list(source.candidates(snapshot(Path(tmp), transcripts, annotations)))


ENGLISH = [
    "Okay, so what is the first step here?",
    "I think it is to add the two numbers.",
    "Good. And what do you get?",
    "Is it twelve?",
    "Yes! You are doing great today.",
    "Thanks, this is fun.",
    "Can you try the next one on your own?",
    "Okay, I will try it.",
    "What is the rule for the tens?",
    "You carry the one to the tens.",
]
SPANISH = [
    "¿Qué número sigue en la serie?",
    "Creo que es el cinco, pero no estoy segura.",
    "Muy bien, lo estás haciendo muy bien.",
]
TRANSCRIPTS = [transcript("t01", ENGLISH), transcript("t02", SPANISH)]
ANNOTATIONS = [
    annotation(
        "t01",
        "scaffolding",
        (2, 3),
        (5, 5),
        (0, 1),
        (9, 11),
        (4, 3),
        ("6", 7),
        (7.0, 8),
        (None, 8),
    ),
    annotation(
        "t01", "scaffolding", (2, 3), (1, 1), (9, 10), pass_type="v2_preselected"
    ),
    annotation("t01", "rapport", (5, 6), (4, 4), (7, 8)),
    annotation("t01", "summary", (1, 10)),
    annotation("t99", "rapport", (1, 1)),
    annotation("t02", "rapport", (1, 2)),
    {
        "transcript_id": "t02",
        "annotation_type": "scaffolding",
        "turn_annotations": None,
    },
]
KINDS = {
    "t01:4-4": "rapport",
    "t01:7-8": "rapport",
    "t01:1-1": "scaffolding",
    "t01:2-3": "scaffolding",
    "t01:9-10": "scaffolding",
    "t02:1-2": "rapport",
}


class TutorMomentsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = snapshot(Path(self.tmp.name) / "snap", TRANSCRIPTS, ANNOTATIONS)
        self.out = list(source.candidates(self.root))
        self.by_id = {c.source_item_id: c for c in self.out}

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_candidates_valid_and_tasks_declared(self):
        for c in self.out:
            self.assertEqual(validate(c, source.SPEC), [])
            self.assertIn(c.language, source.SPEC.languages)
            self.assertIsNone(c.date)
        self.assertEqual({c.task for c in self.out}, set(source.SPEC.tasks))

    def test_one_item_per_distinct_human_moment(self):
        self.assertEqual([c.source_item_id for c in self.out], list(KINDS))
        for c in self.out:
            kind = KINDS[c.source_item_id]
            self.assertEqual(c.gold, kind == "rapport" if c.task == NOUL else kind)
            self.assertEqual(c.group_id, c.source_item_id.split(":")[0])
        self.assertEqual(
            self.by_id["t01:2-3"].state,
            {
                "preceding_turns": rendered(ENGLISH, 1),
                "moment": rendered(ENGLISH, 2, 3),
            },
        )
        self.assertEqual(
            self.by_id["t01:9-10"].state,
            {
                "preceding_turns": rendered(ENGLISH, *range(1, 9)),
                "moment": rendered(ENGLISH, 9, 10),
            },
        )

    def test_duplicate_marks_collapse_into_one_item(self):
        ids = [c.source_item_id for c in self.out]
        self.assertEqual(ids.count("t01:2-3"), 1)
        self.assertEqual(len(ids), len(set(ids)))

    def test_moments_overlapping_the_other_type_are_excluded(self):
        self.assertNotIn("t01:5-5", self.by_id)
        self.assertNotIn("t01:5-6", self.by_id)
        for item in ("t01:2-3", "t01:4-4", "t01:7-8", "t01:9-10"):
            self.assertIn(item, self.by_id)

    def test_out_of_range_malformed_and_foreign_marks_are_ignored(self):
        for item in ("t01:0-1", "t01:9-11", "t01:4-3", "t01:6-7", "t01:1-10"):
            self.assertNotIn(item, self.by_id)
        self.assertEqual({c.group_id for c in self.out}, {"t01", "t02"})
        # Accepting any malformed 6-7 / 7-8 scaffolding mark would exclude this one.
        self.assertEqual(KINDS["t01:7-8"], "rapport")
        self.assertIn("t01:7-8", self.by_id)

    def test_task_split_follows_item_id_hash_parity(self):
        count = 40
        kinds = {n: source.TYPES[(n + 1) % 2] for n in range(1, count + 1)}
        texts = [f"Okay, step {n}: what do you get?" for n in range(1, count + 1)]
        annotations = [
            annotation("split", kind, *[(n, n) for n, k in kinds.items() if k == kind])
            for kind in source.TYPES
        ]
        out = convert([transcript("split", texts)], annotations)
        self.assertEqual(len(out), count)
        for c in out:
            kind = kinds[int(c.source_item_id.split(":")[1].split("-")[0])]
            if odd(c.source_item_id):
                self.assertEqual(c.task, NOUL)
                self.assertIs(c.question, source.RAPPORT_Q)
                self.assertIs(c.gold, kind == "rapport")
            else:
                self.assertEqual(c.task, CHOICE)
                self.assertIs(c.question, source.MOMENT_Q)
                self.assertEqual(c.gold, kind)
            self.assertEqual(c.balance_label, str(c.gold))
            self.assertEqual(validate(c, source.SPEC), [])
        self.assertEqual(
            {(c.task, c.balance_label) for c in out},
            {
                (NOUL, "True"),
                (NOUL, "False"),
                (CHOICE, "rapport"),
                (CHOICE, "scaffolding"),
            },
        )
        self.assertEqual(list(source.MOMENT_Q["criteria"]), ["scaffolding", "rapport"])
        self.assertEqual(
            source.RAPPORT_Q["criteria"],
            {
                "true": source.MOMENT_Q["criteria"]["rapport"],
                "false": source.MOMENT_Q["criteria"]["scaffolding"],
            },
        )

    def test_state_is_the_full_moment_and_preceding_turns_within_context_chars(self):
        self.assertEqual(source.CONTEXT_CHARS, 2_000)
        sizes = [10, 50, 200, 900, 900, 3000, 20, 30]
        texts = [chr(ord("a") + i) * size for i, size in enumerate(sizes)]
        annotations = [
            annotation("ctx", "scaffolding", (6, 7), (7, 7)),
            annotation("ctx", "rapport", (2, 2), (8, 8)),
        ]
        out = convert([transcript("ctx", texts)], annotations)
        self.assertEqual(
            {c.source_item_id: c.state for c in out},
            {
                "ctx:2-2": {
                    "preceding_turns": rendered(texts, 1),
                    "moment": rendered(texts, 2),
                },
                "ctx:8-8": {
                    "preceding_turns": rendered(texts, 7),
                    "moment": rendered(texts, 8),
                },
                "ctx:6-7": {
                    "preceding_turns": rendered(texts, 3, 4, 5),
                    "moment": rendered(texts, 6, 7),
                },
                "ctx:7-7": {"preceding_turns": [], "moment": rendered(texts, 7)},
            },
        )
        self.assertEqual(sum(sizes[2:5]), source.CONTEXT_CHARS)
        by_id = {c.source_item_id: c for c in out}
        self.assertEqual(by_id["ctx:6-7"].overlap_texts, texts[2:7])

    def test_oversize_moments_are_dropped_never_truncated(self):
        texts = ["Okay, you can start.", "o" * MAX_INPUT_CHARS, "Is it five?"]
        annotations = [
            annotation("t30", "scaffolding", (2, 2)),
            annotation("t30", "rapport", (1, 1), (3, 3)),
        ]
        out = {
            c.source_item_id: c
            for c in convert([transcript("t30", texts)], annotations)
        }
        self.assertEqual(sorted(out), ["t30:1-1", "t30:3-3"])
        self.assertEqual(
            out["t30:3-3"].state,
            {"preceding_turns": [], "moment": rendered(texts, 3)},
        )

        item = "big:1-1"
        question = source.RAPPORT_Q if odd(item) else source.MOMENT_Q
        empty = {"preceding_turns": [], "moment": rendered([""], 1)}
        room = MAX_INPUT_CHARS - input_chars(empty, question)
        marks = [annotation("big", "rapport", (1, 1))]
        (fit,) = convert([transcript("big", ["f" * room])], marks)
        self.assertEqual(fit.source_item_id, item)
        self.assertEqual(fit.state["moment"], rendered(["f" * room], 1))
        self.assertEqual(input_chars(fit.state, fit.question), MAX_INPUT_CHARS)
        self.assertEqual(convert([transcript("big", ["f" * (room + 1)])], marks), [])

    def test_session_language_from_function_words(self):
        self.assertEqual(self.by_id["t02:1-2"].language, "es")
        self.assertEqual({c.language for c in self.out if c.group_id == "t01"}, {"en"})
        self.assertEqual(
            source.session_language([{"text": None}, {"text": "12 + 7"}]), "en"
        )

    def test_annotator_descriptions_and_summaries_are_never_read(self):
        for c in self.out:
            self.assertNotIn("LEAK", json.dumps(c.to_json(), ensure_ascii=False))

    def test_runner_counts(self):
        receipt = runner.run("tutormoments", self.root, None)
        self.assertEqual(receipt["invalid"], {})
        self.assertEqual(receipt["valid"], len(KINDS))
        self.assertEqual(receipt["tasks"][NOUL]["candidates"], 4)
        self.assertEqual(receipt["tasks"][NOUL]["groups"], 1)
        self.assertEqual(receipt["tasks"][CHOICE]["groups"], 2)
        self.assertEqual(receipt["tasks"][CHOICE]["language:es"], 1)


if __name__ == "__main__":
    unittest.main()
