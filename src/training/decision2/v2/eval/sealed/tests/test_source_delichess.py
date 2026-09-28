from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import MAX_INPUT_CHARS, input_chars, validate
from v2.eval.sealed.sources import delichess as source

FUNCTION = "delichess/communicative_function"
STANCE = "delichess/epistemic_stance"
HEADER = [
    "gemini_communicative_function",
    "utterance_id",
    "dialogue_id",
    "speaker_id",
    "utterance_position",
    "text",
    "communicative_function_match",
    "human_communicative_function",
    "gemini_epistemic_stance",
    "human_epistemic_stance",
    "epistemic_stance_match",
    "annotator_note",
]
FUNCTION_GOLD = {
    "PROPOSE_ANSWER": "propose_answer",
    "PROVIDE_REASONING": "provide_reasoning",
    "EXPLORE_ALTERNATIVES": "explore_alternatives",
    "REQUEST_REASONING": "request_reasoning",
    "EVALUATE_CRITIQUE": "evaluate_critique",
    "AGREEMENT_ALIGNMENT": "agreement_alignment",
    "COORDINATE_DECISION": "coordinate_decision",
    "SOCIAL_MODERATION": "social_moderation",
    "SOCIAL_OFFTASK": "social_offtask",
}
STANCE_GOLD = {
    "NO_TASK_STANCE": "no_task_stance",
    "HEDGED": "hedged",
    "NO_HEDGED": "unhedged",
}
FUNCTIONS = list(FUNCTION_GOLD)
STANCES = list(STANCE_GOLD)


def conflicting(label: str, labels: list[str]) -> str:
    """A valid label that differs from `label`, as a disagreeing model prediction."""
    if label not in labels:
        return labels[0]
    return labels[(labels.index(label) + 1) % len(labels)]


def utterance(dialogue, position, text, function="PROPOSE_ANSWER", stance="HEDGED"):
    return {
        "dialogue_id": dialogue,
        "utterance_id": f"{dialogue}_u{position:02d}",
        "utterance_position": str(position),
        "speaker_id": f"{dialogue}_s{position % 3}",
        "text": text,
        "human_communicative_function": function,
        "human_epistemic_stance": stance,
        "gemini_communicative_function": conflicting(function, FUNCTIONS),
        "gemini_epistemic_stance": conflicting(stance, STANCES),
        "communicative_function_match": "False",
        "epistemic_stance_match": "False",
        "annotator_note": f"NOTE-{dialogue}-{position}",
    }


DIALOGUE_A = [
    utterance(
        "dlg_a",
        position,
        f"a{position}: maybe option {position} works here?",
        FUNCTIONS[(position - 1) % len(FUNCTIONS)],
        STANCES[(position - 1) % len(STANCES)],
    )
    for position in range(1, 13)
]
QUOTED = 'I said "2", then, well, 3.\nFinal answer?'
DIALOGUE_B = [
    utterance("dlg_b", 3, QUOTED, "AGREEMENT_ALIGNMENT", "NO_HEDGED"),
    utterance("dlg_b", 1, "b1: hmm", "UNCLEAR", "HEDGED"),
    utterance("dlg_b", 2, "b2: no, option 1 fails", "EVALUATE_CRITIQUE", ""),
]
DIALOGUE_C = [
    utterance("dlg_c", 1, "c1: hi all", "SOCIAL_OFFTASK", "NO_TASK_STANCE"),
    utterance("dlg_c", 2, "   ", "SOCIAL_OFFTASK", "NO_TASK_STANCE"),
    utterance("dlg_c", 3, "", "PROPOSE_ANSWER", "HEDGED"),
]
ROWS = DIALOGUE_A[:5:-1] + DIALOGUE_B + DIALOGUE_A[5::-1] + DIALOGUE_C


def snapshot(root: Path, rows=ROWS, header=HEADER) -> Path:
    path = root / "data" / "utterances.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return root


def convert(rows=ROWS):
    with tempfile.TemporaryDirectory() as tmp:
        return list(source.candidates(snapshot(Path(tmp), rows)))


class DeliChessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = snapshot(Path(self.tmp.name) / "snap")
        self.out = list(source.candidates(self.root))
        self.rows = {row["utterance_id"]: row for row in ROWS}

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_candidates_are_valid(self):
        self.assertEqual(len(self.out), 30)
        for c in self.out:
            self.assertEqual(validate(c, source.SPEC), [])
            self.assertEqual(c.balance_label, c.gold)
            self.assertEqual(c.language, "en")
            self.assertIsNone(c.date)
        self.assertEqual({c.task for c in self.out}, set(source.SPEC.tasks))
        identities = [(c.task, c.source_item_id) for c in self.out]
        self.assertEqual(len(identities), len(set(identities)))

    def test_human_labels_map_to_fixed_criteria_keys(self):
        self.assertEqual(
            list(source.FUNCTION_Q["criteria"]), list(FUNCTION_GOLD.values())
        )
        self.assertEqual(
            list(source.STANCE_Q["criteria"]), ["no_task_stance", "hedged", "unhedged"]
        )
        for c in self.out:
            row = self.rows[c.source_item_id]
            if c.task == FUNCTION:
                self.assertIs(c.question, source.FUNCTION_Q)
                expected = FUNCTION_GOLD[row["human_communicative_function"]]
            else:
                self.assertIs(c.question, source.STANCE_Q)
                expected = STANCE_GOLD[row["human_epistemic_stance"]]
            self.assertEqual(c.gold, expected, c.source_item_id)
        for task, keys in ((FUNCTION, FUNCTION_GOLD), (STANCE, STANCE_GOLD)):
            golds = {c.gold for c in self.out if c.task == task}
            self.assertEqual(golds, set(keys.values()))

    def test_unknown_or_missing_human_label_drops_only_that_task(self):
        self.assertEqual(
            [(c.source_item_id, c.task) for c in self.out if c.group_id == "dlg_b"],
            [
                ("dlg_b_u01", STANCE),
                ("dlg_b_u02", FUNCTION),
                ("dlg_b_u03", FUNCTION),
                ("dlg_b_u03", STANCE),
            ],
        )

    def test_model_columns_and_notes_are_never_read(self):
        rows = source.read_rows(self.root / "data" / "utterances.csv")
        self.assertEqual(len(rows), len(ROWS))
        for row in rows:
            self.assertEqual(set(row), set(source.COLUMNS))
        flipped = [
            dict(
                row,
                gemini_communicative_function=row["human_communicative_function"],
                gemini_epistemic_stance=row["human_epistemic_stance"] or "HEDGED",
                communicative_function_match="True",
                epistemic_stance_match="True",
                annotator_note="another note",
            )
            for row in ROWS
        ]
        self.assertEqual(
            [c.to_json() for c in convert(flipped)], [c.to_json() for c in self.out]
        )
        for c in self.out:
            self.assertNotIn("NOTE-", json.dumps(c.to_json()))
            state = json.dumps(c.state)
            for label in (*FUNCTION_GOLD, *STANCE_GOLD):
                self.assertNotIn(label, state)

    def test_empty_utterances_are_skipped(self):
        self.assertEqual(
            {c.source_item_id for c in self.out if c.group_id == "dlg_c"},
            {"dlg_c_u01"},
        )

    def test_state_is_the_utterance_and_its_fixed_preceding_window(self):
        self.assertLess(source.CONTEXT_TURNS, len(DIALOGUE_A) - 1)
        by_position = {int(r["utterance_position"]): r for r in DIALOGUE_A}
        states = {}
        for c in self.out:
            states.setdefault(c.source_item_id, []).append(c.state)
            if c.group_id != "dlg_a":
                continue
            position = int(self.rows[c.source_item_id]["utterance_position"])
            first = max(1, position - source.CONTEXT_TURNS)
            window = [by_position[p] for p in range(first, position)]
            row = by_position[position]
            self.assertEqual(
                c.state,
                {
                    "preceding_utterances": [
                        {"speaker": r["speaker_id"], "text": r["text"]} for r in window
                    ],
                    "current_utterance": {
                        "speaker": row["speaker_id"],
                        "text": row["text"],
                    },
                },
            )
            self.assertEqual(
                c.overlap_texts, [row["text"], *(r["text"] for r in window)]
            )
        for pair in states.values():
            self.assertEqual(pair[0], pair[-1])
        sizes = {
            int(self.rows[item]["utterance_position"]): len(
                pair[0]["preceding_utterances"]
            )
            for item, pair in states.items()
            if item.startswith("dlg_a_")
        }
        self.assertEqual(
            sizes, {p: min(p - 1, source.CONTEXT_TURNS) for p in range(1, 13)}
        )

    def test_context_stays_within_the_dialogue_in_position_order(self):
        by_id = {c.source_item_id: c.state for c in self.out}
        self.assertEqual(by_id["dlg_b_u01"]["preceding_utterances"], [])
        self.assertEqual(
            [u["text"] for u in by_id["dlg_b_u03"]["preceding_utterances"]],
            ["b1: hmm", "b2: no, option 1 fails"],
        )
        self.assertEqual(by_id["dlg_b_u03"]["current_utterance"]["text"], QUOTED)
        self.assertEqual(by_id["dlg_c_u01"]["preceding_utterances"], [])

    def test_ids_and_groups_follow_utterances_and_dialogues(self):
        for c in self.out:
            self.assertEqual(c.group_id, self.rows[c.source_item_id]["dialogue_id"])
        self.assertEqual(
            list(dict.fromkeys(c.group_id for c in self.out)),
            ["dlg_a", "dlg_b", "dlg_c"],
        )

    def test_oversize_items_are_dropped_never_truncated(self):
        last = source.CONTEXT_TURNS + 3
        rows = [
            utterance("dlg_big", p, "x" * MAX_INPUT_CHARS if p == 2 else f"big{p}: ok")
            for p in range(1, last + 1)
        ]
        out = convert(rows)
        self.assertEqual(
            [(c.source_item_id, c.task) for c in out],
            [
                ("dlg_big_u01", FUNCTION),
                ("dlg_big_u01", STANCE),
                (f"dlg_big_u{last:02d}", FUNCTION),
                (f"dlg_big_u{last:02d}", STANCE),
            ],
        )
        for c in out:
            window = [u["text"] for u in c.state["preceding_utterances"]]
            self.assertNotIn("x" * MAX_INPUT_CHARS, window)

        empty = {
            "preceding_utterances": [],
            "current_utterance": {"speaker": "dlg_fit_s1", "text": ""},
        }
        self.assertGreater(
            input_chars(empty, source.FUNCTION_Q), input_chars(empty, source.STANCE_Q)
        )
        room = MAX_INPUT_CHARS - input_chars(empty, source.STANCE_Q)
        fit = convert([utterance("dlg_fit", 1, "y" * room)])
        self.assertEqual([c.task for c in fit], [STANCE])
        self.assertEqual(fit[0].state["current_utterance"]["text"], "y" * room)
        self.assertEqual(input_chars(fit[0].state, fit[0].question), MAX_INPUT_CHARS)
        self.assertEqual(convert([utterance("dlg_fit", 1, "y" * (room + 1))]), [])

    def test_missing_human_column_fails_loudly(self):
        header = [name for name in HEADER if name != "human_epistemic_stance"]
        with tempfile.TemporaryDirectory() as tmp:
            root = snapshot(Path(tmp), header=header)
            with self.assertRaises(ValueError):
                list(source.candidates(root))

    def test_runner_counts(self):
        receipt = runner.run("delichess", self.root, None)
        self.assertEqual(receipt["invalid"], {})
        self.assertEqual(receipt["valid"], 30)
        for task in source.SPEC.tasks:
            self.assertEqual(receipt["tasks"][task]["candidates"], 15)
            self.assertEqual(receipt["tasks"][task]["groups"], 3)
        self.assertEqual(receipt["tasks"][STANCE]["label:unhedged"], 5)


if __name__ == "__main__":
    unittest.main()
