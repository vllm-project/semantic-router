from __future__ import annotations

import random
import string
import unittest

from v2.eval import leak_audit
from v2.eval.leak_audit import Question


def opaque(rng: random.Random) -> str:
    return "X" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=5))


def construction_order_panel(n: int, positional: bool) -> list[Question]:
    """A7 pattern: gold built as result_3 of 5, then options shuffled for display."""
    rng = random.Random(7)
    out = []
    for i in range(n):
        built = [f"result_{k}" for k in range(5)]
        order = list(range(5))
        rng.shuffle(order)
        keys = [built[k] for k in order]
        gold = order.index(3)
        if positional:
            keys = [f"result_{k}" for k in range(5)]
        out.append(
            Question(
                "p", f"i{i}", "intent", f"c{i}", "choice", keys, ["intent"] * 5, gold
            )
        )
    return out


class LeakAuditTest(unittest.TestCase):
    def test_construction_order_keys_are_a_leak(self):
        questions = construction_order_panel(300, False)
        shuffled = sum(q.keys != sorted(q.keys) for q in questions)
        report = leak_audit.audit_panel(questions, 200)
        group = report["groups"]["intent"]
        self.assertEqual(group["combined"]["option_surface"]["verdict"], "LEAK")
        self.assertGreater(group["cues"]["key_number_rank"]["alone"], 95.0)
        self.assertLess(group["label_prior"], 30.0)
        self.assertGreater(shuffled, 290)
        self.assertEqual(group["facts"]["numbered_keys_out_of_display_order"], shuffled)
        self.assertEqual(
            group["facts"]["out_of_order_gold_key_rank"], {"largest_but_one": shuffled}
        )
        self.assertEqual(report["verdict_option_surface"], "LEAK")

    def test_positional_renumbering_is_clean(self):
        report = leak_audit.audit_panel(construction_order_panel(300, True), 200)
        group = report["groups"]["intent"]
        self.assertEqual(group["facts"]["numbered_keys_out_of_display_order"], 0)
        self.assertEqual(group["facts"]["result_n_key_rows"], 300)
        self.assertNotEqual(group["combined"]["option_surface"]["verdict"], "LEAK")
        self.assertLess(
            abs(group["combined"]["option_surface"]["gain_over_reference"]), 5.0
        )

    def test_fixed_order_imbalanced_labels_are_explained_by_the_prior(self):
        rng = random.Random(3)
        keys = ["joy", "anger", "fear"]
        questions = [
            Question(
                "p",
                f"i{i}",
                "emotion",
                f"c{i}",
                "choice",
                keys,
                ["Joy", "Anger", "Fear"],
                rng.choices([0, 1, 2], weights=[6, 3, 1])[0],
            )
            for i in range(400)
        ]
        group = leak_audit.audit_panel(questions, 200)["groups"]["emotion"]
        self.assertGreater(group["label_prior"], 50.0)
        self.assertEqual(group["combined"]["option_surface"]["verdict"], "CLEAN")
        self.assertEqual(group["facts"]["distinct_option_orders"], 1)

    def test_pooled_rows_condition_the_prior_on_the_group(self):
        rng = random.Random(9)
        questions = []
        for task, weights, last in (("a", [8, 1, 1], "C "), ("b", [1, 1, 8], "C")):
            for i in range(300):
                questions.append(
                    Question(
                        "p",
                        f"{task}{i}",
                        task,
                        f"{task}{i}",
                        "choice",
                        ["x", "y", "z"],
                        ["A", "B", last],
                        rng.choices([0, 1, 2], weights=weights)[0],
                    )
                )
        report = leak_audit.audit_panel(questions, 200)
        pooled = report["pooled_by_type"]["choice"]
        self.assertTrue(pooled["prior_conditioned_on_group"])
        self.assertEqual(pooled["combined"]["option_surface"]["verdict"], "CLEAN")
        self.assertEqual(report["verdict_option_surface"], "CLEAN")

    def test_gold_always_displayed_first_is_a_position_leak(self):
        rng = random.Random(5)
        questions = []
        for i in range(300):
            keys = [opaque(rng) for _ in range(4)]
            questions.append(
                Question("p", f"i{i}", "fam", f"c{i}", "choice", keys, ["same"] * 4, 0)
            )
        group = leak_audit.audit_panel(questions, 200)["groups"]["fam"]
        self.assertEqual(group["combined"]["option_surface"]["verdict"], "LEAK")
        self.assertGreater(group["cues"]["position"]["alone"], 95.0)

    def test_state_position_is_reported_separately(self):
        rng = random.Random(11)
        questions = []
        for i in range(300):
            keys = [opaque(rng) for _ in range(4)]
            gold_key = keys[0]
            display = keys[:]
            rng.shuffle(display)
            state = {"offers": [{"id": key} for key in keys]}
            questions.append(
                Question(
                    "p",
                    f"i{i}",
                    "fam",
                    f"c{i}",
                    "choice",
                    display,
                    ["same"] * 4,
                    display.index(gold_key),
                    "",
                    state,
                )
            )
        group = leak_audit.audit_panel(questions, 200)["groups"]["fam"]
        self.assertNotEqual(group["combined"]["option_surface"]["verdict"], "LEAK")
        self.assertEqual(
            group["combined"]["option_and_state_surface"]["verdict"], "LEAK"
        )

    def test_native_and_training_loaders(self):
        noul = leak_audit.native_question(
            "p",
            "a",
            "g",
            "c",
            {
                "type": "noul",
                "instructions": "?",
                "criteria": {"true": "y", "false": "n"},
            },
            False,
            None,
        )
        self.assertEqual((noul.keys, noul.gold), (["true", "false"], 1))
        score = leak_audit.native_question(
            "p",
            "b",
            "g",
            "c",
            {"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]},
            2,
            None,
        )
        self.assertEqual((score.keys, score.gold), (["0", "1", "2"], 2))
        choice = leak_audit.native_question(
            "p",
            "c",
            "g",
            "c",
            {"type": "choice", "instructions": "?", "criteria": {"b": "B", "a": "A"}},
            "a",
            None,
        )
        self.assertEqual((choice.keys, choice.gold), (["b", "a"], 1))
        row = {
            "id": "r",
            "family": "f",
            "group_id": "g",
            "task_type": "choice",
            "options": [
                {"key": "K1", "description": "x"},
                {"key": "K2", "description": "y"},
            ],
            "label": 1,
            "instructions": "?",
            "state": "s",
        }
        question = leak_audit.training_question("select", row)
        self.assertEqual(
            (question.keys, question.gold, question.cluster), (["K1", "K2"], 1, "g")
        )
        self.assertTrue(leak_audit.public_gold("noul", "yes"))
        self.assertFalse(leak_audit.public_gold("noul", "no"))

    def test_markdown_renders(self):
        report = leak_audit.audit_panel(construction_order_panel(60, False), 50)
        report["role"] = "development"
        text = leak_audit.render_markdown({"panels": {"p": report}})
        self.assertIn("| p | development | 60 | LEAK |", text)


if __name__ == "__main__":
    unittest.main()
