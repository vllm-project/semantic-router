import sys
import unittest
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).parent))
import questions
from cases import guard, route, tier_model
from decisions import APPS, ROUTER_LABELS, Decision


class Sampling(unittest.TestCase):
    options: ClassVar[dict] = {
        "guardReview": 0.9,
        "routerEasyBelow": 0.5,
        "routerHardFrom": 1.5,
    }

    def test_closest_on_both_sides(self):
        decisions = [
            guard(i, s)
            for i, s in enumerate([0.5, 0.8, 0.88, 0.89, 0.91, 0.93, 0.99, 0.1])
        ]
        picked = [
            d.score for d in questions.sample(decisions, {}, self.options, ["guard"], 2)
        ]
        self.assertEqual(picked, [0.91, 0.93, 0.89, 0.88])

    def test_skips_labeled_and_scoreless(self):
        decisions = [
            guard(i, s) for i, s in enumerate([0.5, 0.8, 0.88, 0.89, 0.91, 0.93])
        ]
        decisions.append(
            Decision("off1", None, "s", "guard", "off", None, None, "x", {})
        )
        picked = [
            d.score
            for d in questions.sample(
                decisions, {"g3": "no"}, self.options, ["guard"], 2
            )
        ]
        self.assertEqual(picked, [0.91, 0.93, 0.88, 0.8])

    def test_router_uses_the_nearest_cut(self):
        decisions = [
            route(i, s, tier_model(s))
            for i, s in enumerate([0.05, 0.45, 0.55, 1.0, 1.4, 1.6])
        ]
        picked = [
            d.score
            for d in questions.sample(decisions, {}, self.options, ["router"], 1)
        ]
        self.assertEqual(sorted(picked), [0.45, 0.55])


class Questions(unittest.TestCase):
    def test_every_app_has_a_question_whose_answers_are_labels(self):
        self.assertEqual(set(questions.QUESTIONS), set(APPS))
        for app in APPS:
            with self.subTest(app=app):
                d = Decision("x", None, "s", app, "acted", 0.5, 0.5, "subject", {})
                entry = questions.question(d)
                self.assertEqual(entry["header"], app.capitalize())
                allowed = ROUTER_LABELS if app == "router" else ("yes", "no")
                self.assertLessEqual(
                    {o["value"] for o in entry["options"]}, set(allowed)
                )

    def test_a_routed_task_asks_whether_the_model_was_right(self):
        entry = questions.question(route(1, 1.2, "sonnet"))
        self.assertIn("vSR sent it to sonnet (difficulty 1.20 of 2)", entry["question"])
        self.assertEqual(
            [o["value"] for o in entry["options"]], ["too-low", "right", "too-high"]
        )
        kept = questions.question(route(2, 0.3, "opus", 0.2, verdict="kept"))
        self.assertEqual(
            [o["value"] for o in kept["options"]], ["haiku", "sonnet", "opus"]
        )


if __name__ == "__main__":
    unittest.main()
