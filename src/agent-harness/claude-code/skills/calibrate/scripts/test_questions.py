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
    options: ClassVar[dict] = {"routerEasyBelow": 0.5, "routerHardFrom": 1.5}

    def ask(self, d):
        return questions.question(d, self.options)

    def test_every_app_has_a_question_whose_answers_are_labels(self):
        self.assertEqual(set(questions.QUESTIONS), set(APPS))
        for app in APPS:
            with self.subTest(app=app):
                d = Decision("x", None, "s", app, "acted", 0.5, 0.5, "subject", {})
                entry = self.ask(d)
                self.assertLessEqual(len(entry["header"]), 12)
                allowed = ROUTER_LABELS if app == "router" else ("yes", "no")
                self.assertLessEqual(
                    {o["value"] for o in entry["options"]}, set(allowed)
                )

    def test_a_routed_task_is_asked_in_plain_words(self):
        entry = self.ask(route(1, 1.55, "claude-opus-4"))
        self.assertEqual(
            entry["question"],
            'A subagent was given the task "task 1". vSR judged it hard and sent it to Opus. Was that the right model?',
        )
        self.assertEqual(
            [(o["label"], o["value"]) for o in entry["options"]],
            [("Right model", "right"), ("Could be lighter", "too-high")],
        )
        sonnet = self.ask(route(2, 1.2, "sonnet"))
        self.assertIn("judged it multi-step and sent it to Sonnet.", sonnet["question"])
        self.assertEqual(
            [o["label"] for o in sonnet["options"]],
            ["Right model", "Needed stronger", "Could be lighter"],
        )
        self.assertEqual(
            [o["value"] for o in sonnet["options"]], ["right", "too-low", "too-high"]
        )
        haiku = self.ask(route(3, 0.3, "haiku"))
        self.assertIn("judged it easy and sent it to Haiku.", haiku["question"])
        self.assertEqual([o["value"] for o in haiku["options"]], ["right", "too-low"])

    def test_a_kept_task_names_the_session_model(self):
        kept = self.ask(route(2, 0.3, "opus", 0.2, verdict="kept"))
        self.assertIn(
            "vSR judged it easy but was not sure enough, so the subagent kept your session's model.",
            kept["question"],
        )
        self.assertEqual(
            [o["value"] for o in kept["options"]], ["haiku", "sonnet", "opus"]
        )

    def test_no_question_shows_a_score(self):
        for app in APPS:
            with self.subTest(app=app):
                d = Decision("x", None, "s", app, "acted", 0.93, 0.9, "subject", {})
                text = self.ask(d)["question"]
                self.assertNotIn("0.93", text)
                self.assertNotIn("score", text)
                self.assertNotIn("difficulty", text)


if __name__ == "__main__":
    unittest.main()
