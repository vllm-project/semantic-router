"""Evidence lookup used by the Recipe Lab handoff."""

import importlib.util
import json
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "casebook", LAB / "recipe_lab/casebook.py"
)
assert spec is not None and spec.loader is not None
casebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(casebook)


class CasebookTest(unittest.TestCase):
    def test_symptom_lookup(self):
        for query, expected in [
            (
                "tool turns stay on the strong model",
                ["e7b-tool-progress", "vela-execution-phase"],
            ),
            ("long stream repeatedly disconnects", ["active-stream-deadline"]),
            ("SEMANTIC-EXECUTION-PHASE", ["vela-execution-phase"]),
            ("Response is missing [DONE]", ["active-stream-deadline"]),
            ("DSH ended normally but submitted no SQL", []),
            ("model", []),
            ([], []),
        ]:
            with self.subTest(query=query):
                self.assertEqual(
                    [record["id"] for record in casebook.search_cases(query)], expected
                )

    def test_localized_symptoms_match_inside_longer_queries(self):
        path = LAB / "recipe_lab/locales/zh-CN/casebook-search.json"
        aliases = json.loads(path.read_text(encoding="utf-8"))
        cases = {
            "tool_history_concentrated_on_strong": [
                "e7b-tool-progress",
                "vela-execution-phase",
            ],
            "semantic_execution_phase": ["vela-execution-phase"],
            "long_stream_retry": ["active-stream-deadline"],
            "active_stream_cutoff": ["active-stream-deadline"],
            "missing_done": ["active-stream-deadline"],
            "tool_window_expansion": ["tool-window-routing-boundary"],
            "same_backend_different_parameters": ["tool-window-routing-boundary"],
            "strict_failure_native_output_limit": [
                "native-output-limit-on-strict-failure"
            ],
            "native_output_limit": ["native-output-limit-on-strict-failure"],
        }
        self.assertEqual(aliases.keys(), cases.keys())
        for tag, phrases in aliases.items():
            self.assertTrue(phrases, tag)
            for phrase in phrases:
                with self.subTest(tag=tag, phrase=phrase):
                    matches = casebook.search_cases(f"Observed: {phrase}.")
                    self.assertEqual([row["id"] for row in matches], cases[tag])

    def test_evidence_and_retest_conditions_survive_lookup(self):
        cards = casebook.search_cases(
            [
                "tool_history_concentrated_on_strong",
                "semantic_execution_phase",
                "missing_done",
            ]
        )
        self.assertEqual(len(cards), 3)
        self.assertEqual(len({card["id"] for card in cards}), 3)
        self.assertEqual(cards[0]["id"], "vela-execution-phase")
        for card in cards:
            for key in (
                "config_version",
                "applies_when",
                "change",
                "evidence",
                "limitations",
                "verify",
                "sources",
            ):
                self.assertTrue(card[key], (card["id"], key))
            # Sources owned by later slices (the depth draft, the planning
            # space and the development decision summary) stay citable before
            # their files land.
            later = ("speed_draft", "planning_space", "signature", "decision_summary")
            for path in card["sources"]:
                if any(name in path for name in later):
                    continue
                self.assertTrue((LAB.parents[1] / path).is_file(), path)
