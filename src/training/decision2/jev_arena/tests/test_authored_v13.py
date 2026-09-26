"""Pure-CPU contract tests for the v13 authoring gates; no private cases."""

import copy
import unittest

from jev_arena.authored_v13_audit import compare, prompt_text
from jev_arena.authored_v13_pilot import render_case, solve, validate_case


class V13OracleTest(unittest.TestCase):
    def test_twelve_distinct_operation_oracles(self) -> None:
        examples = (
            (
                "eligible_quote",
                {"A": 4, "B": 8},
                {"A": 2, "B": 1},
                {"required_capacity": 5, "priority": ["A", "B"]},
                "B",
            ),
            ("rank_aggregation", ["A", "B"], ["B", "A"], {"priority": ["B", "A"]}, "B"),
            (
                "window_overlap",
                {"start": 10, "end": 20},
                {"A": {"start": 0, "end": 14}, "B": {"start": 17, "end": 30}},
                {"priority": ["A", "B"]},
                "A",
            ),
            (
                "revision_chain",
                [{"rev": 1, "action": "GO"}, {"rev": 2, "action": "STOP"}],
                {"1": "approved", "2": "approved"},
                {"actions": ["GO", "STOP"]},
                "REVIEW",
            ),
            (
                "net_range",
                {"gross": 15},
                {"tare": 3},
                {"minimum": 10, "maximum": 12},
                True,
            ),
            ("required_subset", ["A"], ["A", "B"], {}, True),
            (
                "interval_exclusion",
                {"start": 2, "end": 4},
                [{"start": 4, "end": 7}],
                {},
                True,
            ),
            (
                "latest_ack",
                [{"id": "a", "time": 5}, {"id": "b", "time": 9}],
                {"a": 6, "b": 11},
                {"maximum_minutes": 1},
                False,
            ),
            (
                "residual_risk",
                {"a": 3, "b": 1},
                {"a": False, "b": True},
                {"limits": [1, 2]},
                0,
            ),
            (
                "utilization_band",
                {"capacity": 10},
                {"demand": 7},
                {"ratio_limits": [0.5, 0.9]},
                1,
            ),
            ("median_divergence", [1, 2, 3], [3, 4, 5], {"limits": [1, 2]}, 1),
            (
                "critical_path",
                {"a": 1, "b": 2, "c": 3},
                [["a", "b"], ["b", "c"]],
                {"limits": [3, 5]},
                0,
            ),
        )
        for name, left, right, params, expected in examples:
            with self.subTest(operation=name):
                self.assertEqual(solve(name, left, right, params), expected)

    def sample_case(self) -> dict:
        return {
            "slug": "synthetic-unit-only",
            "operation": "net_range",
            "domain": "unit-only",
            "scene": "A synthetic measurement arrives from two independent registers for one item.",
            "contract": "Subtract the tare from gross and compare the inclusive result against the stated interval.",
            "question": "Does the measured item satisfy the stated synthetic inclusive interval under the rule?",
            "criteria": {"true": "Inside", "false": "Outside"},
            "params": {"minimum": 10, "maximum": 18},
            "sources": [
                {
                    "side": "left",
                    "title": "Synthetic gross register",
                    "form": "ticket",
                    "data": {"gross": 20},
                },
                {
                    "side": "right",
                    "title": "Synthetic tare register",
                    "form": "ledger",
                    "data": {"tare": 5},
                },
            ],
            "witnesses": {
                "left": [{"gross": 18}, {"gross": 10}],
                "right": [{"tare": 2}, {"tare": 12}],
            },
            "variant": {"side": "left", "data": {"gross": 30}},
            "variant_witnesses": {"right": [{"tare": 13}, {"tare": 25}]},
        }

    def test_original_variant_and_both_source_witnesses(self) -> None:
        case = self.sample_case()
        proof = validate_case(case)
        self.assertIs(proof["original"], True)
        self.assertIs(proof["variant"], False)
        self.assertEqual(set(proof["witness_answers"]), {"left", "right"})
        self.assertEqual(set(proof["variant_witness_answers"]), {"left", "right"})
        prompt = render_case(case, case["sources"], "opaque")
        self.assertEqual(prompt["questions"]["decision"]["type"], "noul")
        self.assertIn("gross", prompt["state"])
        self.assertNotIn("synthetic-unit-only", prompt["state"])

    def test_redundant_substituted_source_is_rejected(self) -> None:
        case = self.sample_case()
        case["variant_witnesses"]["right"] = [{"tare": 1}, {"tare": 2}]
        with self.assertRaisesRegex(ValueError, "unnecessary"):
            validate_case(case)

    def test_wrong_native_criteria_are_rejected(self) -> None:
        case = copy.deepcopy(self.sample_case())
        case["criteria"] = ["low", "middle", "high"]
        with self.assertRaises(ValueError):
            render_case(case, case["sources"], "opaque")

    def test_prompt_only_overlap_metrics(self) -> None:
        row = {
            "state": "One evidence phrase",
            "questions": {"decision": {"instructions": "Answer from evidence"}},
            "gold": {"hidden": "unused"},
        }
        self.assertEqual(prompt_text(row), "One evidence phrase\nAnswer from evidence")
        self.assertTrue(compare("a b c d", "a b c d")["exact"])
        self.assertLess(compare("a b c d", "w x y z")["trigram_jaccard"], 0.7)

    def test_utilization_boundaries_use_exact_decimal_arithmetic(self) -> None:
        params = {"ratio_limits": [0.6, 0.9]}
        self.assertEqual(
            solve("utilization_band", {"capacity": 10}, {"demand": 6}, params), 2
        )
        self.assertEqual(
            solve("utilization_band", {"capacity": 10}, {"demand": 9}, params), 1
        )
        self.assertEqual(
            solve("utilization_band", {"capacity": 10}, {"demand": 10}, params), 0
        )


if __name__ == "__main__":
    unittest.main()
