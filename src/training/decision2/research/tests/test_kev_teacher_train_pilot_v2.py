"""Publisher-rounded Kev signal screen stays distinct from the stopped v1."""

from __future__ import annotations

import unittest

from research.kev_teacher_train_pilot_v2 import aggregate, roster, rounded_probabilities


def row(kind: str, index: int, group: str) -> dict:
    keys = (
        ["0", "1", "2"]
        if kind == "score"
        else ["false", "true"] if kind == "noul" else ["a", "b", "c"]
    )
    return {
        "id": f"{kind}-{index}",
        "input_sha256": f"sha-{kind}-{index}",
        "group_id": group,
        "task_type": kind,
        "state": "One stated fact.",
        "instructions": "Choose from the stated fact.",
        "options": [{"key": key, "description": key} for key in keys],
        "label": 0,
    }


class RoundedProbabilityContracts(unittest.TestCase):
    def test_three_option_rounding_is_normalized(self) -> None:
        item = row("choice", 0, "group-0")
        probs = rounded_probabilities(
            item,
            {
                "type": "choice",
                "probabilities": {"a": 0.3333, "b": 0.3333, "c": 0.3333},
            },
        )
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(probs["a"], 1 / 3)

    def test_large_sum_error_or_wrong_keys_fail(self) -> None:
        item = row("choice", 0, "group-0")
        for answer in (
            {"type": "choice", "probabilities": {"a": 0.9, "b": 0.05, "c": 0.0}},
            {"type": "choice", "probabilities": {"a": 1.0, "b": 0.0}},
        ):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                rounded_probabilities(item, answer)

    def test_noul_complement_and_255_option_bound(self) -> None:
        item = row("noul", 0, "group-0")
        self.assertEqual(
            rounded_probabilities(item, {"type": "noul", "noul": 0.3333}),
            {"false": 0.6667000000000001, "true": 0.3333},
        )
        many = row("choice", 0, "group-0")
        many["options"] = [
            {"key": f"key-{index}", "description": "x"} for index in range(255)
        ]
        rounded = {item["key"]: 0.0039 for item in many["options"]}
        self.assertAlmostEqual(
            sum(
                rounded_probabilities(
                    many, {"type": "choice", "probabilities": rounded}
                ).values()
            ),
            1.0,
        )

    def test_new_roster_excludes_v1_groups(self) -> None:
        from research.eikos_teacher_train_pilot import roster as v1_roster

        rows = [
            row(kind, index, f"{kind}-group-{index}")
            for kind in ("choice", "noul", "score")
            for index in range(80)
        ]
        first = v1_roster(rows)
        second = roster(rows)
        self.assertEqual(len(second), 96)
        self.assertFalse(
            {item["group_id"] for item in first} & {item["group_id"] for item in second}
        )

    def test_aggregate_keeps_ties_as_failures(self) -> None:
        item = row("score", 0, "group-0")

        def decide(_state: str, _questions: dict) -> dict:
            return {
                "answers": {
                    "decision": {
                        "type": "score",
                        "probabilities": {"0": 0.3333, "1": 0.3333, "2": 0.3333},
                    }
                }
            }

        result = aggregate([item], decide)["score"]
        self.assertEqual(result["valid"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["correct"], 0)


if __name__ == "__main__":
    unittest.main()
