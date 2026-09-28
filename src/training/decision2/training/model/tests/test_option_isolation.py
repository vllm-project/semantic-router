"""No model weights, benchmark examples or GPU are used in these contracts."""

import hashlib
import math
import unittest

from ..option_isolation import (
    candidate_prompts,
    candidate_token_ids,
    score_independent,
    token_work,
)


def _score(text: str) -> float:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") / 2**32


def _row(task_type: str, count: int) -> dict:
    if task_type == "noul":
        options = [
            {"key": "false", "description": "Not supported"},
            {"key": "true", "description": "Supported"},
        ]
    else:
        options = [
            {"key": str(index), "description": f"Semantic description {index}"}
            for index in range(count)
        ]
    return {
        "state": {"fact": "The shipment arrived Tuesday."},
        "instructions": "Choose the supported answer.",
        "task_type": task_type,
        "options": options,
    }


class TinyTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class OptionIsolationTests(unittest.TestCase):
    def test_candidate_prompt_excludes_other_candidates_and_shares_exact_prefix(
        self,
    ) -> None:
        row = _row("choice", 3)
        prompts = candidate_prompts(row)
        self.assertEqual(len({item.prefix for item in prompts}), 1)
        for index, item in enumerate(prompts):
            self.assertIn(f"Semantic description {index}", item.tail)
            for other in range(3):
                if other != index:
                    self.assertNotIn(f"Semantic description {other}", item.tail)
        encoded = candidate_token_ids(row, TinyTokenizer())
        prefix = TinyTokenizer().encode(prompts[0].prefix, add_special_tokens=False)
        self.assertTrue(all(ids[: len(prefix)] == prefix for _, ids in encoded))

    def test_choice_order_equivariance_at_api_limits(self) -> None:
        for count in (2, 3, 10, 255):
            with self.subTest(count=count):
                row = _row("choice", count)
                original = score_independent(row, _score)
                shuffled = {**row, "options": row["options"][::-1]}
                self.assertEqual(original, score_independent(shuffled, _score))
                self.assertEqual(set(original["probabilities"]), set(range_keys(count)))
                self.assertAlmostEqual(sum(original["probabilities"].values()), 1.0)
                original_text = {item.key: item.text for item in candidate_prompts(row)}
                shuffled_text = {
                    item.key: item.text for item in candidate_prompts(shuffled)
                }
                self.assertEqual(original_text, shuffled_text)

    def test_choice_key_rename_only_when_description_carries_semantics(self) -> None:
        row = _row("choice", 3)
        named = score_independent(row, _score)
        renamed = {
            **row,
            "options": [
                {**option, "key": f"opaque-{index}"}
                for index, option in enumerate(row["options"])
            ],
        }
        other = score_independent(renamed, _score)
        for index in range(3):
            self.assertEqual(
                named["probabilities"][str(index)],
                other["probabilities"][f"opaque-{index}"],
            )
        self.assertEqual(other["choice"], f"opaque-{named['choice']}")
        null_row = {
            **row,
            "options": [{"key": "billing", "description": None}, row["options"][1]],
        }
        self.assertTrue(candidate_prompts(null_row)[0].key_used_as_semantics)
        self.assertIn("billing", candidate_prompts(null_row)[0].text)

    def test_noul_swapped_storage_preserves_true_probability(self) -> None:
        row = _row("noul", 2)
        original = score_independent(row, _score)
        reversed_row = {**row, "options": row["options"][::-1]}
        self.assertEqual(original, score_independent(reversed_row, _score))
        self.assertTrue(0 <= original["noul"] <= 1)

    def test_score_storage_order_not_rubric_order(self) -> None:
        for count in (2, 3, 10):
            with self.subTest(count=count):
                row = _row("score", count)
                original = score_independent(row, _score)
                reversed_row = {**row, "options": row["options"][::-1]}
                self.assertEqual(original, score_independent(reversed_row, _score))
                self.assertTrue(0 <= original["score"] <= count - 1)
                self.assertEqual(original["legend"]["0"], "Semantic description 0")
        row = _row("score", 3)
        swapped_rubric = {
            **row,
            "options": [
                {**row["options"][0], "description": "Semantic description 2"},
                row["options"][1],
                {**row["options"][2], "description": "Semantic description 0"},
            ],
        }
        self.assertNotEqual(
            score_independent(row, _score)["legend"],
            score_independent(swapped_rubric, _score)["legend"],
        )

    def test_ties_and_invalids(self) -> None:
        row = _row("choice", 2)
        self.assertEqual(score_independent(row, lambda _: 0)["choice"], "0")
        self.assertEqual(
            score_independent({**row, "options": row["options"][::-1]}, lambda _: 0)[
                "choice"
            ],
            "0",
        )
        with self.assertRaisesRegex(ValueError, "2..255"):
            candidate_prompts({**row, "options": row["options"] * 128})
        with self.assertRaisesRegex(ValueError, "unique"):
            candidate_prompts({**row, "options": [row["options"][0]] * 2})
        with self.assertRaisesRegex(ValueError, "0..K-1"):
            candidate_prompts(
                {
                    **_row("score", 2),
                    "options": [{**row["options"][0], "key": "x"}, row["options"][1]],
                }
            )
        with self.assertRaisesRegex(ValueError, "2..10"):
            candidate_prompts(_row("score", 11))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_independent(row, lambda _: math.nan)

    def test_token_work_is_counted_not_called_latency(self) -> None:
        for count in (2, 3, 10, 255):
            with self.subTest(count=count):
                costs = token_work(_row("choice", count), TinyTokenizer())
                self.assertEqual(costs["candidate_count"], count)
                self.assertGreater(costs["reference_tokens"], 0)
                self.assertLess(
                    costs["ideal_shared_prefix_tokens"],
                    costs["independent_total_tokens"],
                )
                self.assertGreaterEqual(
                    costs["independent_longest_prompt_tokens"],
                    costs["independent_shortest_prompt_tokens"],
                )
                if count == 255:
                    self.assertGreater(
                        costs["independent_total_tokens"], costs["reference_tokens"]
                    )


def range_keys(count: int) -> list[str]:
    return [str(index) for index in range(count)]


if __name__ == "__main__":
    unittest.main()
