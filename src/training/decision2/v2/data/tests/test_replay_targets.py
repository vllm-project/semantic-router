from __future__ import annotations

import unittest

from training.model.data import digest
from v2.data.build_a0_variants import native_prompt
from v2.data.replay_targets import convert, repeat_max_abs_diff
from v2.data.tests.test_build_a0_variants import _row

MODEL, REV = "org/teacher", "abc123"


def _receipt(row, answer, **overrides):
    prompt = native_prompt(row)
    receipt = {
        "id": row["id"],
        "answers": {"decision": answer},
        "model_id": MODEL,
        "model_revision": REV,
        "revision_attested": True,
        "runtime_matches_validated": True,
        "source_input_sha256": digest(
            {"state": prompt["state"], "questions": prompt["questions"]}
        ),
    }
    receipt.update(overrides)
    return receipt


class ReplayTargetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.choice, self.noul, self.score = (
            _row(1, "choice"),
            _row(2, "noul"),
            _row(3, "score"),
        )
        self.answers = {
            self.choice["id"]: {
                "type": "choice",
                "probabilities": {"k0": 0.1, "k1": 0.6, "k2": 0.2, "k3": 0.1},
            },
            self.noul["id"]: {"type": "noul", "noul": 0.25},
            self.score["id"]: None,
        }

    def _receipts(self, **overrides):
        return [
            _receipt(row, self.answers[row["id"]], **overrides)
            for row in (self.choice, self.noul, self.score)
        ]

    def test_converts_types_and_skips_unanswered(self) -> None:
        replay, report = convert(
            [self.choice, self.noul, self.score],
            self._receipts(),
            teacher="t",
            model_id=MODEL,
            revision=REV,
        )
        self.assertEqual(
            [row["id"] for row in replay], [self.choice["id"], self.noul["id"]]
        )
        self.assertEqual(replay[1]["teacher_probs"], {"false": 0.75, "true": 0.25})
        self.assertEqual(report["score"]["no_native_answer"], 1)
        self.assertEqual(report["choice"]["argmax_accuracy_vs_gold"], 1.0)

    def test_rejects_identity_prompt_or_runtime_mismatch(self) -> None:
        rows = [self.choice, self.noul, self.score]
        for overrides in (
            {"model_revision": "other"},
            {"revision_attested": False},
            {"runtime_matches_validated": False},
            {"source_input_sha256": "0" * 64},
        ):
            with self.assertRaises(ValueError):
                convert(
                    rows,
                    self._receipts(**overrides),
                    teacher="t",
                    model_id=MODEL,
                    revision=REV,
                )

    def test_repeat_diff(self) -> None:
        full = self._receipts()
        again = self._receipts()
        again[0] = _receipt(
            self.choice,
            {
                "type": "choice",
                "probabilities": {"k0": 0.1, "k1": 0.55, "k2": 0.25, "k3": 0.1},
            },
        )
        result = repeat_max_abs_diff(full, again)
        self.assertEqual(result["compared"], 2)
        self.assertAlmostEqual(result["max_abs_diff"], 0.05)


if __name__ == "__main__":
    unittest.main()
