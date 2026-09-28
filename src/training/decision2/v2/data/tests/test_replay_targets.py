from __future__ import annotations

import unittest

from training.model.data import INPUT_FIELDS, digest
from v2.data.build_a0_variants import native_prompt
from v2.data.replay_targets import collector_digest, convert, repeat_max_abs_diff
from v2.data.tests.test_build_a0_variants import _row

MODEL, REV = "org/teacher", "abc123"


def _receipt(prompt, answer, **overrides):
    receipt = {
        "id": prompt["id"],
        "answers": {"decision": answer},
        "model_id": MODEL,
        "model_revision": REV,
        "revision_attested": True,
        "runtime_matches_validated": True,
        "source_input_sha256": collector_digest(prompt),
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
        self.noul["state"] = {"zeta": "z", "alpha": "a"}
        self.noul["input_sha256"] = digest(
            {field: self.noul[field] for field in INPUT_FIELDS}
        )
        self.rows = [self.choice, self.noul, self.score]
        self.prompts = [native_prompt(row) for row in self.rows]
        self.answers = {
            self.choice["id"]: {
                "type": "choice",
                "probabilities": {"k0": 0.1, "k1": 0.6, "k2": 0.2, "k3": 0.1},
            },
            self.noul["id"]: {"type": "noul", "noul": 0.25},
            self.score["id"]: {"type": "score", "error": "context_overflow"},
        }

    def _receipts(self, **overrides):
        return [_receipt(p, self.answers[p["id"]], **overrides) for p in self.prompts]

    def test_converts_types_and_skips_unanswered(self) -> None:
        sorted_rows = [dict(row) for row in self.rows]
        sorted_rows[1]["state"] = {"alpha": "a", "zeta": "z"}
        replay, report = convert(
            sorted_rows,
            self.prompts,
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
        self.assertEqual(report["score"]["no_answer_reasons"], {"context_overflow": 1})
        self.assertEqual(report["choice"]["argmax_accuracy_vs_gold"], 1.0)

    def test_rejects_identity_prompt_or_runtime_mismatch(self) -> None:
        for overrides in (
            {"model_revision": "other"},
            {"revision_attested": False},
            {"runtime_matches_validated": False},
            {"source_input_sha256": "0" * 64},
        ):
            with self.assertRaises(ValueError):
                convert(
                    self.rows,
                    self.prompts,
                    self._receipts(**overrides),
                    teacher="t",
                    model_id=MODEL,
                    revision=REV,
                )

    def test_rejects_prompt_that_differs_from_training_row(self) -> None:
        prompts = [dict(p) for p in self.prompts]
        prompts[0] = {**prompts[0], "state": "tampered"}
        with self.assertRaises(ValueError):
            convert(
                self.rows,
                prompts,
                self._receipts(),
                teacher="t",
                model_id=MODEL,
                revision=REV,
            )

    def test_repeat_diff(self) -> None:
        full = self._receipts()
        again = self._receipts()
        again[0] = _receipt(
            self.prompts[0],
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
