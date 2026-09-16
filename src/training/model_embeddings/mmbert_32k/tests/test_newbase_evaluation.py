"""Prospective gains cannot override a source/language/length retention failure."""

import copy
import math
import unittest

from src.training.model_embeddings.mmbert_32k.newbase_evaluation import (
    check_constraints,
    ndcg,
    select_candidate,
    spearman,
    summarize_query_scores,
)


class NewBaseEvaluationTest(unittest.TestCase):
    def report(self, task="embedding", gain=0.0):
        full = {
            "miracl": 0.7 + gain,
            "controlled_long": 0.6 + gain,
            "32k_end": 0.5 + gain,
            "qasper": 0.3 + gain,
            "nq": 0.8 + gain,
            "scifact": 0.6 + gain,
            "stsb": 0.8 + gain,
            "miracl_languages": {"ar": 0.65 + gain, "zh": 0.75 + gain},
            "lengths": {"4096": 0.7 + gain, "32768": 0.5 + gain},
        }
        return {
            "development_manifest_sha256": "a" * 64,
            "precision": "float32",
            "run_id": "control" if task == "embedding" else "ranking-control",
            "step": 480,
            "exits": {
                f"{layer}x{dimension}": copy.deepcopy(full)
                for layer in (3, 6, 11, 22)
                for dimension in (768, 512, 256, 128, 64)
            },
        }

    def config(self):
        def term(path, weight=1.0):
            return {"path": ["exits", "22x768", *path], "weight": weight}

        return {
            "precision": "float32",
            "expected_exits": list(self.report()["exits"]),
            "constraints": [
                {
                    "name": "aggregate",
                    "terms": [term(["miracl"])],
                    "baseline_delta": 0.01,
                },
                {
                    "name": "language",
                    "terms": [term(["miracl_languages", "ar"])],
                    "baseline_delta": -0.01,
                },
            ],
            "score_terms": [term(["miracl"], 0.6), term(["qasper"], 0.4)],
            "preferred_runs": ["control", "treatment"],
        }

    def test_ranking_ties_use_document_identity_and_missing_rows_fail(self):
        scores = {"b": 1.0, "a": 1.0, "c": 0.0}
        self.assertEqual(ndcg(scores, {"b": 1.0}), 1 / math.log2(3))
        with self.assertRaisesRegex(ValueError, "positive"):
            ndcg(scores, {"a": 0.0})
        self.assertEqual(ndcg(scores, {"absent": 1.0}, allow_unretrieved=True), 0.0)
        row = {
            "id": "q",
            "source": "miracl",
            "candidate_ids": list(scores),
            "scores": scores,
            "relevance": {"b": 1.0},
        }
        with self.assertRaisesRegex(ValueError, "each frozen query"):
            summarize_query_scores([row], ["q", "missing"])

    def test_spearman_ties_and_undefined_constant_are_explicit(self):
        self.assertEqual(spearman([1.0, 1.0, 2.0], [2.0, 2.0, 4.0]), 1.0)
        self.assertEqual(spearman([1.0, 2.0], [2.0, 1.0]), -1.0)
        with self.assertRaisesRegex(ValueError, "constant"):
            spearman([1.0, 1.0], [1.0, 2.0])

    def test_large_average_gain_cannot_hide_one_language_failure(self):
        baseline, candidate = self.report(), self.report(gain=0.04)
        self.assertTrue(
            check_constraints(candidate, baseline, self.config())[
                "all_constraints_passed"
            ]
        )
        candidate["exits"]["22x768"]["miracl_languages"]["ar"] = 0.63
        result = check_constraints(candidate, baseline, self.config())
        self.assertFalse(result["all_constraints_passed"])
        self.assertFalse(result["checks"]["language"]["passed"])
        self.assertFalse(
            select_candidate([candidate], baseline, self.config())["feasible"]
        )

    def test_selector_ties_keep_earlier_checkpoint_then_control(self):
        baseline = self.report()
        candidate = self.report(gain=0.04)
        treatment = copy.deepcopy(candidate)
        treatment["run_id"] = "treatment"
        late = copy.deepcopy(candidate)
        late["step"] = 960
        result = select_candidate([late, treatment, candidate], baseline, self.config())
        self.assertEqual(result["selected"]["run_id"], "control")
        self.assertEqual(result["selected"]["step"], 480)
        candidate["precision"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "precision"):
            check_constraints(candidate, baseline, self.config())


if __name__ == "__main__":
    unittest.main()
