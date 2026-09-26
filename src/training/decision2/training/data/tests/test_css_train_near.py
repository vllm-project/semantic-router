"""The per-task context audit must match the existing frozen near rule."""

from __future__ import annotations

import unittest

from transfer import build as transfer

from training.data import audit_css_train_near as audit
from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted


class CssTrainNearTests(unittest.TestCase):
    def test_per_task_audit_uses_same_near_pair_rule(self) -> None:
        prompts = [
            {"id": f"css/{task}/0", "state": f"Unique phrase for {task} no overlap"}
            for task in transfer.EVALUATION_TASKS
        ]
        prompts[0][
            "state"
        ] = "The carefully written report contains a meaningful conclusion."
        train = [
            {"id": "train-exact", "state": prompts[0]["state"]},
            {
                "id": "train-other",
                "state": "A completely unrelated input appears here.",
            },
        ]
        result = audit.audit(train, prompts)
        task = transfer.EVALUATION_TASKS[0]
        native = pilot.near_duplicates(
            targeted.context_rows(train), targeted.context_rows(prompts)
        )
        self.assertEqual(result[task]["near_pairs"], native["count"])
        self.assertEqual(result[task]["exact_normalized_css_items"], 1)
        self.assertEqual(result[task]["near_css_items"], 1)
        self.assertEqual(
            sum(row["near_pairs"] for row in result.values()), native["count"]
        )

    def test_non_evaluation_prompt_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit.task_for({"id": "css/semeval_stance/one"})


if __name__ == "__main__":
    unittest.main()
