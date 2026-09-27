import unittest
from types import SimpleNamespace

from training.eikos.loss_profile import (
    FROZEN_DATA_SHA256,
    FROZEN_RELEASE_SHA256,
    FROZEN_RIGHTS_SHA256,
    PROFILE_VERSION,
    loss_weights,
    verify_frozen_treatment,
)


class TestLossProfile(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "id": str(i),
                "source": source,
                "task_type": "score" if i == 5 else "choice",
            }
            for i, source in enumerate(
                (
                    "google_goemotions_official_train",
                    "legacy:cosmos_qa",
                    "legacy:squad2_answerability",
                    "legacy:snli",
                    "css_flute_official_train",
                    "legacy:stage4-general-composition-v2",
                )
            )
        ]

    def test_profile_changes_weights_without_reordering_or_touching_score(self):
        control, control_receipt = loss_weights(self.rows, "uniform")
        treatment, treatment_receipt = loss_weights(self.rows, PROFILE_VERSION)
        self.assertEqual(control, [1.0] * 6)
        self.assertEqual(treatment, [1.5] * 5 + [1.0])
        self.assertEqual(control_receipt["score_rows"], 1)
        self.assertEqual(treatment_receipt["human_rows"], 5)
        self.assertNotEqual(
            control_receipt["roster_sha256"], treatment_receipt["roster_sha256"]
        )

    def test_missing_human_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "human source"):
            loss_weights(self.rows[:-1][:4], PROFILE_VERSION)

    def test_duplicate_id_fails_closed(self):
        rows = self.rows + [dict(self.rows[0])]
        with self.assertRaisesRegex(ValueError, "unique"):
            loss_weights(rows, "uniform")

    def test_treatment_rejects_changed_optimizer_before_model_load(self):
        args = SimpleNamespace(
            seed=20260926,
            max_length=8192,
            microbatch=2,
            accumulation=16,
            eval_batch=4,
            max_steps=232,
            save_every=32,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            learning_rate=2e-5,
            brier_weight=0.25,
            weight_decay=0.01,
            select_limit=None,
        )
        kwargs = dict(
            profile=PROFILE_VERSION,
            args=args,
            data_hashes=FROZEN_DATA_SHA256,
            rights_sha256=FROZEN_RIGHTS_SHA256,
            release_sha256=FROZEN_RELEASE_SHA256,
            admitted_rows=7418,
            admitted_tokens=4402743,
            receipt={
                "human_rows": 3974,
                "score_rows": 516,
                "effective_weight_sum": 9405.0,
            },
        )
        verify_frozen_treatment(**kwargs)
        args.max_steps = 231
        with self.assertRaisesRegex(ValueError, "optimizer"):
            verify_frozen_treatment(**kwargs)


if __name__ == "__main__":
    unittest.main()
