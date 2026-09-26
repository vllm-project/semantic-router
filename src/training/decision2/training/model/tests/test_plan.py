import unittest

from training.model.plan import (
    epoch_batches,
    planned_updates,
    replay_count,
    validate_resume_state,
)


class PlanTest(unittest.TestCase):
    def test_replay_sampling_no_train_repeat_and_variable_batch(self):
        lengths = [10 + i for i in range(17)]
        replay_lengths = [25] * 10
        batches = epoch_batches(
            lengths,
            replay_lengths,
            epoch=0,
            seed=29,
            microbatch=3,
            replay_fraction=0.25,
        )
        self.assertEqual(
            batches,
            epoch_batches(
                lengths,
                replay_lengths,
                epoch=0,
                seed=29,
                microbatch=3,
                replay_fraction=0.25,
            ),
        )
        flat = [sample for batch in batches for sample in batch]
        self.assertEqual(
            {sample for sample in flat if sample[0] == "train"},
            {("train", i) for i in range(17)},
        )
        self.assertEqual(len([sample for sample in flat if sample[0] == "train"]), 17)
        self.assertEqual(
            len([sample for sample in flat if sample[0] == "replay"]),
            replay_count(17, 10, 0.25),
        )
        self.assertEqual(min(map(len, batches)), 2)

    def test_multi_epoch_plan_and_exact_cursor(self):
        total = planned_updates(
            17, 10, 0.25, microbatch=3, accumulation=2, epochs=3, max_steps=None
        )
        self.assertEqual(total, 12)
        contract = {
            "planned_updates": 12,
            "train_count": 17,
            "replay_pool_count": 10,
            "replay_fraction": 0.25,
            "microbatch": 3,
            "accumulation": 2,
            "epochs": 3,
        }
        good = {
            "contract": contract,
            "code_sha256": {"train.py": "abc"},
            "step": 2,
            "next_epoch": 0,
            "next_batch": 4,
        }
        validate_resume_state(good, contract, {"train.py": "abc"})
        for mutation, error in (
            ({"step": 3}, "disagree"),
            ({"next_batch": 3}, "boundary"),
            ({"code_sha256": {}}, "source"),
            ({"contract": {}}, "contract"),
        ):
            bad = dict(good, **mutation)
            with self.assertRaisesRegex(ValueError, error):
                validate_resume_state(bad, contract, {"train.py": "abc"})


if __name__ == "__main__":
    unittest.main()
