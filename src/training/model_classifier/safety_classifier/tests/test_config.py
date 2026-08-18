import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.training.model_classifier.safety_classifier.config import (
    ContractError,
    contract_sha256,
    distributed_batch_parameters,
    load_contract,
)
from src.training.model_classifier.safety_classifier.train import (
    synchronize_best_checkpoint,
    synchronized_checkpoint_trainer,
)


class ContractTest(unittest.TestCase):
    def test_checked_in_contract_is_valid_and_stable(self):
        contract = load_contract()
        self.assertEqual(contract["taxonomy_version"], "legacy-9-v1")
        self.assertEqual(len(contract_sha256(contract)), 64)
        self.assertEqual(
            contract["base_model"]["revision"],
            "72a23a6640489471eb4ff7ad3ec5bc80af8a27de",
        )
        self.assertIs(contract["model"]["reference_compile"], False)

    def test_eight_gpu_batch_matches_historical_global_batch(self):
        contract = load_contract()
        self.assertEqual(distributed_batch_parameters(contract, 8), (8, 1))
        self.assertEqual(distributed_batch_parameters(contract, 1), (8, 8))

    def test_invalid_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text('{"contract_version": 2}', encoding="utf-8")
            with self.assertRaises(ContractError):
                load_contract(path)


class CheckpointSynchronizationTest(unittest.TestCase):
    def _fake_torch(self):
        barrier_calls = []
        distributed = SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            barrier=lambda **kwargs: barrier_calls.append(kwargs),
        )
        cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 3)
        return SimpleNamespace(distributed=distributed, cuda=cuda), barrier_calls

    def test_all_ranks_derive_the_best_checkpoint_after_the_save_fence(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-7"
            checkpoint.mkdir()
            trainer = SimpleNamespace(
                state=SimpleNamespace(
                    best_global_step=7,
                    best_model_checkpoint=None,
                ),
                _get_output_dir=lambda _trial: directory,
            )
            torch, barrier_calls = self._fake_torch()

            synchronize_best_checkpoint(trainer, torch, None)

            self.assertEqual(trainer.state.best_model_checkpoint, str(checkpoint))
            self.assertEqual(barrier_calls, [{"device_ids": [3]}])

    def test_missing_best_checkpoint_fails_closed_after_the_fence(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(
                state=SimpleNamespace(
                    best_global_step=9,
                    best_model_checkpoint=None,
                ),
                _get_output_dir=lambda _trial: directory,
            )
            torch, barrier_calls = self._fake_torch()

            with self.assertRaisesRegex(FileNotFoundError, "checkpoint-9"):
                synchronize_best_checkpoint(trainer, torch, None)

            self.assertEqual(barrier_calls, [{"device_ids": [3]}])

    def test_no_best_step_still_fences_distributed_ranks(self):
        trainer = SimpleNamespace(
            state=SimpleNamespace(
                best_global_step=None,
                best_model_checkpoint=None,
            )
        )
        torch, barrier_calls = self._fake_torch()

        synchronize_best_checkpoint(trainer, torch, None)

        self.assertIsNone(trainer.state.best_model_checkpoint)
        self.assertEqual(barrier_calls, [{"device_ids": [3]}])

    def test_uninitialized_process_group_does_not_call_barrier(self):
        distributed = SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: False,
            barrier=lambda **_kwargs: self.fail("barrier must not be called"),
        )
        torch = SimpleNamespace(
            distributed=distributed,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                current_device=lambda: self.fail("device must not be queried"),
            ),
        )
        trainer = SimpleNamespace(
            state=SimpleNamespace(
                best_global_step=None,
                best_model_checkpoint=None,
            )
        )

        synchronize_best_checkpoint(trainer, torch, None)

        self.assertIsNone(trainer.state.best_model_checkpoint)

    def test_resume_accepts_an_existing_matching_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_checkpoint = root / "previous" / "checkpoint-9"
            previous_checkpoint.mkdir(parents=True)
            current_output = root / "current"
            current_output.mkdir()
            trainer = SimpleNamespace(
                state=SimpleNamespace(
                    best_global_step=9,
                    best_model_checkpoint=str(previous_checkpoint),
                ),
                _get_output_dir=lambda _trial: str(current_output),
            )
            torch, barrier_calls = self._fake_torch()

            synchronize_best_checkpoint(trainer, torch, None)

            self.assertEqual(
                trainer.state.best_model_checkpoint,
                str(previous_checkpoint),
            )
            self.assertEqual(barrier_calls, [{"device_ids": [3]}])

    def test_resume_rejects_an_existing_checkpoint_for_a_different_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale_checkpoint = root / "previous" / "checkpoint-8"
            stale_checkpoint.mkdir(parents=True)
            current_output = root / "current"
            current_output.mkdir()
            trainer = SimpleNamespace(
                state=SimpleNamespace(
                    best_global_step=9,
                    best_model_checkpoint=str(stale_checkpoint),
                ),
                _get_output_dir=lambda _trial: str(current_output),
            )
            torch, barrier_calls = self._fake_torch()

            with self.assertRaisesRegex(FileNotFoundError, "checkpoint-9"):
                synchronize_best_checkpoint(trainer, torch, None)

            self.assertEqual(barrier_calls, [{"device_ids": [3]}])

    def test_trainer_wrapper_fences_after_the_base_save(self):
        events = []

        class BaseTrainer:
            def __init__(self, output_dir):
                self.output_dir = output_dir
                self.state = SimpleNamespace(
                    best_global_step=5,
                    best_model_checkpoint=None,
                )

            def _get_output_dir(self, _trial):
                return self.output_dir

            def _save_checkpoint(self, _model, _trial):
                events.append("save")
                (Path(self.output_dir) / "checkpoint-5").mkdir()

        torch, barrier_calls = self._fake_torch()
        trainer_class = synchronized_checkpoint_trainer(BaseTrainer, torch)
        with tempfile.TemporaryDirectory() as directory:
            trainer = trainer_class(directory)
            trainer._save_checkpoint(None, None)

        self.assertEqual(events, ["save"])
        self.assertEqual(barrier_calls, [{"device_ids": [3]}])
        self.assertTrue(trainer.state.best_model_checkpoint.endswith("checkpoint-5"))


if __name__ == "__main__":
    unittest.main()
