import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier.config import load_contract
from src.training.model_classifier.safety_classifier.release import build_model_card


class ModelCardTest(unittest.TestCase):
    def test_card_distinguishes_reconstruction_and_artifact_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "metrics.json").write_text(
                '{"test":{"test_accuracy":0.75,"test_runtime":1.0}}',
                encoding="utf-8",
            )
            (run_root / "training_manifest.json").write_text(
                '{"source_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                '"global_train_batch_size":64,'
                '"precision":"bf16"}',
                encoding="utf-8",
            )
            card = build_model_card(
                "level1",
                "adapter",
                "example/model-lora",
                run_root,
                load_contract(),
            )
            self.assertIn("new,\ndeterministic reconstruction", card)
            self.assertIn("**adapter** artifact", card)
            self.assertNotIn("test_runtime", card)

    def test_card_refuses_a_mutable_or_missing_source_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "metrics.json").write_text(
                '{"test":{"test_accuracy":0.75}}',
                encoding="utf-8",
            )
            (run_root / "training_manifest.json").write_text(
                '{"source_commit":null,"global_train_batch_size":64,'
                '"precision":"bf16"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "immutable source commit"):
                build_model_card(
                    "level1",
                    "adapter",
                    "example/model-lora",
                    run_root,
                    load_contract(),
                )


if __name__ == "__main__":
    unittest.main()
