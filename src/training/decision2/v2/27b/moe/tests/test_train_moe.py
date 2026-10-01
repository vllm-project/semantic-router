"""The MoE trainer wrapper picks the pin and the prompt version without touching train.py."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class TrainMoE(unittest.TestCase):
    def test_resolve_by_base_and_by_resume_checkpoint(self):
        from training.model.decision_model import BOS_PROMPT_VERSION, PROMPT_VERSION

        wrapper = importlib.import_module("v2.27b.moe.train_moe")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, kind in (("g", "gemma4"), ("q", "qwen3_5_moe"), ("d", "qwen3_5")):
                (root / name).mkdir()
                (root / name / "config.json").write_text(
                    json.dumps({"model_type": kind})
                )
            argv = ["--experts-implementation", "grouped_mm", "--output", "/o"]
            experts, version, rest = wrapper.resolve(
                [*argv, "--model-path", str(root / "g")]
            )
            self.assertEqual((experts, version), ("grouped_mm", BOS_PROMPT_VERSION))
            self.assertNotIn("--experts-implementation", rest)
            _, version, _ = wrapper.resolve([*argv, "--model-path", str(root / "q")])
            self.assertEqual(version, PROMPT_VERSION)
            with self.assertRaises(SystemExit):
                wrapper.resolve([*argv, "--model-path", str(root / "d")])
            (root / "ck").mkdir()
            (root / "ck" / "decision_config.json").write_text(
                json.dumps(
                    {
                        "prompt_version": BOS_PROMPT_VERSION,
                        "experts_implementation": "eager",
                    }
                )
            )
            with self.assertRaises(SystemExit):
                wrapper.resolve([*argv, "--resume", str(root / "ck")])

    def test_trainer_bytes_are_the_archived_control(self):
        from training.model import train
        from training.model.data import file_sha256

        self.assertEqual(
            file_sha256(Path(train.__file__)),
            "0111bbfd0e6a372661a88c44e0c716c18a94b78f651c0662fa2b18bbe59ad96d",
        )


if __name__ == "__main__":
    unittest.main()
