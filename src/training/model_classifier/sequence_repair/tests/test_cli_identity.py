"""A local checkpoint path must not silently inherit another Hub identity."""

import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.training.model_classifier.safety_classifier import train_vela_hazard
from src.training.model_classifier.sequence_repair import export, initialize, train


class BaseIdentityArgumentTests(unittest.TestCase):
    def test_missing_identity_is_rejected_before_loading_local_artifacts(self):
        common = [
            "tool",
            "--base",
            "unavailable-local-checkpoint",
            "--base-revision",
            "revision",
            "--contract",
            "unavailable-contract",
            "--output",
            "unused-output",
        ]
        training = ["--train", "unused-train", "--dev", "unused-dev"]
        cases = [
            (initialize, []),
            (train, training),
            (train_vela_hazard, training),
            (
                export,
                ["--run-manifest", "unused-run", "--runtime-task", "fact-check"],
            ),
        ]
        dependencies = {
            "torch": SimpleNamespace(),
            "peft": SimpleNamespace(
                LoraConfig=None, TaskType=None, get_peft_model=None
            ),
        }
        for module, arguments in cases:
            stderr = io.StringIO()
            with (
                self.subTest(entrypoint=module.__name__),
                patch.dict(sys.modules, dependencies),
                patch.object(sys, "argv", common + arguments),
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                module.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("required: --base-id", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
