"""Dependency-light checks for the Transformers training argument boundary."""

import importlib.util
import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "model_classifier" / "training_args_compat.py"
)
SPEC = importlib.util.spec_from_file_location("training_args_compat", MODULE_PATH)
COMPAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPAT)


class LegacyArguments:
    def __init__(self, *, warmup_ratio=0.0, warmup_steps=0, logging_dir=None):
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.logging_dir = logging_dir

    def get_warmup_steps(self, total):
        return self.warmup_steps or math.ceil(total * self.warmup_ratio)


class CurrentArguments:
    def __init__(self, *, warmup_steps=0):
        self.warmup_steps = warmup_steps

    def get_warmup_steps(self, total):
        return (
            int(self.warmup_steps)
            if self.warmup_steps >= 1
            else math.ceil(total * self.warmup_steps)
        )


class TrainingArgumentsCompatibilityTest(unittest.TestCase):
    def test_fractional_warmup_preserves_schedule_across_apis(self):
        for arguments_type in (LegacyArguments, CurrentArguments):
            for ratio in (0, 0.06, 0.1):
                with self.subTest(api=arguments_type.__name__, ratio=ratio):
                    args = COMPAT.create_training_arguments(
                        arguments_type, warmup_ratio=ratio
                    )
                    self.assertEqual(args.get_warmup_steps(101), math.ceil(101 * ratio))

    def test_explicit_steps_still_take_precedence(self):
        for arguments_type in (LegacyArguments, CurrentArguments):
            args = COMPAT.create_training_arguments(
                arguments_type, warmup_ratio=0.1, warmup_steps=25
            )
            self.assertEqual(args.get_warmup_steps(1000), 25)

    def test_logging_directory_uses_supported_contract(self):
        with patch.dict(os.environ, {}, clear=True):
            old = COMPAT.create_training_arguments(
                LegacyArguments, logging_dir="/tmp/old-training-logs"
            )
            self.assertEqual(old.logging_dir, "/tmp/old-training-logs")
            self.assertNotIn("TENSORBOARD_LOGGING_DIR", os.environ)
            COMPAT.create_training_arguments(
                CurrentArguments, logging_dir="/tmp/new-training-logs"
            )
            self.assertEqual(
                os.environ["TENSORBOARD_LOGGING_DIR"], "/tmp/new-training-logs"
            )

    def test_invalid_ratio_does_not_silently_change_schedule(self):
        for ratio in (-0.1, 1, 1.1):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                COMPAT.create_training_arguments(CurrentArguments, warmup_ratio=ratio)

    def test_unrelated_invalid_arguments_remain_errors(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(TypeError):
                COMPAT.create_training_arguments(
                    CurrentArguments, unknown_option=True, logging_dir="/tmp/logs"
                )
            self.assertNotIn("TENSORBOARD_LOGGING_DIR", os.environ)


if __name__ == "__main__":
    unittest.main()
