"""Offline regressions for the Vela checkpoint label identity check."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


RUNNER_PATH = Path(__file__).parents[1] / "vela" / "run-vela.py"
SPEC = importlib.util.spec_from_file_location("vela_runner", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import failure
    raise ImportError(f"cannot load Vela runner from {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

MAPPING = {"biology": 0, "business": 1, "chemistry": 2}


class ModelConfig:
    def __init__(self, id2label: object) -> None:
        self.id2label = id2label


class TestLabelIdentity(unittest.TestCase):
    def test_integer_keys_are_normalized_and_checked(self) -> None:
        config = ModelConfig({0: "biology", 1: "business", 2: "chemistry"})

        RUNNER.validate_label_identity(config, MAPPING)

    def test_string_keys_are_normalized_and_checked(self) -> None:
        config = ModelConfig({"0": "biology", "1": "business", "2": "chemistry"})

        RUNNER.validate_label_identity(config, MAPPING)

    def test_permuted_semantic_labels_fail(self) -> None:
        config = ModelConfig({0: "business", 1: "biology", 2: "chemistry"})

        with self.assertRaisesRegex(ValueError, "ordering differs"):
            RUNNER.validate_label_identity(config, MAPPING)

    def test_partial_label_map_fails(self) -> None:
        config = ModelConfig({"0": "biology", "1": "business"})

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            RUNNER.validate_label_identity(config, MAPPING)

    def test_generic_transformers_labels_are_allowed(self) -> None:
        config = ModelConfig({"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"})

        RUNNER.validate_label_identity(config, MAPPING)


if __name__ == "__main__":
    unittest.main()
