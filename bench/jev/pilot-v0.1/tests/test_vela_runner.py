"""Offline regressions for the Vela checkpoint label identity check."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

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

    def test_string_keyed_permutation_fails(self) -> None:
        config = ModelConfig({"0": "business", "1": "biology", "2": "chemistry"})

        with self.assertRaisesRegex(ValueError, "ordering differs"):
            RUNNER.validate_label_identity(config, MAPPING)

    def test_empty_or_invalid_metadata_fails(self) -> None:
        for metadata in ({}, None, [], "LABEL_0"):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(
                ValueError, "non-empty mapping"
            ):
                RUNNER.validate_label_identity(ModelConfig(metadata), MAPPING)

    def test_non_integer_indices_fail(self) -> None:
        for index in (False, 0.25, "not-an-index"):
            metadata = {index: "biology", 1: "business", 2: "chemistry"}
            with self.subTest(index=index), self.assertRaisesRegex(
                ValueError, "non-integer index"
            ):
                RUNNER.validate_label_identity(ModelConfig(metadata), MAPPING)

    def test_duplicate_normalized_indices_fail(self) -> None:
        config = ModelConfig(
            {0: "biology", "0": "biology", 1: "business", 2: "chemistry"}
        )

        with self.assertRaisesRegex(ValueError, "repeats index"):
            RUNNER.validate_label_identity(config, MAPPING)

    def test_invalid_label_values_fail(self) -> None:
        for label in (None, "", 123):
            metadata = {0: label, 1: "business", 2: "chemistry"}
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "invalid label"
            ):
                RUNNER.validate_label_identity(ModelConfig(metadata), MAPPING)

    def test_extra_or_negative_indices_fail(self) -> None:
        for metadata in (
            {0: "biology", 1: "business", 2: "chemistry", 3: "physics"},
            {-1: "biology", 1: "business", 2: "chemistry"},
        ):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(
                ValueError, "exactly cover"
            ):
                RUNNER.validate_label_identity(ModelConfig(metadata), MAPPING)


if __name__ == "__main__":
    unittest.main()
