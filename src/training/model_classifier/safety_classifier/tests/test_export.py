import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier.export import (
    validate_artifact_shape,
)


class ArtifactShapeTest(unittest.TestCase):
    def test_adapter_and_merged_are_distinct_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            merged = root / "merged"
            adapter.mkdir()
            merged.mkdir()
            for filename in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "label_mapping.json",
            ):
                (adapter / filename).write_text("{}", encoding="utf-8")
            for filename in ("config.json", "model.safetensors", "label_mapping.json"):
                (merged / filename).write_text("{}", encoding="utf-8")
            validate_artifact_shape(adapter, "adapter")
            validate_artifact_shape(merged, "merged")

    def test_adapter_rejects_full_model_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for filename in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "label_mapping.json",
                "model.safetensors",
            ):
                (path / filename).write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_artifact_shape(path, "adapter")


if __name__ == "__main__":
    unittest.main()
