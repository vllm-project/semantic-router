"""Validate the public artifact-to-training-source index."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRAINING_ROOT.parents[1]


class ModelArtifactIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (TRAINING_ROOT / "model_artifacts.json").read_text(encoding="utf-8")
        )

    def test_collection_counts_and_unique_artifacts(self) -> None:
        collections = self.manifest["collections"]
        self.assertEqual(len(collections["mom-multilingual-embed"]), 5)
        self.assertEqual(len(collections["mom-multilingual-class"]), 14)
        artifacts = [
            item["artifact"]
            for collection in collections.values()
            for item in collection
        ]
        self.assertEqual(len(artifacts), len(set(artifacts)))

    def test_every_mapping_resolves_inside_training_tree(self) -> None:
        for collection in self.manifest["collections"].values():
            for item in collection:
                with self.subTest(artifact=item["artifact"]):
                    owner = REPOSITORY_ROOT / item["owner"]
                    self.assertTrue(owner.is_dir(), owner)
                    self.assertTrue((owner / item["entrypoint"]).is_file())
                    if "config" in item:
                        self.assertTrue((owner / item["config"]).is_file())

    def test_index_contains_only_current_ownership(self) -> None:
        serialized = json.dumps(self.manifest)
        self.assertNotIn('"upstream"', serialized)
        self.assertNotIn('"producer"', serialized)
        self.assertNotIn('"source_blob"', serialized)

    def test_safety_binary_and_hazard_are_owned(self) -> None:
        safety = {
            item["artifact"]: item
            for item in self.manifest["collections"]["mom-multilingual-class"]
            if "safety" in item["artifact"]
        }
        self.assertEqual(
            set(safety),
            {
                "llm-semantic-router/mmbert-safety-binary-merged",
                "llm-semantic-router/mmbert-safety-binary-hazard",
            },
        )
        self.assertEqual(
            {item["task"] for item in safety.values()}, {"level1", "level2"}
        )
        self.assertEqual({item["artifact_type"] for item in safety.values()}, {"lora"})
        self.assertEqual(
            {item["artifact_base_model"] for item in safety.values()},
            {"jhu-clsp/mmBERT-base"},
        )


if __name__ == "__main__":
    unittest.main()
