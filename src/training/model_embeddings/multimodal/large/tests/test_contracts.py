"""Lightweight contracts for the consolidated multimodal-large producer."""

from __future__ import annotations

import ast
import json
import os
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class LargeProducerContracts(unittest.TestCase):
    def test_artifact_manifest_maps_to_local_entrypoints(self) -> None:
        manifest = json.loads((ROOT / "artifacts.json").read_text())
        self.assertEqual(
            manifest["owner"],
            "src/training/model_embeddings/multimodal/large",
        )
        artifact = manifest["artifacts"][0]
        self.assertEqual(
            artifact["revision"],
            "e21cde3ccc414c56f504b322662f42c603a939ee",
        )
        for key in (
            "config",
            "train_entrypoint",
            "evaluate_entrypoint",
            "upload_entrypoint",
        ):
            self.assertTrue((ROOT / artifact[key]).is_file(), key)

    def test_production_config_matches_released_hyperparameters(self) -> None:
        text = (ROOT / "configs" / "production.yaml").read_text()
        expected = {
            "text_encoder_name: llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
            "image_encoder_name: google/siglip2-so400m-patch14-384",
            "audio_encoder_name: openai/whisper-medium",
            "embedding_dim: 768",
            "max_text_length: 32768",
            "epochs: 10",
            "batch_size: 12",
            "grad_accum_steps: 8",
            "learning_rate: 1.0e-5",
            "mixed_precision: bf16",
            "type: cached_mnrl",
            "scale: 20.0",
        }
        for line in expected:
            self.assertIn(line, text)

    def test_configs_use_environment_owned_paths(self) -> None:
        for path in (ROOT / "configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            sections = (config, config.get("data", {}), config.get("validation", {}))
            for section in sections:
                for key, value in section.items():
                    if key.endswith(("_dir", "_root", "_path")):
                        self.assertTrue(
                            str(value).startswith("${MM_EMBED_LARGE_"),
                            f"{path.name}:{key}",
                        )

    def test_modules_use_package_imports(self) -> None:
        for path in ROOT.glob("*.py"):
            source = path.read_text()
            ast.parse(source)
            self.assertNotIn("sys.path.insert", source, path.name)
            self.assertNotIn("from hf_st_mm", source, path.name)
            self.assertNotIn("from train_st_multimodal", source, path.name)

    def test_runtime_rejects_unresolved_config_variables(self) -> None:
        source = (ROOT / "runtime.py").read_text()
        self.assertIn("os.path.expandvars", source)
        self.assertIn('if "${" in rendered', source)

    def test_expected_environment_variables_are_documented(self) -> None:
        readme = (ROOT / "README.md").read_text()
        config = (ROOT / "configs" / "production.yaml").read_text()
        for name in (
            "MM_EMBED_LARGE_OUTPUT_DIR",
            "MM_EMBED_LARGE_TRAIN_CACHE",
            "MM_EMBED_LARGE_VAL_CACHE",
        ):
            self.assertIn(name, readme)
            self.assertIn(f"${{{name}}}", config)
            self.assertIsNone(os.environ.get("THIS_TEST_DOES_NOT_SET_" + name))


if __name__ == "__main__":
    unittest.main()
