"""Lightweight source and configuration contracts for the compact producer."""

from __future__ import annotations

import ast
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.training.model_embeddings.multimodal.small.runtime import load_config

ROOT = Path(__file__).resolve().parents[1]


class ArtifactContractTest(unittest.TestCase):
    def test_manifest_maps_artifact_to_local_entrypoints(self) -> None:
        manifest = json.loads((ROOT / "artifacts.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["owner"],
            "src/training/model_embeddings/multimodal/small",
        )
        self.assertNotIn("producer", manifest)
        self.assertNotIn("upstream", manifest)
        artifact = manifest["artifacts"][0]
        self.assertEqual(
            artifact["repo_id"],
            "llm-semantic-router/multi-modal-embed-small",
        )
        for key in (
            "config",
            "train_entrypoint",
            "evaluate_entrypoint",
            "release_entrypoint",
        ):
            self.assertTrue((ROOT / artifact[key]).is_file(), key)

    def test_production_config_matches_published_family(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "production.yaml").read_text(encoding="utf-8")
        )
        model = config["model"]
        self.assertEqual(
            model["text_encoder_name"], "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.assertEqual(model["image_encoder_name"], "google/siglip-base-patch16-512")
        self.assertEqual(model["audio_encoder_name"], "openai/whisper-tiny")
        for key in (
            "text_encoder_revision",
            "image_encoder_revision",
            "audio_encoder_revision",
        ):
            self.assertRegex(model[key], r"^[0-9a-f]{40}$")
        self.assertEqual(model["output_dim"], 384)
        self.assertEqual(model["num_fusion_layers"], 2)
        training = config["training"]
        self.assertEqual(training["batch_size"], 64)
        self.assertEqual(training["num_epochs"], 6)
        self.assertEqual(training["learning_rate"], 1e-4)
        self.assertEqual(training["loss_type"], "matryoshka")
        self.assertEqual(training["matryoshka_dims"], [32, 64, 128, 256, 384])

    def test_config_paths_are_explicit_environment_inputs(self) -> None:
        config_path = ROOT / "configs" / "production.yaml"
        text = config_path.read_text(encoding="utf-8")
        for name in (
            "MM_EMBED_SMALL_TRAIN_CACHE",
            "MM_EMBED_SMALL_VAL_CACHE",
            "MM_EMBED_SMALL_OUTPUT_DIR",
        ):
            self.assertIn(f"${{{name}}}", text)

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "Unresolved environment variable"),
        ):
            load_config(config_path)

    def test_model_card_matches_artifact_architecture(self) -> None:
        text = (ROOT / "model_card_template.md").read_text(encoding="utf-8")
        self.assertIn("MiniLM-L6-v2", text)
        self.assertIn("SigLIP-base-patch16-512", text)
        self.assertIn("Whisper-tiny", text)
        self.assertNotIn("sys.path.append", text)


class SourceLayoutTest(unittest.TestCase):
    def test_python_sources_parse_and_do_not_patch_import_paths(self) -> None:
        for path in ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                ast.parse(source)
                self.assertNotIn("sys.path.insert", source)
                self.assertNotIn("sys.path.append", source)

    def test_training_is_split_across_narrow_owners(self) -> None:
        expected = {
            "data/cached.py",
            "data/sequential.py",
            "models/embedder.py",
            "losses/contrastive.py",
            "runtime.py",
            "stages.py",
            "training.py",
            "runner.py",
            "evaluate.py",
            "release.py",
        }
        actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*.py")}
        self.assertTrue(expected.issubset(actual))

    def test_architecture_load_fails_closed(self) -> None:
        source = (ROOT / "models" / "image_encoder.py").read_text(encoding="utf-8")
        self.assertIn("refusing to substitute a different architecture", source)
        self.assertNotIn("falling back to ViT", source)

    def test_tail_accumulation_is_flushed(self) -> None:
        source = (ROOT / "training.py").read_text(encoding="utf-8")
        self.assertIn("group_end = min(group_start + grad_accum, len(batches))", source)
        self.assertIn("synchronize = batch_index + 1 == group_end", source)


if __name__ == "__main__":
    unittest.main()
