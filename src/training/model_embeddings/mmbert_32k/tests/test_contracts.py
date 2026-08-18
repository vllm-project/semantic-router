"""Lightweight source and configuration contracts for mmBERT-32K training."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

from src.training.model_embeddings.mmbert_32k.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "3bc41e1322ee5a53e08d18eb940855dec53c1539"
TRAINERS = {
    "foundation": ROOT / "foundation.py",
    "embedder": ROOT / "embedder.py",
    "reranker": ROOT / "reranker.py",
}
LAUNCHERS = {
    "foundation": ROOT / "run_rope_training.sh",
    "embedder": ROOT / "run_bge_style_training.sh",
    "reranker": ROOT / "run_rerank_2d_matryoshka_training.sh",
}


def _load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class SourceContractTest(unittest.TestCase):
    def test_one_trainer_per_family_and_valid_python(self) -> None:
        self.assertEqual(set(TRAINERS), {"foundation", "embedder", "reranker"})
        for family, source_path in TRAINERS.items():
            with self.subTest(family=family):
                ast.parse(source_path.read_text(encoding="utf-8"))

    def test_imported_sources_retain_attribution(self) -> None:
        provenance = _load_json("provenance.json")
        self.assertEqual(provenance["upstream"]["commit"], UPSTREAM_COMMIT)
        self.assertEqual(len(provenance["imports"]), 4)
        for item in provenance["imports"]:
            self.assertRegex(item["source_blob"], r"^[0-9a-f]{40}$")
            source = (ROOT / item["destination"]).read_text(encoding="utf-8")
            self.assertIn(UPSTREAM_COMMIT, source[:400])
            self.assertIn(item["source_blob"], source[:400])

    def test_reranker_export_includes_custom_heads(self) -> None:
        source = TRAINERS["reranker"].read_text(encoding="utf-8")
        self.assertIn('"classification_heads.pt"', source)
        self.assertIn('"matryoshka_config.json"', source)
        self.assertIn("weights_only=True", source)

    def test_historical_launcher_names_delegate_to_canonical_runner(self) -> None:
        for family, launcher in LAUNCHERS.items():
            with self.subTest(family=family):
                source = launcher.read_text(encoding="utf-8")
                self.assertTrue(source.startswith("#!/usr/bin/env bash\n"))
                self.assertIn("set -euo pipefail", source)
                self.assertIn("training.model_embeddings.mmbert_32k", source)
                self.assertIn(f"configs/{family}.json", source)
                self.assertNotRegex(source, r"/(?:home|data|scratch|workspace)/")

        self.assertIn("MMBERT32K_STAGE", LAUNCHERS["foundation"].read_text())


class ConfigurationContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.configs = {
            name: _load_json(f"configs/{name}.json")
            for name in ("foundation", "embedder", "reranker")
        }

    def test_configs_are_versioned_and_revision_pinned(self) -> None:
        for family, config in self.configs.items():
            with self.subTest(family=family):
                self.assertEqual(config["schema_version"], 1)
                self.assertEqual(config["family"], family)
                self.assertEqual(config["source_commit"], UPSTREAM_COMMIT)
                self.assertRegex(config["artifact"]["revision"], r"^[0-9a-f]{40}$")
                self.assertRegex(
                    config["train_arguments"]["model_revision"],
                    r"^[0-9a-f]{40}$",
                )

    def test_foundation_extends_8k_to_32k(self) -> None:
        foundation = self.configs["foundation"]
        args = foundation["train_arguments"]
        self.assertEqual(args["model_name_or_path"], "jhu-clsp/mmBERT-base")
        self.assertEqual(args["rope_scaling_type"], "yarn")
        self.assertEqual(args["model_max_length"], 32768)
        self.assertEqual(args["rope_original_max_position_embeddings"], 8192)
        self.assertEqual(
            args["model_max_length"] // args["rope_original_max_position_embeddings"],
            4,
        )
        self.assertEqual(args["attn_implementation"], "sdpa")
        prepare = foundation["prepare_arguments"]
        self.assertEqual(prepare["dataset_name"], "statmt/cc100")
        self.assertEqual(prepare["target_sequence_count"], 30_774)
        self.assertEqual(len(prepare["languages"]), len(prepare["source_etags"]))
        self.assertEqual(
            len(prepare["languages"]), len(prepare["source_content_lengths"])
        )
        self.assertTrue(all(value > 0 for value in prepare["source_content_lengths"]))
        self.assertGreater(prepare["max_document_bytes"], 0)
        self.assertGreater(prepare["max_document_tokens"], 0)
        self.assertTrue(prepare["acknowledge_cc100_license_unknown"])
        self.assertEqual(
            sum(foundation["data_source"]["language_quotas"].values()), 30_774
        )
        self.assertEqual(
            set(prepare["languages"]),
            set(foundation["data_source"]["language_quotas"]),
        )
        self.assertIsNone(foundation["artifact"]["observed_config"]["rope_scaling"])
        self.assertEqual(args["expected_train_samples"], 30_774)
        self.assertEqual(args["packing_languages"], prepare["languages"])
        self.assertEqual(args["packing_source_etags"], prepare["source_etags"])
        self.assertEqual(
            args["packing_source_content_lengths"],
            prepare["source_content_lengths"],
        )
        self.assertEqual(
            args["packing_max_document_bytes"], prepare["max_document_bytes"]
        )
        self.assertEqual(
            args["packing_max_document_tokens"], prepare["max_document_tokens"]
        )
        self.assertTrue(args["acknowledge_cc100_license_unknown"])
        self.assertEqual(
            foundation["data_source"]["release_gate"],
            "blocked-pending-data-governance-review",
        )

    def test_2d_matryoshka_contracts_align(self) -> None:
        embedder = self.configs["embedder"]["train_arguments"]
        reranker = self.configs["reranker"]["train_arguments"]
        self.assertTrue(embedder["use_adaptive_layer"])
        self.assertTrue(embedder["use_matryoshka"])
        self.assertTrue(reranker["use_2d_matryoshka"])
        self.assertEqual(embedder["matryoshka_dims"], reranker["dim_indices"])
        self.assertEqual(reranker["layer_indices"], [3, 6, 11, 22])
        self.assertEqual(embedder["model_revision"], reranker["model_revision"])

    def test_training_data_revisions_are_pinned_and_aligned(self) -> None:
        foundation = self.configs["foundation"]
        self.assertEqual(
            foundation["prepare_arguments"]["dataset_revision"],
            foundation["data_source"]["revision"],
        )

        embedder = self.configs["embedder"]["data_sources"]["bge_m3"]
        reranker = self.configs["reranker"]["data_source"]
        self.assertEqual(embedder["repo_id"], reranker["repo_id"])
        self.assertEqual(embedder["revision"], reranker["revision"])
        self.assertRegex(embedder["revision"], r"^[0-9a-f]{40}$")

    def test_manifests_contain_no_absolute_machine_paths(self) -> None:
        forbidden = re.compile(r"^/(?:data|home|root|workspace)(?:/|$)")

        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)

        manifests = [*self.configs.values(), _load_json("runtime.json")]
        for manifest in manifests:
            for value in strings(manifest):
                self.assertIsNone(forbidden.match(value), value)

    def test_requirements_are_exactly_pinned(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        packages = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertGreater(len(packages), 5)
        self.assertTrue(all("==" in package for package in packages))

    def test_runtime_uses_digest_not_mutable_tag(self) -> None:
        image = _load_json("runtime.json")["image"]
        self.assertRegex(image, r"^[^@]+@sha256:[0-9a-f]{64}$")


class RunnerContractTest(unittest.TestCase):
    def test_print_command_is_import_light_and_resolves_environment(self) -> None:
        config_path = ROOT / "configs" / "reranker.json"
        environment = {
            "MMBERT32K_BGE_DATA": "/tmp/bge-data",
            "MMBERT32K_RERANKER_OUTPUT": "/tmp/reranker-output",
        }
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, environment, clear=False),
            contextlib.redirect_stdout(output),
        ):
            status = main(["--config", str(config_path), "--print-command"])
        self.assertEqual(status, 0)
        command = output.getvalue()
        self.assertIn("mmbert_32k.reranker", command)
        self.assertIn("/tmp/bge-data", command)
        self.assertIn("--model_revision", command)


if __name__ == "__main__":
    unittest.main()
