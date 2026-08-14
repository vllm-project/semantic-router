from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))
sys.path.insert(0, str(REPO_ROOT / "src" / "vllm-sr"))

import snapshot_model_catalog as snapshot  # noqa: E402
from cli.model_bundle import MODEL_BUNDLE_FILES, model_bundle_digest  # noqa: E402


class ModelCatalogSnapshotTests(unittest.TestCase):
    def _built_in_root(self, temporary: str) -> Path:
        root = Path(temporary) / "built-in"
        shutil.copytree(
            REPO_ROOT / "config" / "recipes" / "built-in" / "latest",
            root / "latest",
        )
        return root

    def test_snapshot_is_a_release_bound_exact_five_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._built_in_root(temporary)

            destination = snapshot.create_release_snapshot(root, "v9.8")

            self.assertEqual(snapshot.release_snapshot_errors(root, "v9.8"), [])
            catalog = yaml.safe_load(
                (destination / "catalog.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(catalog["channel"], "release")
            self.assertEqual(catalog["release"], "v9.8")
            self.assertEqual(catalog["catalog_version"], "v9.8")
            bundle = destination / "chorus-v1"
            self.assertEqual(
                sorted(path.name for path in bundle.iterdir()),
                sorted(MODEL_BUNDLE_FILES),
            )
            digest = model_bundle_digest(bundle)
            self.assertEqual(catalog["assets"][0]["sha256"], digest)
            self.assertTrue(
                all(
                    model["verification"]["asset_sha256"] == digest
                    for model in catalog["models"]
                )
            )
            probes = (bundle / "probes.yaml").read_text(encoding="utf-8")
            self.assertIn("config/recipes/built-in/v9.8/chorus-v1/config.yaml", probes)
            self.assertNotIn(
                "config/recipes/built-in/latest/chorus-v1/config.yaml", probes
            )
            metadata_text = (bundle / "metadata.yaml").read_text(encoding="utf-8")
            metadata = yaml.safe_load(metadata_text)
            self.assertTrue(
                metadata["links"]["source"].endswith(
                    "/config/recipes/built-in/v9.8/chorus-v1"
                )
            )
            self.assertTrue(
                metadata["links"]["documentation"].endswith(
                    "/config/recipes/built-in/v9.8/chorus-v1/README.md"
                )
            )
            for name in MODEL_BUNDLE_FILES:
                self.assertNotIn(
                    "config/recipes/built-in/latest/chorus-v1",
                    (bundle / name).read_text(encoding="utf-8"),
                )

    def test_snapshot_refuses_to_overwrite_existing_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._built_in_root(temporary)
            destination = snapshot.create_release_snapshot(root, "v9.8")
            catalog_before = (destination / "catalog.yaml").read_bytes()

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                snapshot.create_release_snapshot(root, "v9.8")

            self.assertEqual(
                (destination / "catalog.yaml").read_bytes(), catalog_before
            )

    def test_failed_snapshot_validation_cleans_staging_and_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._built_in_root(temporary)
            with (
                mock.patch.object(
                    snapshot, "_release_tree_errors", return_value=["injected failure"]
                ),
                self.assertRaisesRegex(ValueError, "injected failure"),
            ):
                snapshot.create_release_snapshot(root, "v9.8")

            self.assertFalse((root / "v9.8").exists())
            self.assertEqual(
                [path.name for path in root.iterdir() if ".staging-" in path.name],
                [],
            )

    def test_snapshot_check_rejects_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._built_in_root(temporary)
            destination = snapshot.create_release_snapshot(root, "v9.8")
            (destination / "chorus-v1" / "README.md").write_text(
                "drift\n", encoding="utf-8"
            )

            errors = snapshot.release_snapshot_errors(root, "v9.8")

            self.assertEqual(
                errors,
                ["release snapshot drifted from latest: chorus-v1/README.md"],
            )

    def test_published_tag_snapshot_is_immutable_but_new_version_is_allowed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            published_root = repo / "config" / "recipes" / "built-in" / "v9.8"
            published_root.mkdir(parents=True)
            (published_root / "catalog.yaml").write_bytes(b"published\n")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "add",
                    "config/recipes/built-in/v9.8",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Catalog Test",
                    "-c",
                    "user.email=catalog@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "publish snapshot",
                ],
                check=True,
            )
            subprocess.run(["git", "-C", str(repo), "tag", "v9.8.0"], check=True)
            base_ref = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            built_in_root = repo / "config" / "recipes" / "built-in"
            published = snapshot.published_snapshots_from_git(
                repo, built_in_root, base_ref
            )

            (built_in_root / "v9.9").mkdir()
            (built_in_root / "v9.9" / "catalog.yaml").write_bytes(b"new\n")
            self.assertEqual(
                snapshot.published_snapshot_errors(built_in_root, published), []
            )

            (published_root / "catalog.yaml").write_bytes(b"drift\n")
            self.assertEqual(
                snapshot.published_snapshot_errors(built_in_root, published),
                [
                    "v9.8: release snapshot drifted from published tag v9.8.0: "
                    "catalog.yaml"
                ],
            )


if __name__ == "__main__":
    unittest.main()
