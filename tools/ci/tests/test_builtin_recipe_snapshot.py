from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))

import snapshot_builtin_recipes as snapshot  # noqa: E402
from recipe_bundle import RECIPE_BUNDLE_FILES  # noqa: E402


class BuiltInRecipeSnapshotTests(unittest.TestCase):
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
            self.assertEqual(
                sorted(path.name for path in destination.iterdir()), ["mom-v1"]
            )
            bundle = destination / "mom-v1"
            self.assertEqual(
                sorted(path.name for path in bundle.iterdir()),
                sorted(RECIPE_BUNDLE_FILES),
            )
            probes = (bundle / "probes.yaml").read_text(encoding="utf-8")
            self.assertIn("config/recipes/built-in/v9.8/mom-v1/config.yaml", probes)
            metadata = (bundle / "metadata.yaml").read_text(encoding="utf-8")
            self.assertIn("/config/recipes/built-in/v9.8/mom-v1", metadata)
            for name in RECIPE_BUNDLE_FILES:
                self.assertNotIn(
                    "config/recipes/built-in/latest/mom-v1",
                    (bundle / name).read_text(encoding="utf-8"),
                )

    def test_latest_accepts_only_complete_named_recipe_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._built_in_root(temporary)
            (root / "latest" / "README.md").write_text("unexpected\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "only named bundle directories"):
                snapshot.latest_bundles(root / "latest")

            (root / "latest" / "README.md").unlink()
            (root / "latest" / "mom-v1" / "config.yaml").unlink()
            with self.assertRaisesRegex(ValueError, "five-file contract"):
                snapshot.latest_bundles(root / "latest")

    def test_snapshot_refuses_to_overwrite_existing_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._built_in_root(temporary)
            destination = snapshot.create_release_snapshot(root, "v9.8")
            content_before = (destination / "mom-v1" / "config.yaml").read_bytes()

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                snapshot.create_release_snapshot(root, "v9.8")

            self.assertEqual(
                (destination / "mom-v1" / "config.yaml").read_bytes(),
                content_before,
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
            (destination / "mom-v1" / "README.md").write_text(
                "drift\n", encoding="utf-8"
            )

            self.assertEqual(
                snapshot.release_snapshot_errors(root, "v9.8"),
                ["Recipe snapshot drifted from latest: mom-v1/README.md"],
            )

    def test_published_tag_snapshot_is_immutable_but_new_version_is_allowed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            published_root = repo / "config" / "recipes" / "built-in" / "v9.8"
            published_file = published_root / "mom-v1" / "config.yaml"
            published_file.parent.mkdir(parents=True)
            published_file.write_bytes(b"published\n")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "add", "config/recipes/built-in/v9.8"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Recipe Test",
                    "-c",
                    "user.email=recipe@example.invalid",
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

            new_file = built_in_root / "v9.9" / "mom-v1" / "config.yaml"
            new_file.parent.mkdir(parents=True)
            new_file.write_bytes(b"new\n")
            self.assertEqual(
                snapshot.published_snapshot_errors(built_in_root, published), []
            )

            published_file.write_bytes(b"drift\n")
            self.assertEqual(
                snapshot.published_snapshot_errors(built_in_root, published),
                [
                    "v9.8: Recipe snapshot drifted from published tag v9.8.0: "
                    "mom-v1/config.yaml"
                ],
            )


if __name__ == "__main__":
    unittest.main()
