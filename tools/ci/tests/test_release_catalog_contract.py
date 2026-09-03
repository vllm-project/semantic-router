from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "release"))

import check_version_contract as release_contract  # noqa: E402


class ReleaseCatalogContractTests(unittest.TestCase):
    def _validate_manifest(
        self, content: str | None, *, version: str = "9.8.7"
    ) -> tuple[str, list[str]]:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            catalog_root = Path(temporary) / "config" / "recipes" / "built-in"
            if content is not None:
                snapshot = release_contract.catalog_snapshot_for_version(version)
                snapshot_dir = catalog_root / snapshot
                snapshot_dir.mkdir(parents=True)
                (snapshot_dir / "catalog.yaml").write_text(content, encoding="utf-8")
            errors: list[str] = []
            with (
                mock.patch.object(
                    release_contract, "BUILT_IN_CATALOG_ROOT", catalog_root
                ),
                mock.patch.object(
                    release_contract, "release_snapshot_errors", return_value=[]
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                snapshot = release_contract.validate_release_catalog(errors, version)
            return snapshot, errors

    def test_release_semver_maps_to_minor_catalog_snapshot(self) -> None:
        self.assertEqual(release_contract.catalog_snapshot_for_version("9.8.7"), "v9.8")
        self.assertEqual(
            release_contract.catalog_snapshot_for_version("12.34.5-rc.1"),
            "v12.34",
        )

    def test_release_catalog_requires_aligned_release_manifest(self) -> None:
        snapshot, errors = self._validate_manifest(
            """schema_version: vllm-sr/model-catalog/v1
catalog_version: v9.8
channel: release
release: v9.8
"""
        )
        self.assertEqual(snapshot, "v9.8")
        self.assertEqual(errors, [])

    def test_release_catalog_rejects_missing_minor_snapshot(self) -> None:
        snapshot, errors = self._validate_manifest(None)
        self.assertEqual(snapshot, "v9.8")
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "requires built-in catalog snapshot config/recipes/built-in/v9.8",
            errors[0],
        )

    def test_release_catalog_rejects_channel_and_version_drift(self) -> None:
        _, errors = self._validate_manifest(
            """schema_version: vllm-sr/model-catalog/v1
catalog_version: v9.9
channel: latest
release: v9.9
"""
        )
        self.assertEqual(len(errors), 3)
        messages = "\n".join(errors)
        self.assertIn(
            "release catalog channel has 'latest' but expected 'release'", messages
        )
        self.assertIn(
            "release catalog release has 'v9.9' but expected 'v9.8'", messages
        )
        self.assertIn(
            "release catalog catalog_version has 'v9.9' but expected 'v9.8'",
            messages,
        )

    def test_github_outputs_expose_validated_catalog_snapshot(self) -> None:
        contract = release_contract.ReleaseContract(
            pyproject_version="9.8.7",
            sim_version="0.1.0",
            candle_version="9.8.7",
            candle_lock_version="9.8.7",
            helm_chart_version="9.8.7",
            helm_app_version="latest",
            release_images=("vllm-sr",),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "github-output"
            release_contract.write_github_outputs(output, contract, "9.8.7")
            self.assertIn("catalog_snapshot=v9.8", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
