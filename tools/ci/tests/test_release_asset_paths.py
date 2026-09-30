"""Keep the GitHub Release upload aligned with downloaded artifact paths."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_WORKFLOW = REPO_ROOT / ".github/workflows/release.yml"
RUST_ASSETS = (
    "release-assets/rust/release/libcandle_semantic_router.a",
    "release-assets/rust/release/libcandle_semantic_router.so",
    "release-assets/rust/package/candle-semantic-router-0.4.1.crate",
    "release-assets/rust/package/qualified.sha256",
)
PYTHON_ASSETS = (
    "release-assets/python/vllm_sr-0.4.0-py3-none-any.whl",
    "release-assets/python/vllm_sr-0.4.0.tar.gz",
    "release-assets/python/manifest.json",
)


class ReleaseAssetPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
        cls.steps = workflow["jobs"]["release-notes"]["steps"]

    def _fixture(self, root: Path) -> None:
        for relative in (*RUST_ASSETS, *PYTHON_ASSETS):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"asset")

    def _verify_step(self, root: Path) -> subprocess.CompletedProcess[str]:
        step = next(
            s for s in self.steps if s.get("name") == "Verify Rust release assets"
        )
        return subprocess.run(
            ["bash", "-c", step["run"]],
            cwd=root,
            env={**os.environ, "CANDLE_CRATE_VERSION": "0.4.1"},
            capture_output=True,
            text=True,
            check=False,
        )

    def test_nested_download_paths_match_every_release_asset(self) -> None:
        upload = next(
            s
            for s in self.steps
            if s.get("name") == "Create release after every publisher succeeds"
        )
        self.assertTrue(upload["with"]["fail_on_unmatched_files"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            matched: set[str] = set()
            for pattern in upload["with"]["files"].splitlines():
                paths = list(root.glob(pattern))
                self.assertTrue(paths, f"Unmatched release asset glob: {pattern}")
                self.assertTrue(all(path.is_file() for path in paths), pattern)
                matched.update(str(path.relative_to(root)) for path in paths)
            self.assertEqual(matched, set(RUST_ASSETS) | set(PYTHON_ASSETS))
            self.assertEqual(self._verify_step(root).returncode, 0)

    def test_missing_or_empty_rust_asset_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            for relative in RUST_ASSETS:
                with self.subTest(asset=relative):
                    path = root / relative
                    path.write_bytes(b"")
                    result = self._verify_step(root)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(relative, result.stdout)
                    path.write_bytes(b"asset")


if __name__ == "__main__":
    unittest.main()
