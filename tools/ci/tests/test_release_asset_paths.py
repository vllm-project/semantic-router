"""Keep the GitHub Release upload aligned with downloaded artifact paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_WORKFLOW = REPO_ROOT / ".github/workflows/release.yml"
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
        for relative in PYTHON_ASSETS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"asset")

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
            self.assertEqual(matched, set(PYTHON_ASSETS))


if __name__ == "__main__":
    unittest.main()
