"""Tests for packaged built-in model bundle contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "vllm-sr"))

from cli.model_bundle import (  # noqa: E402
    MODEL_BUNDLE_FILES,
    OPTIONAL_MOM_BUNDLE_FILES,
    model_bundle_digest,
    model_bundle_optional_files,
)


class ModelBundleTests(unittest.TestCase):
    def test_mom_v1_bundle_allows_optional_evaluation_files(self) -> None:
        bundle = REPO_ROOT / "config" / "recipes" / "built-in" / "latest" / "mom-v1"
        optional = model_bundle_optional_files(bundle)
        self.assertEqual(sorted(optional), sorted(OPTIONAL_MOM_BUNDLE_FILES))
        digest = model_bundle_digest(bundle)
        self.assertTrue(digest.startswith("sha256:"))

    def test_exact_five_bundle_still_digests(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "sample"
            bundle.mkdir()
            for name in MODEL_BUNDLE_FILES:
                (bundle / name).write_text(f"{name}\n", encoding="utf-8")
            digest = model_bundle_digest(bundle)
            self.assertTrue(digest.startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
