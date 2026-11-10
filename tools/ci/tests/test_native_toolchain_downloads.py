"""Untrusted or incomplete offline inputs must never become SDK inputs."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/docker/riscv64/fetch-toolchains.py"


class NativeToolchainDownloadsTests(unittest.TestCase):
    def run_fetch(self, directory, manifest):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(manifest),
                "--verify-only",
                str(directory),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    def fixture(self, root, **changes):
        payload = b"test archive bytes"
        item = {
            "filename": "tool.tar.xz",
            "url": "https://example.org/tool.tar.xz",
            "sha256": hashlib.sha256(payload).hexdigest(),
        } | changes
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps([item]))
        return payload, manifest

    def test_verified_archive_is_reported_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, manifest = self.fixture(root)
            (root / "tool.tar.xz").write_bytes(payload)
            result = self.run_fetch(root, manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("tool.tar.xz", result.stdout)

    def test_corrupt_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = self.fixture(root)
            (root / "tool.tar.xz").write_bytes(b"corrupted")
            result = self.run_fetch(root, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum mismatch", result.stderr)

    def test_missing_archive_never_falls_back_to_network_in_verify_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = self.fixture(root)
            result = self.run_fetch(root, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("archive not found", result.stderr)

    def test_path_traversal_is_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = self.fixture(root, filename="../outside")
            result = self.run_fetch(root, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid archive filename", result.stderr)

    def test_insecure_download_url_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = self.fixture(root, url="http://example.org/tool.tar.xz")
            result = self.run_fetch(root, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("HTTPS", result.stderr)

    def test_symlink_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, manifest = self.fixture(root)
            (root / "other").write_bytes(payload)
            (root / "tool.tar.xz").symlink_to(root / "other")
            result = self.run_fetch(root, manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("regular file", result.stderr)


if __name__ == "__main__":
    unittest.main()
