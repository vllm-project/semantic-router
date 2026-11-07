"""Fail closed before compiling an unverified native dependency."""

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/docker/riscv64/build-glide.sh"


class GlideSourceBuildTests(unittest.TestCase):
    def run_builder(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_corrupt_archive_is_rejected_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "source.tar.gz"
            archive.write_bytes(b"not the pinned release")
            output = Path(tmp) / "output"
            result = self.run_builder(
                "--archive", str(archive), "--output", str(output)
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source checksum mismatch", result.stderr)
            self.assertFalse(output.exists())

    def test_missing_explicit_archive_does_not_fall_back_to_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_builder(
                "--archive",
                str(Path(tmp) / "missing"),
                "--output",
                str(Path(tmp) / "out"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source archive not found", result.stderr)

    def test_unsupported_target_fails_before_downloading(self):
        result = self.run_builder(
            "--target", "not-a-rust-target", "--output", "/unused"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported GLIDE target", result.stderr)

    def test_missing_argument_is_a_usage_error(self):
        result = self.run_builder("--output")
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing value", result.stderr)

    def test_output_is_required(self):
        result = self.run_builder()
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
