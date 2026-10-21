"""Check the published benchmark kit and its source identity recording."""

import importlib.util
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "website/static/files/ospp-riscv64"
SPEC = importlib.util.spec_from_file_location(
    "published_probe", KIT / "run-go-probe.py"
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class BenchmarkCompanionTests(unittest.TestCase):
    def git(self, source, *args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    def repository(self, source):
        self.git(source, "init", "--quiet")
        (source / "sample").write_text("first\n")
        self.git(source, "add", "sample")
        self.git(
            source,
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        )

    def identity(self, source):
        identity = getattr(PROBE, "source_identity", None)
        self.assertIsNotNone(
            identity, "benchmark must record the actual source checkout"
        )
        return identity(source)

    def test_clean_checkout_records_actual_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            self.repository(source)
            result = self.identity(source)
            self.assertEqual(
                result["source_revision"], self.git(source, "rev-parse", "HEAD")
            )
            self.assertFalse(result["source_dirty"])

    def test_modified_and_untracked_inputs_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            self.repository(source)
            (source / "sample").write_text("modified\n")
            self.assertTrue(self.identity(source)["source_dirty"])
            (source / "sample").write_text("first\n")
            (source / "untracked").write_text("extra input\n")
            self.assertTrue(self.identity(source)["source_dirty"])

    def test_non_git_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNotNone(getattr(PROBE, "source_identity", None))
            with self.assertRaises(subprocess.CalledProcessError):
                PROBE.source_identity(Path(tmp))

    def test_download_archive_matches_public_sources(self):
        expected = {
            "go-router-probe/main.go",
            "go-router-probe/main_test.go",
            "go-router-probe/go.mod",
            "run-go-probe.py",
            "go-router-probe-summary.json",
        }
        with zipfile.ZipFile(KIT / "benchmark-kit.zip") as archive:
            self.assertEqual(set(archive.namelist()), expected)
            for name in expected:
                self.assertEqual(archive.read(name), (KIT / name).read_bytes(), name)
