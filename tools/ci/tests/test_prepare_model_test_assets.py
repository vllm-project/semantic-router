"""Exercise shared preparation without downloading models or invoking Docker."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_model_test_assets as preparation


class PreparationTests(unittest.TestCase):
    def test_runtime_then_calibration_reuses_verified_nano_and_mini_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def export(args, **kwargs):
                variant = args[args.index("--build-arg") + 1].split("=")[1]
                destination = Path(args[args.index("--output") + 1].split("dest=")[1])
                artifact = destination / ("vela-1.0-omni-" + variant)
                artifact.mkdir()
                (artifact / "verified").write_text("valid")

            def valid(path):
                return (path / "verified").is_file()

            with patch.object(
                preparation, "fingerprint", return_value="source-one"
            ), patch.object(preparation, "verify", side_effect=valid), patch.object(
                preparation.subprocess, "run", side_effect=export
            ) as build:
                preparation.prepare(output, ["nano"])
                preparation.prepare(output, ["nano", "mini"])
                preparation.prepare(output, ["nano"])
                self.assertEqual(build.call_count, 2)
                (output / "vela-1.0-omni-nano/verified").unlink()
                preparation.prepare(output, ["nano"])
                self.assertEqual(build.call_count, 3)
                with patch.object(
                    preparation, "fingerprint", return_value="source-two"
                ):
                    preparation.prepare(output, ["mini"])
                self.assertEqual(build.call_count, 4)
                (output / "vela-1.0-omni-nano.preparation.json").write_text("{corrupt")
                preparation.prepare(output, ["nano"])
                self.assertEqual(build.call_count, 5)
                self.assertEqual(
                    json.loads(
                        (output / "vela-1.0-omni-mini.preparation.json").read_text()
                    )["inputs_sha256"],
                    "source-two",
                )

    def test_invalid_export_never_replaces_prior_artifact_or_publishes_receipt(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            preparation, "fingerprint", return_value="source"
        ), patch.object(preparation, "verify", return_value=False), patch.object(
            preparation.subprocess, "run"
        ):
            output = Path(temporary)
            existing = output / "vela-1.0-omni-nano"
            existing.mkdir()
            (existing / "prior").write_text("prior")
            with self.assertRaisesRegex(ValueError, "verification"):
                preparation.prepare(output, ["nano"])
            self.assertTrue((existing / "prior").is_file())
            self.assertFalse((output / "vela-1.0-omni-nano.preparation.json").exists())

    def test_fingerprint_tracks_source_bytes_but_ignores_python_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prepare"
            source.mkdir()
            (source / "Dockerfile").write_text("FROM example")
            with patch.object(preparation, "ROOT", root), patch.object(
                preparation, "PREPARATION", source
            ):
                first = preparation.fingerprint()
                (source / "__pycache__").mkdir()
                (source / "__pycache__/cache.pyc").write_text("cache")
                self.assertEqual(first, preparation.fingerprint())
                (source / "Dockerfile").write_text("FROM changed")
                self.assertNotEqual(first, preparation.fingerprint())
                second = preparation.fingerprint()
                (root / ".dockerignore").write_text("tools/models/vela_omni/omitted.py")
                self.assertNotEqual(second, preparation.fingerprint())
