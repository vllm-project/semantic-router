"""Shared libraries must never silently reuse another revision or missing build."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_artifact


class NativeArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        files = {}
        for name in native_artifact.LIBRARIES:
            path = self.root / name
            path.parent.mkdir(parents=True)
            path.write_bytes(name.encode())
            files[name] = native_artifact.digest(path)
        self.manifest = {
            "source_sha": "a" * 40,
            "platform": "linux/amd64",
            "build": native_artifact.BUILD,
            "files": files,
        }
        for name, value in (("system", "Linux"), ("machine", "x86_64")):
            patch = mock.patch.object(
                native_artifact.platform, name, return_value=value
            )
            patch.start()
            self.addCleanup(patch.stop)

    def validate(self) -> None:
        native_artifact.validate(self.manifest, self.root, "a" * 40)

    def test_complete_same_source_artifact_passes(self) -> None:
        self.validate()

    def test_wrong_source_fails(self) -> None:
        self.manifest["source_sha"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "source SHA"):
            self.validate()

    def test_modified_library_fails(self) -> None:
        (self.root / native_artifact.LIBRARIES[0]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.validate()

    def test_missing_inventory_fails(self) -> None:
        del self.manifest["files"][native_artifact.LIBRARIES[0]]
        with self.assertRaisesRegex(ValueError, "exactly four"):
            self.validate()

    def test_gpu_build_cannot_qualify_cpu(self) -> None:
        self.manifest = json.loads(json.dumps(self.manifest))
        self.manifest["build"]["candle"] = "cuda"
        with self.assertRaisesRegex(ValueError, "build settings"):
            self.validate()

    def test_wrong_host_fails(self) -> None:
        with mock.patch.object(
            native_artifact.platform, "machine", return_value="arm64"
        ), self.assertRaisesRegex(ValueError, "Linux x86_64"):
            self.validate()

    def test_parity_reuses_and_verifies_prebuilt_libraries(self) -> None:
        with mock.patch.dict(
            native_artifact.os.environ,
            {"PREBUILT_NATIVE_LIBS": "1", "NATIVE_ARTIFACT_DIR": str(self.root)},
        ), mock.patch.object(native_artifact.subprocess, "run") as run:
            path = native_artifact.prepare_ml_library()
        run.assert_called_once_with(
            [
                sys.executable,
                native_artifact.__file__,
                "verify",
                "--directory",
                str(self.root),
            ],
            check=True,
        )
        self.assertEqual(
            path.parent, native_artifact.ROOT / "ml-binding/target/release"
        )

    def test_parity_rejects_invalid_prebuilt_instead_of_rebuilding(self) -> None:
        error = native_artifact.subprocess.CalledProcessError(1, "verify")
        with mock.patch.dict(
            native_artifact.os.environ,
            {"PREBUILT_NATIVE_LIBS": "1", "NATIVE_ARTIFACT_DIR": str(self.root)},
        ), mock.patch.object(
            native_artifact.subprocess, "run", side_effect=error
        ) as run, self.assertRaises(
            native_artifact.subprocess.CalledProcessError
        ):
            native_artifact.prepare_ml_library()
        self.assertEqual(run.call_count, 1)

    def test_parity_requires_prebuilt_manifest_directory(self) -> None:
        with mock.patch.dict(
            native_artifact.os.environ, {"PREBUILT_NATIVE_LIBS": "1"}, clear=True
        ), mock.patch.object(
            native_artifact.subprocess, "run"
        ) as run, self.assertRaisesRegex(
            ValueError, "NATIVE_ARTIFACT_DIR"
        ):
            native_artifact.prepare_ml_library()
        run.assert_not_called()

    def test_local_parity_still_builds_current_source(self) -> None:
        with mock.patch.dict(
            native_artifact.os.environ, {"PREBUILT_NATIVE_LIBS": "0"}
        ), mock.patch.object(native_artifact.subprocess, "run") as run:
            native_artifact.prepare_ml_library()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["cargo", "build", "--release", "--locked"])
        self.assertIn(str(native_artifact.ROOT / "ml-binding/Cargo.toml"), command)
        self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
