"""Offline contracts; real tensor parity is mandatory in export.py, never skipped."""

import hashlib
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

from contract import (
    CLAP_WINDOW_SAMPLES,
    artifact_manifest,
    digest,
    endpoint_windows,
    inventory,
    safe_file,
    sources,
    verify_inventory,
)
from source import verify_source


class ArtifactContractTests(unittest.TestCase):
    def test_source_identity_rejects_modified_code_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "vela_omni.py"
            path.write_text("raise RuntimeError('never execute this fixture')\n")
            identity = {
                "files": {
                    "vela_omni.py": {
                        "size": path.stat().st_size,
                        "algorithm": "sha256",
                        "digest": digest(path),
                    }
                }
            }
            verify_source(root, identity)
            path.write_text("raise RuntimeError('changed unreviewed source!!')\n")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_source(root, identity)

    def test_unreviewed_import_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "torch.py").write_text("raise RuntimeError('shadow module')")
            with self.assertRaisesRegex(ValueError, "unexpected/missing"):
                verify_source(root, {"files": {}})

    def test_hub_symlink_source_is_authenticated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            blob = root / "blob"
            blob.write_bytes(b"native weights")
            (snapshot / "model.safetensors").symlink_to(blob)
            verify_source(
                snapshot,
                {
                    "files": {
                        "model.safetensors": {
                            "size": blob.stat().st_size,
                            "algorithm": "sha256",
                            "digest": digest(blob),
                        }
                    }
                },
            )

    def test_git_blob_identity_uses_git_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test"
            path.write_bytes(b"abc")
            self.assertEqual(
                digest(path, "git-blob-sha1"), hashlib.sha1(b"blob 3\0abc").hexdigest()
            )

    def test_four_graphs_cannot_hide_clap_or_projection(self):
        for variant, dimension, limit, size in (
            ("nano", 384, 512, 512),
            ("mini", 768, 32768, 384),
        ):
            manifest = artifact_manifest(variant, "right", 0)
            self.assertEqual(
                set(manifest["graphs"]), {"text", "image", "clap", "audio"}
            )
            self.assertEqual(manifest["embedding"]["dimensions"], [dimension])
            self.assertEqual(manifest["max_text_length"], limit)
            self.assertEqual(
                manifest["graphs"]["image"]["inputs"][0]["shape"], [1, 3, size, size]
            )
            self.assertEqual(
                manifest["graphs"]["audio"]["inputs"][1]["shape"], [1, 512]
            )
            self.assertEqual(
                manifest["graphs"]["audio"]["output"]["shape"], [1, dimension]
            )
            self.assertFalse(manifest["reference_parity"]["passed"])

    def test_endpoint_windows_cover_tail_and_use_all_duration(self):
        for length in (1, 100, 480000, 480001, 959999, 960000, 960001, 1440000):
            windows = endpoint_windows(length)
            self.assertEqual(windows[0][0], 0)
            self.assertEqual(windows[-1][1], length)
            self.assertTrue(
                all(end - start <= CLAP_WINDOW_SAMPLES for start, end in windows)
            )
            self.assertTrue(
                all(right[0] <= left[1] for left, right in pairwise(windows))
            )
        self.assertEqual(
            endpoint_windows(1000000),
            [(0, 480000), (260000, 740000), (520000, 1000000)],
        )
        for length in (0, -1, 1440001):
            with self.assertRaises(ValueError):
                endpoint_windows(length)

    def test_artifact_digest_and_path_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights").write_bytes(b"weights")
            manifest = {"files": inventory(root)}
            verify_inventory(root, manifest)
            (root / "weights").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_inventory(root, manifest)
            for name in ("../secret", "/etc/passwd", "x/../../y", "x\\y"):
                with self.assertRaises(ValueError):
                    safe_file(root, name)

    def test_releases_have_content_locked_weights_and_code(self):
        for source in sources().values():
            self.assertEqual(len(source["revision"]), 40)
            self.assertEqual(
                source["files"]["model.safetensors"]["algorithm"], "sha256"
            )
            self.assertIn("vela_omni.py", source["files"])
            self.assertIn("components/text/tokenizer.json", source["files"])
            self.assertIn(
                "components/audio_clap/preprocessor_config.json", source["files"]
            )


if __name__ == "__main__":
    unittest.main()
