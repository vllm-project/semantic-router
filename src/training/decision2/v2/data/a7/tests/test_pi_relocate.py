from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from v2.data.a7.pi_relocate import relocate


class RelocateTest(unittest.TestCase):
    def test_first_directory_with_identical_bytes_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp, "a"), Path(tmp, "b")
            first.mkdir()
            second.mkdir()
            data = b'{"id": "p1", "state": "protected"}\n'
            (first / "typed_dev.jsonl").write_bytes(b"different bytes\n")
            (second / "typed_dev.jsonl").write_bytes(data)
            (second / "css_pilot.jsonl").write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            entries = [
                {
                    "role": "typed_dev",
                    "path": "/elsewhere/typed_dev.jsonl",
                    "sha256": digest,
                },
                {
                    "role": "css_pilot",
                    "path": "/elsewhere/css_pilot.jsonl",
                    "sha256": digest.upper(),
                },
            ]
            out = relocate(entries, [first, second])
            self.assertEqual([item["role"] for item in out], ["typed_dev", "css_pilot"])
            self.assertTrue(all(item["sha256"] == digest for item in out))
            self.assertEqual(
                out[0]["path"], str((second / "typed_dev.jsonl").resolve())
            )

    def test_missing_or_changed_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "typed_dev.jsonl").write_bytes(b"changed\n")
            entry = {"role": "typed_dev", "path": "x", "sha256": "0" * 64}
            with self.assertRaises(FileNotFoundError):
                relocate([entry], [Path(tmp)])


if __name__ == "__main__":
    unittest.main()
