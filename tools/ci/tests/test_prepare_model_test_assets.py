"""Exercise the pinned Omni download without network access."""

import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_model_test_assets as preparation

CONTENT = {"config.json": b"{}", "model.safetensors": b"weights"}


def pinned_table(files):
    table = preparation.pins()
    entry = replace(
        table.lookup("vllm-sr/Vela-1.0-Omni-Nano"),
        files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    )
    return table, entry


class PreparationTests(unittest.TestCase):
    def test_downloads_once_then_reuses_the_verified_files(self):
        table, entry = pinned_table(CONTENT)
        downloads = []

        def download(directory, pinned):
            downloads.append(pinned.repo_id)
            directory.mkdir(parents=True, exist_ok=True)
            for name, data in CONTENT.items():
                (directory / name).write_bytes(data)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(table, "lookup", return_value=entry),
            patch.object(preparation, "download", side_effect=download),
        ):
            output = Path(temporary)
            preparation.prepare(output, ["nano"])
            preparation.prepare(output, ["nano"])
            self.assertEqual(downloads, ["vllm-sr/Vela-1.0-Omni-Nano"])
            (output / "vela-1.0-omni-nano/model.safetensors").write_bytes(b"changed")
            preparation.prepare(output, ["nano"])
            self.assertEqual(len(downloads), 2)

    def test_a_download_that_differs_from_the_pins_fails(self):
        table, entry = pinned_table(CONTENT)

        def download(directory, pinned):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_bytes(b"{}")

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(table, "lookup", return_value=entry),
            patch.object(preparation, "download", side_effect=download),
            self.assertRaisesRegex(ValueError, "pinned files"),
        ):
            preparation.prepare(Path(temporary), ["nano"])


if __name__ == "__main__":
    unittest.main()
