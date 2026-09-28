from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.data.assemble_hf_upload import assemble


class AssembleHfUploadTest(unittest.TestCase):
    def test_strips_absolute_paths_and_writes_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "m.json"
            manifest.write_text(
                json.dumps({"path": "/data/x/private/a.jsonl", "rows": 3})
            )
            rows = root / "r.jsonl"
            rows.write_text('{"id":"a"}\n')
            registry = assemble(
                [
                    {"src": str(manifest), "dst": "v2/arms/A/manifest.json"},
                    {"src": str(rows), "dst": "v2/arms/A/train.jsonl"},
                ],
                root / "out",
            )
            written = json.loads((root / "out/v2/arms/A/manifest.json").read_text())
            self.assertEqual(written["path"], "a.jsonl")
            self.assertEqual(
                set(registry["files"]),
                {"v2/arms/A/manifest.json", "v2/arms/A/train.jsonl"},
            )
            self.assertTrue((root / "out/v2/registry.json").exists())
            with self.assertRaises(FileExistsError):
                assemble([], root / "out")


if __name__ == "__main__":
    unittest.main()
