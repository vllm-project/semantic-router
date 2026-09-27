"""Gold-free raw manifest line ordering regression tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.reorder_v3_raw_manifest import reorder


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RawManifestRepairTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        files = {}
        for panel in ("typed", "css", "public"):
            path = self.root / f"{panel}.jsonl"
            path.write_text(panel + "\n", encoding="utf-8")
            files[panel] = str(path)
        self.plan = self.root / "plan.json"
        self.plan.write_text(
            json.dumps(
                {
                    "plan_version": "decision2-first-release-v3-plan/2",
                    "evaluation_root": str(self.root),
                    "inference": [{"group": "decision1", "paths": files}],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.raw = self.root / "RAW_PREDICTIONS.sha256"
        self.raw.write_text(
            "".join(f"{sha(Path(files[p]))}  {files[p]}\n" for p in files),
            encoding="utf-8",
        )
        self.original = self.raw.read_bytes()
        self.copy = self.root / "planned-order.sha256"
        self.receipt = self.root / "receipt.json"

    def test_only_order_changes_and_original_retained(self):
        result = reorder(self.plan, sha(self.plan), self.raw, self.copy, self.receipt)
        self.assertEqual(self.copy.read_bytes(), self.original)
        self.assertEqual(result["line_count"], 3)
        self.assertEqual(
            [
                Path(line.split("  ", 1)[1]).stem
                for line in self.raw.read_text().splitlines()
            ],
            ["css", "public", "typed"],
        )
        self.assertEqual(result["auditor_order_sha256"], sha(self.raw))

    def test_changed_digest_fails_without_mutation(self):
        first = Path(
            json.loads(self.plan.read_text())["inference"][0]["paths"]["typed"]
        )
        first.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "beyond line ordering"):
            reorder(self.plan, sha(self.plan), self.raw, self.copy, self.receipt)
        self.assertEqual(self.raw.read_bytes(), self.original)
        self.assertFalse(self.copy.exists())


if __name__ == "__main__":
    unittest.main()
