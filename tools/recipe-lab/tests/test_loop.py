"""Exercise the ledger primitives shared by the development loop."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recipe_lab import loop
from recipe_lab.loop_evidence import digest


class LedgerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.workspace = Path(tmp.name) / "ledger"
        self.ledger = loop.Ledger(self.workspace)

    def measurement(self, arm, seconds):
        return dict(
            kind="measurement",
            split="dev",
            arm=arm,
            recipe="main",
            public_model="public",
            strong_model="A",
            config=f"{arm}/config.yaml",
            config_sha256=digest(f"{arm}".encode()),
            agent_seconds=seconds,
            hypothesis="Probe tool use",
            evidence={},
            cases={},
        )

    def test_round_trip(self):
        first = self.ledger.append(self.measurement("fixed", 12.5))
        self.assertTrue(first["record_id"].startswith("rec-"))
        # Re-appending the identical measurement is a no-op, not a new row.
        again = self.ledger.append({k: v for k, v in first.items() if k != "record_id"})
        self.assertEqual(again, first)
        candidate = self.ledger.append(self.measurement("draft", 10))
        self.assertNotEqual(candidate["record_id"], first["record_id"])
        rows = self.ledger.rows()
        self.assertEqual(
            [r["record_id"] for r in rows],
            [first["record_id"], candidate["record_id"]],
        )
        # measure() resolves measurements by record id and rejects unknown ones.
        self.assertEqual(self.ledger.measure(first["record_id"]), first)
        with self.assertRaises(ValueError):
            self.ledger.measure("rec-unknown000")

    def test_tampered_ledger_is_rejected(self):
        self.ledger.append(self.measurement("fixed", 12.5))
        # Any rewrite without recomputing the fingerprint is refused.
        rows = [json.loads(line) for line in self.ledger.path.read_bytes().splitlines()]
        rows[0]["arm"] = "draft"
        self.ledger.path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            self.ledger.rows()

    def test_failed_publication_preserves_previous_ledger(self):
        self.ledger.append(self.measurement("fixed", 12.5))
        before = self.ledger.path.read_bytes()
        # A killed writer must leave the previous JSONL intact, including its
        # last line: publish the unchanged prefix and the new line atomically.
        with patch.object(
            loop.os, "replace", side_effect=OSError("interrupted publish")
        ):
            with self.assertRaises(OSError):
                self.ledger.append(self.measurement("draft", 10))
        self.assertEqual(before, self.ledger.path.read_bytes())
        leftovers = [
            p.name
            for p in self.workspace.iterdir()
            if p.name not in ("ledger.jsonl", "ledger.lock")
        ]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
