"""Exercise the ledger primitives shared by the development loop."""

import argparse
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
        return {
            "kind": "measurement",
            "split": "dev",
            "arm": arm,
            "recipe": "main",
            "public_model": "public",
            "strong_model": "A",
            "config": f"{arm}/config.yaml",
            "config_sha256": digest(f"{arm}".encode()),
            "agent_seconds": seconds,
            "hypothesis": "Probe tool use",
            "evidence": {},
            "cases": {},
        }

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

    def _graded_fixtures(self):
        """A real config file plus one physical episode (result + requests)."""
        run = self.workspace.parent / "run"
        folder = run / "dsh" / "runs" / "task" / "fixed" / "main"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "result.json").write_text(
            json.dumps({"termination": "natural_completion"}), encoding="utf-8"
        )
        (folder / "requests.jsonl").write_text(
            json.dumps({"event": "dispatch", "call": 1}) + "\n", encoding="utf-8"
        )
        config = run / "config.yaml"
        config.write_text(
            "version: v0.5\n"
            "entrypoints:\n"
            "  - model_names: [public]\n"
            "    recipe: main\n"
            "recipes:\n"
            "  - name: sibling\n"
            "  - name: main\n"
            "providers:\n"
            "  models:\n"
            "    - name: A\n",
            encoding="utf-8",
        )
        evidence = {
            str(folder / "result.json"): digest((folder / "result.json").read_bytes()),
            str(folder / "requests.jsonl"): digest(
                (folder / "requests.jsonl").read_bytes()
            ),
        }
        return config, evidence

    def test_relabelled_measurement_is_not_an_independent_comparison(self):
        config, evidence = self._graded_fixtures()

        def measurement(hypothesis):
            return {
                "kind": "measurement",
                "split": "dev",
                "arm": "fixed",
                "recipe": "main",
                "public_model": "public",
                "strong_model": "A",
                "config": str(config),
                "config_sha256": digest(config.read_bytes()),
                "hypothesis": hypothesis,
                "evidence": evidence,
                "cases": {},
            }

        first = self.ledger.append(measurement("first import"))
        second = self.ledger.append(measurement("Relabel the same physical run"))
        self.assertNotEqual(first["record_id"], second["record_id"])
        # Same physical episode (identical result/request fingerprints): a
        # relabelled re-import must not become an independent comparison.
        with self.assertRaises(ValueError):
            loop.compare(
                self.ledger,
                argparse.Namespace(
                    baseline=first["record_id"],
                    candidate=second["record_id"],
                    disposition=None,
                    reason=None,
                ),
            )

    def test_failed_publication_preserves_previous_ledger(self):
        self.ledger.append(self.measurement("fixed", 12.5))
        before = self.ledger.path.read_bytes()
        # A killed writer must leave the previous JSONL intact, including its
        # last line: publish the unchanged prefix and the new line atomically.
        with (
            patch.object(
                loop.os, "replace", side_effect=OSError("interrupted publish")
            ),
            self.assertRaises(OSError),
        ):
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
