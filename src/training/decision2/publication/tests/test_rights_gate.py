"""Exact noncommercial attestation checks for generic LoRA publication."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from publication.noncommercial_attestation import SOURCES, build
from publication.rights_gate import OLD_HOLDOUT_SHA, verify_rights


class RightsGateTest(unittest.TestCase):
    def test_pilot_needs_run_bound_statement_and_restrictive_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, provenance, statement = (
                root / name
                for name in ("manifest.json", "provenance.json", "attestation.json")
            )
            data_sha = {"train": "a" * 64, **OLD_HOLDOUT_SHA}
            source_counts = dict.fromkeys(SOURCES, 1)
            data = {
                "schema_version": "decision2-balanced-human-5824/1",
                "counts": {"source": source_counts},
                "outputs": {
                    "balanced_human_5824.train.jsonl": {"sha256": data_sha["train"]},
                    "select.jsonl": {"sha256": data_sha["select"]},
                    "cal.jsonl": {"sha256": data_sha["cal"]},
                },
            }
            manifest.write_text(json.dumps(data))
            provenance.write_text(json.dumps({"contract": {"data_sha256": data_sha}}))
            statement.write_text(json.dumps(build(manifest, provenance)))
            options = dict(
                data=data,
                data_manifest_path=manifest,
                run_provenance_path=provenance,
                partition_sha=data_sha,
                partition_rows={"train": 19, "select": 600, "cal": 900},
                source_counts=source_counts,
                attestation_path=statement,
                license_id="other",
            )
            result = verify_rights(**options)
            self.assertEqual(result["mode"], "noncommercial_research")
            with self.assertRaisesRegex(ValueError, "license metadata"):
                verify_rights(**{**options, "license_id": "apache-2.0"})
            provenance.write_text(
                json.dumps(
                    {"contract": {"data_sha256": data_sha}, "new": "run mutation"}
                )
            )
            with self.assertRaisesRegex(ValueError, "differs from exact run"):
                verify_rights(**options)


if __name__ == "__main__":
    unittest.main()
