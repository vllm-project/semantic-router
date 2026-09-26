"""Keep research-use attestation tied to exact frozen source and run bytes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from publication.noncommercial_attestation import SOURCES, build


class AttestationTest(unittest.TestCase):
    def test_exact_source_roster_and_partition_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, provenance = root / "manifest.json", root / "provenance.json"
            sha = {"train": "a" * 64, "select": "b" * 64, "cal_audited_only": "c" * 64}
            data = {
                "schema_version": "decision2-balanced-human-5824/1",
                "counts": {"source": dict.fromkeys(SOURCES, 1)},
                "outputs": {
                    "balanced_human_5824.train.jsonl": {"sha256": sha["train"]},
                    "select.jsonl": {"sha256": sha["select"]},
                    "cal.jsonl": {"sha256": sha["cal_audited_only"]},
                },
            }
            manifest.write_text(json.dumps(data))
            provenance.write_text(json.dumps({"data_sha256": sha}))
            statement = build(manifest, provenance)
            self.assertEqual(set(statement["source_groups"]), set(SOURCES))
            self.assertEqual(statement["data_sha256"], sha)
            self.assertTrue(statement["no_raw_training_rows"])
            data["outputs"]["cal.jsonl"]["sha256"] = "d" * 64
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "cal_audited_only"):
                build(manifest, provenance)

    def test_qwen_run_contract_can_bind_structured_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, provenance = root / "manifest.json", root / "provenance.json"
            sha = {"train": "a" * 64, "select": "b" * 64, "cal": "c" * 64}
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "decision2-nox4b-structured-replay/1",
                        "counts": {"source": dict.fromkeys(SOURCES, 1)},
                        "outputs": {
                            "nox4b_structured.train.jsonl": {"sha256": sha["train"]},
                            "select.jsonl": {"sha256": sha["select"]},
                            "cal.jsonl": {"sha256": sha["cal"]},
                        },
                    }
                )
            )
            provenance.write_text(json.dumps({"contract": {"data_sha256": sha}}))
            self.assertEqual(build(manifest, provenance)["data_sha256"], sha)


if __name__ == "__main__":
    unittest.main()
