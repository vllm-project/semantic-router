"""Exact noncommercial attestation checks for generic LoRA publication."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from publication.noncommercial_attestation import SOURCES, build
from publication.rights_gate import (
    NONCOMMERCIAL_SCHEMA,
    NONCOMMERCIAL_SCOPE,
    OLD_HOLDOUT_COUNTS,
    OLD_HOLDOUT_SHA,
    verify_rights,
)


class RightsGateTest(unittest.TestCase):
    def test_structured_replay_requires_per_source_public_weight_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, provenance, statement_path = (
                root / name
                for name in ("manifest.json", "provenance.json", "attestation.json")
            )
            data_sha = {"train": "a" * 64, **OLD_HOLDOUT_SHA}
            source_counts = {"internal": 3, "external": 2}
            data = {
                "schema_version": "decision2-human-structured-replay/1",
                "merged_counts": {"source": source_counts},
            }
            manifest.write_text(json.dumps(data))
            provenance.write_text(json.dumps({"contract": {"data_sha256": data_sha}}))
            holdout_groups = {
                role: dict.fromkeys(counts, "holdout")
                for role, counts in OLD_HOLDOUT_COUNTS.items()
            }
            statement = {
                "schema_version": NONCOMMERCIAL_SCHEMA,
                "noncommercial_use": True,
                "publication_scope": NONCOMMERCIAL_SCOPE,
                "no_raw_training_rows": True,
                "data_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "training_provenance_sha256": hashlib.sha256(
                    provenance.read_bytes()
                ).hexdigest(),
                "data_sha256": data_sha,
                "source_counts": source_counts,
                "source_groups": {name: name for name in source_counts},
                "holdout_source_counts": OLD_HOLDOUT_COUNTS,
                "holdout_groups": holdout_groups,
                "rights_conditions": {
                    name: {"terms": "Reviewed condition", "evidence": "review receipt"}
                    for name in (*source_counts, "holdout")
                },
                "public_weight_review": {
                    "decision": "approved_noncommercial_public_weights",
                    "reviewer_role": "project rights reviewer",
                    "review_record_sha256": "d" * 64,
                    "source_decisions": {
                        name: {
                            "status": "approved",
                            "rows": rows,
                            "terms": "Reviewed weight release scope",
                            "evidence": "source-specific review receipt",
                        }
                        for name, rows in source_counts.items()
                    },
                },
            }
            options = dict(
                data=data,
                data_manifest_path=manifest,
                run_provenance_path=provenance,
                partition_sha=data_sha,
                partition_rows={"train": 5, "select": 600, "cal": 900},
                source_counts=source_counts,
                attestation_path=statement_path,
                license_id="other",
            )

            def check(value: dict) -> dict:
                statement_path.write_text(json.dumps(value))
                return verify_rights(**options)

            result = check(statement)
            self.assertEqual(result["public_weight_review_sha256"], "d" * 64)
            missing_review = {
                key: value
                for key, value in statement.items()
                if key != "public_weight_review"
            }
            with self.assertRaisesRegex(ValueError, "reviewed public-weight decision"):
                check(missing_review)
            missing_source = json.loads(json.dumps(statement))
            del missing_source["public_weight_review"]["source_decisions"]["external"]
            with self.assertRaisesRegex(ValueError, "review is incomplete"):
                check(missing_source)
            wrong_count = json.loads(json.dumps(statement))
            wrong_count["public_weight_review"]["source_decisions"]["external"][
                "rows"
            ] = 1
            with self.assertRaisesRegex(ValueError, "source decision is incomplete"):
                check(wrong_count)
            with self.assertRaisesRegex(ValueError, "license metadata"):
                verify_rights(**{**options, "license_id": "apache-2.0"})
            with self.assertRaisesRegex(ValueError, "holdout row counts"):
                verify_rights(
                    **{
                        **options,
                        "partition_rows": {"train": 5, "select": 599, "cal": 900},
                    }
                )
            check(statement)
            manifest.write_text(json.dumps({**data, "unexpected_change": True}))
            with self.assertRaisesRegex(ValueError, "differs from exact run"):
                verify_rights(**options)

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
