import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training.eikos.rights import (
    NONCOMMERCIAL_SCHEMA,
    NONCOMMERCIAL_SCOPE,
    PILOT_HOLDOUT_SOURCE_COUNTS,
    SCOPE,
    SPLIT_NAMES,
    clean_manifest,
    verify_clean_files,
    verify_noncommercial_package_attestation,
)
from training.model.data import file_sha256


class CleanRightsReceiptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = [self.root / name for name in SPLIT_NAMES]
        for index, path in enumerate(self.paths):
            path.write_text(
                json.dumps({"id": f"example-{index}"}) + "\n", encoding="utf-8"
            )
        self.manifest = self.root / "rights_clean.manifest.json"
        self.receipt = {
            "schema_version": "decision2-rights-clean-splits/1",
            "publication_eligible": True,
            "publication_scope": SCOPE,
            "source_rights": [{"name": "fixture", "license": "CC0-1.0", "rows": 3}],
            "publication_conditions": ["retain attribution"],
            "overlap_audits": {"train_select": 0},
            "outputs": {
                path.name: {
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                    "rows": 1,
                }
                for path in self.paths
            },
        }
        self.manifest.write_text(json.dumps(self.receipt), encoding="utf-8")

    def test_accept_matching_clean_receipt(self):
        receipt = verify_clean_files(self.manifest, *self.paths, (1, 1, 1))
        self.assertTrue(receipt["publication_eligible"])

    def test_reject_restricted_or_mismatched_receipt(self):
        expected = {
            name: item["sha256"] for name, item in self.receipt["outputs"].items()
        }
        with self.assertRaisesRegex(ValueError, "Restricted pilot"):
            clean_manifest(
                self.manifest,
                {
                    **expected,
                    "select.jsonl": "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38",
                },
                dict.fromkeys(SPLIT_NAMES, 1),
            )
        self.receipt["publication_eligible"] = False
        self.manifest.write_text(json.dumps(self.receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "rights-cleared"):
            verify_clean_files(self.manifest, *self.paths, (1, 1, 1))

    def test_reject_unexpected_split_filenames(self):
        with self.assertRaisesRegex(ValueError, "filenames"):
            verify_clean_files(
                self.manifest,
                self.root / "old.train.jsonl",
                self.paths[1],
                self.paths[2],
                (1, 1, 1),
            )

    def test_noncommercial_package_attestation_is_bound_to_every_source(self):
        package = {
            "training_data_manifest_sha256": "1" * 64,
            "training_provenance_sha256": "2" * 64,
            "training_data_sha256": {
                "train": "3" * 64,
                "select": "4" * 64,
                "cal_audited_only": "5" * 64,
            },
            "training_source_counts": {"upstream:A": 5, "upstream:B": 3},
        }
        statement = {
            "schema_version": NONCOMMERCIAL_SCHEMA,
            "noncommercial_use": True,
            "publication_scope": NONCOMMERCIAL_SCOPE,
            "no_raw_training_rows": True,
            "data_manifest_sha256": "1" * 64,
            "training_provenance_sha256": "2" * 64,
            "data_sha256": package["training_data_sha256"],
            "source_groups": {"upstream:A": "cc-by-nc", "upstream:B": "cc-by"},
            "holdout_source_counts": PILOT_HOLDOUT_SOURCE_COUNTS,
            "holdout_groups": {
                name: dict.fromkeys(counts, "cc-by-nc")
                for name, counts in PILOT_HOLDOUT_SOURCE_COUNTS.items()
            },
            "rights_conditions": {
                "cc-by-nc": {
                    "terms": "noncommercial research; attribution",
                    "evidence": "https://example.org/a",
                },
                "cc-by": {"terms": "attribution", "evidence": "https://example.org/b"},
            },
        }
        path = self.root / "research-attestation.json"
        path.write_text(json.dumps(statement), encoding="utf-8")
        self.assertEqual(
            verify_noncommercial_package_attestation(path, package), statement
        )
        statement["source_groups"].pop("upstream:B")
        path.write_text(json.dumps(statement), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact noncommercial"):
            verify_noncommercial_package_attestation(path, package)


if __name__ == "__main__":
    unittest.main()
