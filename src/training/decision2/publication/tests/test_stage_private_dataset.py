"""Prevent raw restricted-source upload and unbound split packaging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from publication import stage_private_dataset as private


class PrivateDatasetStageTest(unittest.TestCase):
    def test_rejects_restricted_source_even_with_valid_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for name in ("rights_clean.train.jsonl", "select.jsonl", "cal.jsonl"):
                (source / name).write_text("{}\n")
            outputs = {
                name: {"sha256": private.sha(source / name), "rows": 1}
                for name in ("rights_clean.train.jsonl", "select.jsonl", "cal.jsonl")
            }
            manifest = {
                "derivation_version": "decision2-goemotions-human-v2/1",
                "publication_eligible": True,
                "publication_scope": private.SCOPE,
                "outputs": outputs,
                "counts": {
                    "source": {
                        "google_goemotions_official_train": 2800,
                        "tweeteval_train:emotion": 1,
                    }
                },
                "source_rights": [{"source": "GoEmotions"}],
                "overlap_audits": {"train_vs_select": {}},
            }
            (source / "rights_clean.manifest.json").write_text(json.dumps(manifest))
            expected = {name: private.sha(source / name) for name in private.EXPECTED}
            with mock.patch.dict(private.EXPECTED, expected, clear=True):
                with self.assertRaisesRegex(ValueError, "excluded"):
                    private.verify(source)
                manifest["counts"]["source"].pop("tweeteval_train:emotion")
                (source / "rights_clean.manifest.json").write_text(json.dumps(manifest))
                private.EXPECTED["rights_clean.manifest.json"] = private.sha(
                    source / "rights_clean.manifest.json"
                )
                self.assertEqual(private.verify(source), manifest)
                (source / "cal.jsonl").write_text('{"changed":true}\n')
                with self.assertRaisesRegex(ValueError, "cal.jsonl"):
                    private.verify(source)


if __name__ == "__main__":
    unittest.main()
