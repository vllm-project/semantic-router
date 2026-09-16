"""Two arms receive the same fixed source/stratum/family sequence."""

import tempfile
import unittest
from pathlib import Path

from src.training.model_embeddings.mmbert_32k.newbase_data import FrozenCorpus
from src.training.model_embeddings.mmbert_32k.newbase_stream import (
    load_stream,
    prepare_stream,
)


class NewBaseStreamTest(unittest.TestCase):
    def test_identical_arm_streams_and_cycle_source_counts(self):
        records = [
            {"id": f"{language}:{i}", "parent_groups": [f"family:{i}"]}
            for language in ("en", "zh")
            for i in range(8)
        ]
        corpus = FrozenCorpus({}, tuple(records), "a" * 64, {"split": "train"})
        spec = {
            "seed": 125,
            "cycles": 3,
            "sources": {
                "short": {
                    "steps_per_cycle": 4,
                    "batch_queries": 2,
                    "strata": {
                        language: [
                            row["id"]
                            for row in records
                            if row["id"].startswith(language)
                        ]
                        for language in ("en", "zh")
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            first, second = prepare_stream(corpus, spec, left), prepare_stream(
                corpus, spec, right
            )
            self.assertEqual(first, second)
            self.assertEqual(first["source_counts"], {"short": 12})
            self.assertEqual(first["stratum_counts"], {"short:en": 6, "short:zh": 6})
            draws, _ = load_stream(left, corpus)
            self.assertEqual(len(draws), 12)
            (left / "draws.jsonl").write_text("[]")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_stream(left, corpus)


if __name__ == "__main__":
    unittest.main()
