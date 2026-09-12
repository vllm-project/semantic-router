import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.training.model_classifier.sequence_repair.hydrate_annotations import (
    hydrate_records,
    sha256_text,
)


class AnnotationTests(unittest.TestCase):
    def test_aya_author_and_text_both_identify_the_reviewed_request(self):
        source = {
            "inputs": "读这段文本\u2028再总结。",
            "language_code": "zho",
            "user_id": "public-contributor",
        }
        sidecar = {
            "source": {"repository": "CohereLabs/aya_dataset"},
            "records": [
                {
                    "id": "review",
                    "source_text_sha256": sha256_text(source["inputs"]),
                    "language": "zho",
                    "group_id": "aya-author-" + sha256_text(source["user_id"]),
                    "label": "NO_FACT_CHECK_NEEDED",
                }
            ],
        }
        result = hydrate_records(sidecar, [source])
        self.assertEqual(result[0]["text"], source["inputs"])
        with self.assertRaises(ValueError):
            hydrate_records(sidecar, [{**source, "user_id": "another-contributor"}])

    def test_dolly_context_is_reconstructed_and_verified_not_discarded(self):
        source = {
            "instruction": " Extract the year. ",
            "context": " The club opened in 1910. ",
        }
        expected = "Extract the year.\n\nThe club opened in 1910."
        sidecar = {
            "source": {"repository": "databricks/databricks-dolly-15k"},
            "records": [
                {
                    "source_row": 0,
                    "source_text_sha256": sha256_text(expected),
                    "label": "NO_FACT_CHECK_NEEDED",
                }
            ],
        }
        self.assertEqual(hydrate_records(sidecar, [source])[0]["text"], expected)
        with self.assertRaises(ValueError):
            hydrate_records(sidecar, [{**source, "context": ""}])


if __name__ == "__main__":
    unittest.main()
