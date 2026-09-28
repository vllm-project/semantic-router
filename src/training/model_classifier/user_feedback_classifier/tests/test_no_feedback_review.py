import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.model_classifier.user_feedback_classifier.vela_data import (
    TRAIN_AND_VALIDATION_PERCENT,
    TRAIN_PERCENT,
)
from src.training.model_classifier.user_feedback_classifier.vela_no_feedback import (
    main,
    project_review,
)


class NoFeedbackReviewTests(unittest.TestCase):
    def fixture(self):
        source, items = [], []
        for index in range(30):
            text = f"A new task about subject {index}"
            source.append(
                {
                    "UtterranceId": 0,
                    "Role": "User",
                    "State": "NEWTOPIC",
                    "Content": text,
                }
            )
            group = f"wildfeedback:conversation:{index}"
            part = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
            if part >= TRAIN_AND_VALIDATION_PERCENT:
                continue
            items.append(
                {
                    "source_index": index,
                    "group_id": group,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "split": "train" if part < TRAIN_PERCENT else "validation",
                    "label": "NO_FEEDBACK",
                    "language": "en",
                }
            )
        return {"items": items, "review": "test review"}, source

    def test_only_exact_reviewed_rows_are_materialized(self):
        review, source = self.fixture()
        review["items"][0]["label"] = None
        result = project_review(review, source)
        self.assertEqual(sum(map(len, result.values())), len(review["items"]) - 1)
        self.assertTrue(
            all(
                row["label"] == "NO_FEEDBACK"
                for rows in result.values()
                for row in rows
            )
        )

    def test_cli_consumes_an_explicit_review_artifact(self):
        review, source = self.fixture()
        source_bytes = json.dumps(source).encode()
        review.update(
            source="fixture",
            revision="fixed-revision",
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            policy="Exact reviewed rows only",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.json").write_bytes(source_bytes)
            (root / "review.json").write_text(json.dumps(review))
            argv = [
                "vela_no_feedback",
                "--source",
                str(root / "source.json"),
                "--review",
                str(root / "review.json"),
                "--output",
                str(root / "output"),
            ]
            with patch("sys.argv", argv), patch("builtins.print"):
                main()
            manifest = json.loads((root / "output/manifest.json").read_text())
            self.assertEqual(manifest["reviewed_rows"], len(review["items"]))
            self.assertEqual(
                manifest["review_sha256"],
                hashlib.sha256((root / "review.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                sum(item["rows"] for item in manifest["files"].values()),
                len(review["items"]),
            )

    def test_changed_text_and_repartition_are_refused(self):
        for field, value in [("text_sha256", "wrong"), ("split", "test")]:
            review, source = self.fixture()
            review["items"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                project_review(review, source)


if __name__ == "__main__":
    unittest.main()
