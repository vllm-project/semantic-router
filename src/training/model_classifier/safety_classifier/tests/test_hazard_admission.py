"""Crosswalk metadata cannot silently become reviewed Hazard supervision."""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.training.model_classifier.safety_classifier.vela_data import LABELS
from src.training.model_classifier.safety_classifier.vela_hazard_admission import (
    PROTOCOL,
    admit_corpus,
    main,
)

RUBRIC = json.dumps({"labels": dict.fromkeys(LABELS, "Test rubric")}).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture(*, positive=(), unknown=(), text="Reviewed complete text."):
    row = {
        "id": "one",
        "group_id": "source-family",
        "source": "raw-source",
        "source_revision": "fixed-source-revision",
        "source_split": "train",
        "text": text,
        "label": "unsafe",
        "targets": [1] * len(LABELS),
        "label_mask": [1] * len(LABELS),
        "annotation_provenance": "weak inherited categories",
        "violated_categories": ["broad sensitive topic"],
    }
    item = {
        key: row[key]
        for key in ("id", "group_id", "source", "source_revision", "source_split")
    }
    item.update(
        text_sha256=sha(text.encode()),
        positive_labels=list(positive),
        negative_labels=[
            name for name in LABELS if name not in positive and name not in unknown
        ],
        unknown_labels=list(unknown),
        unknown_reasons=dict.fromkeys(unknown, "Necessary context is absent"),
        binary_label="unsafe" if positive else "unknown" if unknown else "safe",
        reason="Explicit review under the complete test rubric",
    )
    source = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    review = {
        "protocol": "prediction-blind-review-v1",
        "reviewer": "test reviewer",
        "reviewer_kind": "ai",
        "labels": list(LABELS),
        "input_sha256": sha(source),
        "rubric_sha256": sha(RUBRIC),
        "items": [item],
    }
    return row, source, review


class HazardAdmissionTests(unittest.TestCase):
    def test_explicit_review_replaces_weak_annotations_and_preserves_provenance(self):
        row, source, review = fixture(positive=[LABELS[0]], unknown=[LABELS[-1]])
        before = deepcopy(review)
        result = admit_corpus(source, review, rubric_bytes=RUBRIC)
        actual = result["rows"][0]
        self.assertEqual(actual["targets"], [1] + [0] * (len(LABELS) - 1))
        self.assertEqual(actual["label_mask"][-1], 0)
        self.assertEqual(actual["label"], "unsafe")
        self.assertTrue(actual["binary_observed"])
        self.assertEqual(actual["annotation_provenance"], PROTOCOL)
        metadata = actual["source_annotations"]["input_metadata"]
        self.assertEqual(metadata["targets"], row["targets"])
        self.assertEqual(
            metadata["annotation_provenance"], row["annotation_provenance"]
        )
        self.assertEqual(metadata["violated_categories"], row["violated_categories"])
        self.assertEqual(review, before)
        actual["current_supervision"]["item"]["positive_labels"].clear()
        self.assertEqual(review, before)

    def test_partial_negatives_are_observed_but_never_binary_safe(self):
        _, source, review = fixture(unknown=[LABELS[-1]])
        result = admit_corpus(source, review, rubric_bytes=RUBRIC)
        row = result["rows"][0]
        self.assertEqual(row["label"], "unknown")
        self.assertFalse(row["binary_observed"])
        self.assertEqual(sum(row["label_mask"]), len(LABELS) - 1)
        self.assertEqual(result["manifest"]["binary_labels"], {"unknown": 1})
        review["items"][0]["binary_label"] = "safe"
        with self.assertRaisesRegex(ValueError, "Binary label"):
            admit_corpus(source, review, rubric_bytes=RUBRIC)

    def test_safe_requires_all_reviewed_negatives(self):
        _, source, review = fixture()
        result = admit_corpus(source, review, rubric_bytes=RUBRIC)
        self.assertEqual(result["rows"][0]["label"], "safe")
        self.assertEqual(result["rows"][0]["targets"], [0] * len(LABELS))
        self.assertEqual(result["rows"][0]["label_mask"], [1] * len(LABELS))

    def test_all_unknown_excluded_and_group_reserved(self):
        _, source, review = fixture(unknown=LABELS)
        result = admit_corpus(source, review, rubric_bytes=RUBRIC)
        self.assertFalse(result["rows"])
        self.assertEqual(len(result["excluded"]), 1)
        self.assertEqual(result["manifest"]["reserved_groups"], ["source-family"])
        for label in LABELS:
            counts = result["manifest"]["reviewed_supervision_including_exclusions"]
            self.assertEqual(counts[label]["unknown"], {"rows": 1, "groups": 1})
        self.assertEqual(
            result["excluded"][0]["text_sha256"], review["items"][0]["text_sha256"]
        )

    def test_missing_duplicate_overlap_and_unknown_category_rejected(self):
        _, source, original = fixture(positive=[LABELS[0]])
        for mutation in ("missing", "duplicate", "overlap", "foreign"):
            with self.subTest(mutation=mutation):
                review = deepcopy(original)
                item = review["items"][0]
                if mutation == "missing":
                    item.pop("negative_labels")
                elif mutation == "duplicate":
                    item["positive_labels"].append(LABELS[0])
                elif mutation == "overlap":
                    item["negative_labels"].append(LABELS[0])
                else:
                    item["negative_labels"][-1] = "source_only_category"
                with self.assertRaises(ValueError):
                    admit_corpus(source, review, rubric_bytes=RUBRIC)

    def test_every_unknown_has_a_reason(self):
        _, source, review = fixture(unknown=[LABELS[-1]])
        review["items"][0]["unknown_reasons"] = {}
        with self.assertRaisesRegex(ValueError, "unknown"):
            admit_corpus(source, review, rubric_bytes=RUBRIC)

    def test_source_text_group_revision_split_and_hashes_are_bound(self):
        _, source, original = fixture()
        for field in (
            "text_sha256",
            "group_id",
            "source_revision",
            "source_split",
            "source",
        ):
            with self.subTest(field=field):
                review = deepcopy(original)
                review["items"][0][field] = "changed"
                with self.assertRaisesRegex(ValueError, "changed"):
                    admit_corpus(source, review, rubric_bytes=RUBRIC)
        for field in ("input_sha256", "rubric_sha256"):
            review = deepcopy(original)
            review[field] = "changed"
            with self.assertRaisesRegex(ValueError, "changed"):
                admit_corpus(source, review, rubric_bytes=RUBRIC)

    def test_review_and_rubric_order_must_match_classifier(self):
        _, source, review = fixture()
        review["labels"].reverse()
        with self.assertRaisesRegex(ValueError, "label order"):
            admit_corpus(source, review, rubric_bytes=RUBRIC)
        review["labels"].reverse()
        changed = json.dumps({"labels": list(reversed(LABELS))}).encode()
        review["rubric_sha256"] = sha(changed)
        with self.assertRaisesRegex(ValueError, "label order"):
            admit_corpus(source, review, rubric_bytes=changed)

    def test_exact_coverage_and_unique_review_ids_required(self):
        _, source, original = fixture()
        for items in ([], original["items"] * 2):
            review = {**original, "items": items}
            with self.assertRaises(ValueError):
                admit_corpus(source, review, rubric_bytes=RUBRIC)

    def test_existing_targets_are_not_silently_rewritten(self):
        row, source, review = fixture()
        with self.assertRaisesRegex(ValueError, "Existing reviewed"):
            admit_corpus(source, review, rubric_bytes=RUBRIC, preserve_existing=True)
        row["targets"] = [0] * len(LABELS)
        source = (json.dumps(row) + "\n").encode()
        review["input_sha256"] = sha(source)
        result = admit_corpus(
            source, review, rubric_bytes=RUBRIC, preserve_existing=True
        )
        self.assertEqual(result["rows"][0]["targets"], row["targets"])
        self.assertEqual(result["rows"][0]["label_mask"], row["label_mask"])

    def test_conflicting_identical_text_requires_adjudication(self):
        row, _, review = fixture()
        other = {**row, "id": "two", "group_id": "another-source-family"}
        source = (json.dumps(row) + "\n" + json.dumps(other) + "\n").encode()
        item = deepcopy(review["items"][0])
        item.update(
            id="two",
            group_id=other["group_id"],
            positive_labels=[LABELS[0]],
            negative_labels=list(LABELS[1:]),
            binary_label="unsafe",
        )
        review["items"].append(item)
        review["input_sha256"] = sha(source)
        with self.assertRaisesRegex(ValueError, "Conflicting reviews"):
            admit_corpus(source, review, rubric_bytes=RUBRIC)

    def test_unicode_line_separators_do_not_split_jsonl_records(self):
        _, source, review = fixture(text="First\u2028second\u2029third\u0085last")
        result = admit_corpus(source, review, rubric_bytes=RUBRIC)
        self.assertEqual(len(result["rows"]), 1)

    def test_cli_receipts_and_overwrite_rejection(self):
        _, source, review = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.jsonl").write_bytes(source)
            (root / "review.json").write_text(json.dumps(review))
            (root / "rubric.json").write_bytes(RUBRIC)
            arguments = [
                "admit",
                "--input",
                str(root / "input.jsonl"),
                "--review",
                str(root / "review.json"),
                "--rubric",
                str(root / "rubric.json"),
                "--output",
                str(root / "output"),
            ]
            with patch("sys.argv", arguments), patch("builtins.print"):
                main()
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    main()
            manifest = json.loads((root / "output/manifest.json").read_text())
            self.assertEqual(
                manifest["rows_sha256"], sha((root / "output/rows.jsonl").read_bytes())
            )
            self.assertEqual(
                manifest["review_file_sha256"], sha((root / "review.json").read_bytes())
            )

    def test_cli_hashes_the_review_bytes_actually_used(self):
        _, source, review = fixture()
        original = json.dumps(review).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.jsonl").write_bytes(source)
            review_path = root / "review.json"
            review_path.write_bytes(original)
            (root / "rubric.json").write_bytes(RUBRIC)

            def mutate_file_after_review(*args, **kwargs):
                result = admit_corpus(*args, **kwargs)
                review_path.write_bytes(b"A different file after admission")
                return result

            arguments = [
                "admit",
                "--input",
                str(root / "input.jsonl"),
                "--review",
                str(review_path),
                "--rubric",
                str(root / "rubric.json"),
                "--output",
                str(root / "output"),
            ]
            target = "src.training.model_classifier.safety_classifier.vela_hazard_admission.admit_corpus"
            with (
                patch("sys.argv", arguments),
                patch(target, side_effect=mutate_file_after_review),
                patch("builtins.print"),
            ):
                main()
            manifest = json.loads((root / "output/manifest.json").read_text())
            self.assertEqual(manifest["review_file_sha256"], sha(original))
            self.assertNotEqual(
                manifest["review_file_sha256"], sha(review_path.read_bytes())
            )


if __name__ == "__main__":
    unittest.main()
