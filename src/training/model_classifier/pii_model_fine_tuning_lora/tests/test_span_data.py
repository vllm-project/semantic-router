import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from span_data import (
    align_record,
    assert_split_isolation,
    decode_entities,
    entity_metrics,
    load_label_contract,
    validated_spans,
)

LABELS = {
    "O": 0,
    "B-EMAIL_ADDRESS": 1,
    "I-EMAIL_ADDRESS": 2,
    "B-DOMAIN_NAME": 3,
    "I-DOMAIN_NAME": 4,
}


def record(text, spans, **metadata):
    return {
        "full_text": text,
        "spans": [
            {
                "entity_type": label,
                "start_position": start,
                "end_position": end,
                "entity_value": text[start:end],
            }
            for label, start, end in spans
        ],
        **metadata,
    }


class FakeTokenizer:
    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        return {
            "input_ids": list(range(len(self.offsets))),
            "attention_mask": [1] * len(self.offsets),
            "offset_mapping": self.offsets,
        }


class SpanDataTests(unittest.TestCase):
    def test_label_contract_rejects_reordered_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = {
                "label2id": LABELS,
                "id2label": {str(index): label for label, index in LABELS.items()},
            }
            path.write_text(json.dumps(config))
            self.assertEqual(load_label_contract(path)[0], LABELS)
            config["id2label"]["1"] = "I-EMAIL_ADDRESS"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_label_contract(path)

    def test_all_email_subwords_are_supervised_with_unicode_offsets(self):
        sample = record("中文🙂 a.b@x.io", [("EMAIL_ADDRESS", 4, 12)])
        tokenizer = FakeTokenizer(
            [
                (0, 0),
                (0, 3),
                (3, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 12),
                (0, 0),
            ]
        )
        encoded = align_record(sample, tokenizer, LABELS, 10)
        self.assertEqual(encoded["labels"], [-100, 0, 1, 2, 2, 2, 2, 2, 2, -100])
        self.assertEqual(
            decode_entities(
                sample["full_text"],
                encoded["offset_mapping"],
                [
                    (
                        next(label for label, index in LABELS.items() if index == value)
                        if value != -100  # noqa: PLR2004
                        else "O"
                    )
                    for value in encoded["labels"]
                ],
            ),
            [("EMAIL_ADDRESS", 4, 12)],
        )

    def test_overlapping_domain_cannot_overwrite_email(self):
        sample = record("a@x.io", [("EMAIL_ADDRESS", 0, 6), ("DOMAIN_NAME", 2, 6)])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validated_spans(sample, LABELS)

    def test_invalid_span_and_missing_label_fail(self):
        sample = record("a@x.io", [("EMAIL_ADDRESS", 0, 6)])
        sample["spans"][0]["entity_value"] = "different"
        with self.assertRaisesRegex(ValueError, "does not match"):
            validated_spans(sample, LABELS)
        sample["spans"][0]["entity_value"] = "a@x.io"
        sample["spans"][0]["entity_type"] = "PERSON"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validated_spans(sample, LABELS)

    def test_no_hidden_truncation_or_punctuation_expansion(self):
        sample = record("a@x.io.", [("EMAIL_ADDRESS", 0, 6)])
        with self.assertRaisesRegex(ValueError, "budget"):
            align_record(sample, FakeTokenizer([(0, 0), (0, 6), (0, 0)]), LABELS, 2)
        with self.assertRaisesRegex(ValueError, "boundary"):
            align_record(sample, FakeTokenizer([(0, 0), (0, 7), (0, 0)]), LABELS, 3)

    def test_decoder_keeps_type_transitions_and_repeated_entities(self):
        text = "a@x.io a@x.io"
        self.assertEqual(
            decode_entities(
                text,
                [(0, 2), (2, 6), (7, 13)],
                ["I-EMAIL_ADDRESS", "I-DOMAIN_NAME", "B-EMAIL_ADDRESS"],
            ),
            [("EMAIL_ADDRESS", 0, 2), ("DOMAIN_NAME", 2, 6), ("EMAIL_ADDRESS", 7, 13)],
        )

    def test_exact_metrics_do_not_reward_partial_email_or_o_tokens(self):
        rows = [
            record("a@x.io", [("EMAIL_ADDRESS", 0, 6)]),
            record("ordinary text", []),
        ]
        metrics = entity_metrics(
            rows,
            [[("EMAIL_ADDRESS", 0, 2)], [("DOMAIN_NAME", 0, 8)]],
            ["EMAIL_ADDRESS", "DOMAIN_NAME"],
        )
        self.assertEqual(metrics["micro"]["f1"], 0)
        self.assertEqual(metrics["micro"]["fp"], 2)
        self.assertEqual(metrics["micro"]["fn"], 1)
        self.assertEqual(metrics["complete_email"]["recall"], 0)
        self.assertEqual(metrics["negative_documents"]["with_false_positive"], 1)

    def test_split_isolation_tracks_templates_sources_and_values(self):
        train = record(
            "email a@x.io",
            [("EMAIL_ADDRESS", 6, 12)],
            template_family="train",
            source_group="one",
        )
        test = record(
            "mail b@y.io",
            [("EMAIL_ADDRESS", 5, 11)],
            template_family="test",
            source_group="two",
        )
        assert_split_isolation({"train": [train], "test": [test]})
        for field in ("template_family", "source_group"):
            contaminated = {**test, field: train[field]}
            with self.assertRaisesRegex(ValueError, "Cross-split"):
                assert_split_isolation({"train": [train], "test": [contaminated]})
        contaminated = record(
            "mail a@x.io",
            [("EMAIL_ADDRESS", 5, 11)],
            template_family="test",
            source_group="two",
        )
        with self.assertRaisesRegex(ValueError, "entity_value"):
            assert_split_isolation({"train": [train], "test": [contaminated]})


if __name__ == "__main__":
    unittest.main()
