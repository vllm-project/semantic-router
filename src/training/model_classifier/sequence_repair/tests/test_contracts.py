import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.training.model_classifier.sequence_repair.data import (
    assert_disjoint,
    classification_metrics,
    load_contract,
    read_records,
)
from src.training.model_classifier.sequence_repair.model import configuration_overrides
from src.training.model_classifier.sequence_repair.train import (
    length_sampling_pools,
    microbatch_loss,
    sample_length_balanced,
    sample_source_balanced,
    selection_score,
    source_sampling_pools,
)


class ContractTests(unittest.TestCase):
    def test_explicit_pooling_preserves_legacy_default_and_rejects_unknown_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text("{}")
            self.assertNotIn("classifier_pooling", configuration_overrides(path))
            for pooling in ["mean", "cls"]:
                path.write_text(json.dumps({"classifier_pooling": pooling}))
                self.assertEqual(
                    configuration_overrides(path)["classifier_pooling"], pooling
                )
            for pooling in [None, "last", "CLS"]:
                path.write_text(json.dumps({"classifier_pooling": pooling}))
                with self.assertRaises(ValueError):
                    configuration_overrides(path)

    def test_small_reviewed_source_remains_visible_with_length_and_label_balance(self):
        rows = [{"source": "weak", "length_bucket": "short", "label": "a"}] * 1000 + [
            {"source": "reviewed", "length_bucket": length, "label": label}
            for length in ["short", 32768]
            for label in ["a", "b"]
        ]
        pools = source_sampling_pools(rows, balance_lengths=True)
        rng = random.Random(17)
        counts = dict.fromkeys(range(1000, 1004), 0)
        for _ in range(4000):
            index = sample_source_balanced(pools, rng, True)
            if index in counts:
                counts[index] += 1
        for count in counts.values():
            self.assertGreater(count, 400)
            self.assertLess(count, 600)
        with self.assertRaises(ValueError):
            source_sampling_pools([{"label": "a"}], False)

    def test_source_selection_does_not_hide_a_reviewed_source_regression(self):
        metrics = {
            "macro_f1": 0.93,
            "breakdowns": {
                "source": {"weak": {"macro_f1": 0.99}, "reviewed": {"macro_f1": 0.5}}
            },
        }
        self.assertAlmostEqual(selection_score(metrics, "source-macro-f1"), 0.745)

    def test_length_sampling_keeps_a_rare_long_bucket_visible(self):
        rows = [{"length_bucket": "short", "label": "a"}] * 1000 + [
            {"length_bucket": 32768, "label": "a"}
        ]
        pools = length_sampling_pools(rows)
        rng = random.Random(42)
        long_samples = sum(
            sample_length_balanced(pools, rng, True) == len(rows) - 1
            for _ in range(1000)
        )
        self.assertGreater(long_samples, 400)
        self.assertLess(long_samples, 600)

    def test_unicode_separator_inside_json_string_is_not_a_record_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            row = {
                "id": "unicode",
                "text": "one\u2028two\u2029three\u0085four",
                "group_id": "unicode",
                "label": "x",
            }
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
            self.assertEqual(read_records([path], {"x": 0}), [row])

    def test_label_ids_are_bijective_not_resorted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {"id2label": {"0": "z", "1": "a"}, "label2id": {"z": 0, "a": 1}}
                )
            )
            self.assertEqual(load_contract(path)[0], {"z": 0, "a": 1})
            path.write_text(
                json.dumps(
                    {"id2label": {"0": "z", "1": "a"}, "label2id": {"a": 0, "z": 1}}
                )
            )
            with self.assertRaises(ValueError):
                load_contract(path)

    def test_unicode_normalized_text_and_translation_groups_cannot_leak(self):
        train = [{"id": "a", "text": "\uff21  cat", "group_id": "question-1"}]
        for dev in [
            [{"id": "b", "text": "a cat", "group_id": "other"}],
            [{"id": "b", "text": "一只猫", "group_id": "question-1"}],
        ]:
            with self.assertRaises(ValueError):
                assert_disjoint(train, dev)
        assert_disjoint(
            train, [{"id": "c", "text": "new question", "group_id": "question-2"}]
        )

    def test_conflicting_annotations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = [
                {"id": "a", "text": "Hello", "group_id": "g1", "label": "x"},
                {"id": "b", "text": "hello", "group_id": "g2", "label": "y"},
            ]
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaises(ValueError):
                read_records([path], {"x": 0, "y": 1})

    def test_macro_f1_exposes_failure_on_minority_label(self):
        metrics = classification_metrics(
            [0] * 9 + [1], [0] * 10, {0: "majority", 1: "minority"}
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.9)
        self.assertEqual(metrics["per_label"]["minority"]["recall"], 0)
        self.assertLess(metrics["macro_f1"], 0.5)

    def test_length_selection_cannot_be_dominated_by_dense_bucket(self):
        metrics = {
            "macro_f1": 0.9,
            "breakdowns": {
                "length_bucket": {"short": {"macro_f1": 0.9}, "long": {"macro_f1": 0.1}}
            },
        }
        self.assertAlmostEqual(selection_score(metrics, "length-macro-f1"), 0.5)

    def test_present_source_selection_does_not_penalize_unobserved_labels(self):
        labels = {i: str(i) for i in range(5)}
        metrics = {
            "breakdowns": {
                "source": {
                    "legacy_four": classification_metrics(
                        [0, 1, 2, 3], [0, 1, 2, 3], labels
                    ),
                    "new_negative": classification_metrics([4], [4], labels),
                }
            }
        }
        self.assertAlmostEqual(selection_score(metrics, "source-macro-f1"), 0.5)
        self.assertAlmostEqual(selection_score(metrics, "source-present-macro-f1"), 1.0)
        metrics["breakdowns"]["source"]["new_negative"] = classification_metrics(
            [4], [0], labels
        )
        self.assertAlmostEqual(selection_score(metrics, "source-present-macro-f1"), 0.5)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "Optional real tensor gradient test requires Torch",
    )
    def test_accumulated_mean_cross_entropy_matches_full_batch_gradient(self):
        import torch  # noqa: PLC0415 - optional tensor test

        torch.manual_seed(7)
        inputs = torch.randn(16, 5, dtype=torch.float64)
        labels = torch.arange(16) % 3
        full = torch.nn.Linear(5, 3, dtype=torch.float64)
        micro = torch.nn.Linear(5, 3, dtype=torch.float64)
        micro.load_state_dict(full.state_dict())
        microbatch_loss(full(inputs), labels, 1).backward()
        for start in range(0, 16, 2):
            microbatch_loss(
                micro(inputs[start : start + 2]), labels[start : start + 2], 8
            ).backward()
        for expected, actual in zip(full.parameters(), micro.parameters(), strict=True):
            torch.testing.assert_close(expected.grad, actual.grad, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
