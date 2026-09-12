"""The public inference helper must not silently truncate late feedback."""

# Heavy inference dependencies are optional in the contract-only test environment.
# ruff: noqa: PLC0415

import importlib.util
import unittest
from types import SimpleNamespace

HAS_RUNTIME = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
)


@unittest.skipUnless(HAS_RUNTIME, "requires the optional inference dependencies")
class InferenceBudgetTest(unittest.TestCase):
    def setUp(self):
        import torch

        from src.training.model_classifier.user_feedback_classifier.inference_feedback import (
            FeedbackDetector,
        )

        self.torch = torch
        self.detector = object.__new__(FeedbackDetector)
        self.detector.device = "cpu"
        self.detector.max_length = 512
        self.tokenizer_calls = []
        self.forward_calls = []

        class Batch(dict):
            def to(self, _device):
                return self

        def tokenize(texts, **kwargs):
            self.tokenizer_calls.append(kwargs)
            rows = [texts] if isinstance(texts, str) else texts
            lengths = [len(row.split()) + 2 for row in rows]
            width = max(lengths)
            return Batch(
                input_ids=torch.tensor(
                    [[1] * length + [0] * (width - length) for length in lengths]
                ),
                attention_mask=torch.tensor(
                    [[1] * length + [0] * (width - length) for length in lengths]
                ),
            )

        def forward(**inputs):
            self.forward_calls.append(inputs)
            return SimpleNamespace(
                logits=torch.tensor([[3.0, 1.0, 0.0, 0.0]]).repeat(
                    len(inputs["input_ids"]), 1
                )
            )

        self.detector.tokenizer = tokenize
        self.detector.model = forward

    def test_single_boundary_preserves_all_tokens_and_rejects_overflow(self):
        self.assertEqual(self.detector.classify("word " * 510).label, "SAT")
        self.assertEqual(self.forward_calls[0]["input_ids"].shape[1], 512)
        with self.assertRaisesRegex(ValueError, "513"):
            self.detector.classify("word " * 511)
        self.assertEqual(len(self.forward_calls), 1)
        self.assertTrue(all(not call["truncation"] for call in self.tokenizer_calls))

    def test_mixed_batch_rejects_overflow_before_any_forward(self):
        with self.assertRaisesRegex(ValueError, r"\(1, 513\)"):
            self.detector.classify_batch(["Thanks", "word " * 511])
        self.assertEqual(self.forward_calls, [])
        results = self.detector.classify_batch(["Thanks", "word " * 510])
        self.assertEqual([result.label for result in results], ["SAT", "SAT"])
        self.assertEqual(
            self.forward_calls[0]["attention_mask"].sum(1).tolist(), [3, 512]
        )

    def test_empty_batch_does_not_invoke_tokenizer(self):
        self.assertEqual(self.detector.classify_batch([]), [])
        self.assertEqual(self.tokenizer_calls, [])
