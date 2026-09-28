# Keep the optional Torch reference independent of dependency-light contract CI.
# ruff: noqa: PLC0415
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_records


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name)
        for name in ["torch", "tokenizers", "transformers"]
    ),
    "Optional Torch and HF tokenizer reference test",
)
class WindowEvaluationTests(unittest.TestCase):
    def test_overflow_windows_keep_unicode_offsets_tail_entities_and_deduplicate(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast

        vocabulary = {
            "[UNK]": 0,
            "[PAD]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "背景": 4,
            "alice@example.test": 5,
        }
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
        )

        class EntityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, input_ids, **_kwargs):
                logits = torch.zeros((*input_ids.shape, 3))
                logits[..., 0] = 1
                # Orphan I-tags intentionally exercise the public BIO policy.
                logits[..., 2] = (
                    input_ids == vocabulary["alice@example.test"]
                ).float() * 2
                return SimpleNamespace(logits=logits)

        texts = [
            "背景 " * 14 + "alice@example.test " + "背景 " * 12 + "alice@example.test",
            "背景 alice@example.test",
        ]
        rows = []
        for index, text in enumerate(texts):
            spans, cursor = [], 0
            while (start := text.find("alice@example.test", cursor)) >= 0:
                cursor = start + len("alice@example.test")
                spans.append(
                    {
                        "entity_type": "EMAIL_ADDRESS",
                        "start_position": start,
                        "end_position": cursor,
                    }
                )
            rows.append({"id": str(index), "full_text": text, "spans": spans})
        labels = {0: "O", 1: "B-EMAIL_ADDRESS", 2: "I-EMAIL_ADDRESS"}
        full, full_predictions = evaluate_records(
            EntityModel(), tokenizer, rows, labels
        )
        windowed, predictions = evaluate_records(
            EntityModel(),
            tokenizer,
            rows,
            labels,
            batch_size=3,
            window_tokens=8,
            window_stride=3,
        )
        self.assertEqual(predictions, full_predictions)
        self.assertEqual([len(entities) for entities in predictions], [2, 1])
        self.assertEqual(full["micro"]["f1"], 1.0)
        self.assertEqual(windowed["micro"]["f1"], 1.0)
        self.assertEqual(windowed["inference"]["max_forward_tokens"], 8)
        self.assertGreater(windowed["actual_tokens"]["max"], 8)
        self.assertFalse(windowed["inference"]["runtime_policy_parity"])


if __name__ == "__main__":
    unittest.main()
