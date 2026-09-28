"""A full Hazard checkpoint owns its encoder and independent masked-BCE head."""

# ruff: noqa: PLC0415

import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier.vela_hazard import masked_loss
from src.training.model_classifier.sequence_repair.model import load_model
from src.training.model_classifier.sequence_repair.optimization import (
    load_trainable_model,
    optimizer_groups,
    save_training_checkpoint,
)


class HazardFullEncoderTests(unittest.TestCase):
    def test_masked_bce_full_encoder_update_and_complete_reload(self):
        try:
            import torch
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from transformers import (
                ModernBertConfig,
                ModernBertForSequenceClassification,
                PreTrainedTokenizerFast,
            )
        except ImportError as error:
            self.skipTest(f"Optional training dependencies: {error}")

        torch.set_num_threads(1)
        torch.manual_seed(73)
        labels = {f"risk_{i}": i for i in range(12)}
        config = ModernBertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            local_attention=8,
            global_attn_every_n_layers=2,
            classifier_pooling="mean",
            classifier_dropout=0.0,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            mlp_dropout=0.0,
            problem_type="multi_label_classification",
            label2id=labels,
            id2label={i: label for label, i in labels.items()},
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            reference_compile=False,
        )
        config._attn_implementation = "sdpa"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = ModernBertForSequenceClassification(config).float()
            initial.save_pretrained(root / "base")
            raw = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 3}, unk_token="[UNK]"))
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]"
            )
            tokenizer.save_pretrained(root / "base")
            contract = root / "contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "label2id": labels,
                        "id2label": config.id2label,
                        "problem_type": "multi_label_classification",
                        "classifier_pooling": "mean",
                    }
                )
            )
            model, tokenizer, _, _ = load_trainable_model(
                root / "base", contract, None, "full", fresh_head=True
            )
            for key, tensor in initial.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, model.model.state_dict()[key]))
            model.train()
            inputs = {
                "input_ids": torch.tensor([[1, 4, 5, 2, 0], [1, 8, 7, 9, 2]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
            }
            targets = torch.zeros(2, 12)
            targets[0, 0] = 1
            targets[1, 1] = 1
            observed = torch.ones_like(targets)
            observed[:, -1] = 0
            optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-3, 1e-2))
            before = model.model.layers[0].attn.Wqkv.weight.detach().clone()
            loss = masked_loss(model(**inputs).logits, targets, observed)
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertTrue(bool((model.classifier.weight.grad[-1] == 0).all()))
            optimizer.step()
            self.assertFalse(
                torch.equal(before, model.model.layers[0].attn.Wqkv.weight)
            )
            model.eval()
            with torch.inference_mode():
                expected = model(**inputs).logits
            save_training_checkpoint(
                model, tokenizer, root / "saved", "full", "fixture-base", "fixture"
            )
            restored, _, restored_labels, _ = load_model(root / "saved", contract)
            self.assertEqual(restored_labels, labels)
            self.assertEqual(restored.config.problem_type, "multi_label_classification")
            self.assertFalse((root / "saved/adapter_model.safetensors").exists())
            with torch.inference_mode():
                self.assertTrue(torch.equal(expected, restored(**inputs).logits))


if __name__ == "__main__":
    unittest.main()
