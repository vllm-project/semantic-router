"""Real ModernBERT/PEFT serialization tests; no model downloads are required."""

# ruff: noqa: PLC0415

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.model_classifier.sequence_repair import complete_adapter
from src.training.model_classifier.sequence_repair.model import load_model, save_adapter
from src.training.model_classifier.sequence_repair.task_head import (
    assert_task_head_preserved,
    task_head_scope,
    verify_saved_task_head,
)


class TaskHeadStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import peft
            import torch
            import transformers
        except ImportError as error:
            raise unittest.SkipTest(
                "Install the optional training dependencies"
            ) from error
        cls.torch, cls.peft, cls.transformers = torch, peft, transformers
        torch.set_num_threads(1)

    def setUp(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.contract = self.root / "contract.json"
        self.torch.manual_seed(1729)
        config = self.transformers.ModernBertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            local_attention=16,
            global_attn_every_n_layers=2,
            num_labels=2,
            classifier_pooling="cls",
            classifier_dropout=0.0,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            mlp_dropout=0.0,
            reference_compile=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            label2id={"safe": 0, "unsafe": 1},
            id2label={0: "safe", 1: "unsafe"},
        )
        config._attn_implementation = "sdpa"
        model = self.transformers.ModernBertForSequenceClassification(config)
        model.save_pretrained(self.base)
        raw = Tokenizer(
            WordLevel(
                {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}, unk_token="[UNK]"
            )
        )
        raw.pre_tokenizer = Whitespace()
        raw.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
        )
        self.tokenizer = self.transformers.PreTrainedTokenizerFast(
            tokenizer_object=raw,
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            unk_token="[UNK]",
            model_max_length=64,
        )
        self.tokenizer.save_pretrained(self.base)
        self.contract.write_text(
            json.dumps(
                {
                    "label2id": config.label2id,
                    "id2label": config.id2label,
                    "classifier_pooling": "cls",
                }
            )
        )
        self.inputs = {
            "input_ids": self.torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 9, 2]]),
            "attention_mask": self.torch.ones((2, 5), dtype=self.torch.long),
        }
        self.legacy = self.peft.get_peft_model(
            model,
            self.peft.LoraConfig(
                task_type=self.peft.TaskType.SEQ_CLS,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["attn.Wqkv", "attn.Wo"],
                modules_to_save=["classifier"],
                bias="none",
            ),
        )
        self.legacy_path = self.root / "legacy"
        self._update(self.legacy)
        save_adapter(
            self.legacy, self.tokenizer, self.legacy_path, str(self.base), "fixture"
        )

    def _update(self, model):
        model.train()
        optimizer = self.torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        for _ in range(3):
            optimizer.zero_grad()
            output = model(**self.inputs, labels=self.torch.tensor([0, 1]))
            self.assertTrue(bool(self.torch.isfinite(output.loss)))
            output.loss.backward()
            optimizer.step()
        model.eval()

    def test_completed_head_survives_training_adapter_merge_and_reload(self):
        from safetensors.torch import load_file

        legacy_scope = verify_saved_task_head(
            self.legacy, self.legacy_path / "adapter_model.safetensors"
        )
        self.assertTrue(
            all(
                not item["trainable"] and item["saved_key"] is None
                for name, item in legacy_scope["tensors"].items()
                if name.startswith("head.")
            )
        )
        destination = self.root / "complete"
        argv = [
            "complete",
            "--base",
            str(self.base),
            "--adapter",
            str(self.legacy_path),
            "--contract",
            str(self.contract),
            "--output",
            str(destination),
        ]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            complete_adapter.main()
        original = load_file(self.legacy_path / "adapter_model.safetensors")
        completed = load_file(destination / "adapter_model.safetensors")
        saved_scope = json.loads((destination / "task-head.json").read_text())
        self.assertTrue(
            all(
                item["adapter_owned"] and item["saved_key"] is not None
                for item in saved_scope["tensors"].values()
            )
        )
        for key, tensor in original.items():
            self.torch.testing.assert_close(tensor, completed[key], atol=0, rtol=0)
        model, _, _, _ = load_model(
            self.base, self.contract, destination, trainable=True
        )
        model.eval()
        with self.torch.inference_mode():
            self.torch.testing.assert_close(
                self.legacy(**self.inputs).logits,
                model(**self.inputs).logits,
                atol=0,
                rtol=0,
            )
        before = task_head_scope(model)
        self._update(model)
        after = task_head_scope(model)
        for prefix in ("head.dense.", "head.norm."):
            self.assertTrue(
                any(
                    item["sha256"] != before["tensors"][name]["sha256"]
                    for name, item in after["tensors"].items()
                    if name.startswith(prefix)
                )
            )
        trained = self.root / "trained"
        save_adapter(model, self.tokenizer, trained, str(self.base), "fixture")
        restored, _, _, _ = load_model(self.base, self.contract, trained)
        assert_task_head_preserved(after, restored)
        restored.eval()
        with self.torch.inference_mode():
            expected = model(**self.inputs).logits
            self.torch.testing.assert_close(
                expected, restored(**self.inputs).logits, atol=0, rtol=0
            )
            merged = restored.merge_and_unload(safe_merge=True)
            self.torch.testing.assert_close(
                expected, merged(**self.inputs).logits, atol=1e-6, rtol=1e-6
            )
        assert_task_head_preserved(after, merged)
        merged_path = self.root / "merged"
        merged.save_pretrained(merged_path)
        reloaded = (
            type(merged)
            .from_pretrained(
                merged_path, attn_implementation="sdpa", reference_compile=False
            )
            .eval()
        )
        assert_task_head_preserved(after, reloaded)
        with self.torch.inference_mode():
            self.torch.testing.assert_close(
                merged(**self.inputs).logits,
                reloaded(**self.inputs).logits,
                atol=0,
                rtol=0,
            )

    def test_missing_trainable_head_and_corrupted_saved_classifier_are_rejected(self):
        from safetensors.torch import load_file, save_file

        self.legacy.get_base_model().head.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "omitted effective task tensor"):
            verify_saved_task_head(
                self.legacy, self.legacy_path / "adapter_model.safetensors"
            )
        self.legacy.get_base_model().head.requires_grad_(False)
        weights = load_file(self.legacy_path / "adapter_model.safetensors")
        key = "base_model.model.classifier.weight"
        weights[key] = weights[key] + 0.5
        corrupt = self.root / "corrupt.safetensors"
        save_file(weights, corrupt)
        with self.assertRaisesRegex(ValueError, "changed effective task tensor"):
            verify_saved_task_head(self.legacy, corrupt)


if __name__ == "__main__":
    unittest.main()
