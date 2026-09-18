"""Offline training and checkpoint contracts for the supported dependency stack."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        BertConfig,
        BertForSequenceClassification,
        ModernBertConfig,
        ModernBertForMaskedLM,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    torch = None

from src.training.model_classifier.safety_classifier import train
from src.training.model_classifier.safety_classifier.config import load_contract
from src.training.model_embeddings.mmbert_32k.rope_config import (
    configure_modernbert_yarn,
    verify_loaded_modernbert_yarn,
)


@unittest.skipIf(torch is None, "requires the pinned training dependencies")
class DependencyRuntimeTest(unittest.TestCase):
    def test_trainer_and_sharded_reload_avoid_accelerate_checkpoint_loaders(self):
        # Accelerate's upstream sharded loaders remain unpatched. The supported
        # Transformers loading path must not call them, including device_map.
        unsafe_apis = (
            "accelerate.load_checkpoint_and_dispatch",
            "accelerate.utils.load_checkpoint_in_model",
            "accelerate.big_modeling.load_checkpoint_in_model",
            "accelerate.utils.modeling.load_checkpoint_in_model",
        )
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            for api in unsafe_apis:
                stack.enter_context(
                    mock.patch(api, side_effect=AssertionError("unsafe loader called"))
                )
            model = BertForSequenceClassification(
                BertConfig(
                    vocab_size=16,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_labels=2,
                )
            )
            rows = [
                {
                    "input_ids": torch.tensor([1, 2, 3, 0]),
                    "attention_mask": torch.tensor([1, 1, 1, 0]),
                    "labels": torch.tensor(label),
                }
                for label in (0, 1)
            ]
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=temporary,
                    use_cpu=True,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    save_strategy="no",
                    report_to="none",
                    disable_tqdm=True,
                ),
                train_dataset=rows,
            )
            result = trainer.train()
            self.assertEqual(result.global_step, 1)
            checkpoint = Path(temporary) / "sharded"
            model.save_pretrained(checkpoint, max_shard_size="1KB")
            self.assertTrue((checkpoint / "model.safetensors.index.json").is_file())
            loaded = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, local_files_only=True, device_map="cpu"
            ).eval()
            model.eval()
            inputs = {
                key: value.unsqueeze(0)
                for key, value in rows[0].items()
                if key != "labels"
            }
            with torch.no_grad():
                torch.testing.assert_close(
                    model(**inputs).logits, loaded(**inputs).logits
                )

    def test_safety_workflow_constructs_real_training_arguments(self):
        contract = load_contract()
        with tempfile.TemporaryDirectory() as temporary:
            runtime = train._TrainingRuntime(
                stack={"TrainingArguments": TrainingArguments},
                torch=torch,
                world_size=1,
                per_device_batch=2,
                gradient_accumulation=1,
                use_bf16=False,
                source_commit=None,
                output_root=Path(temporary),
            )
            args = SimpleNamespace(
                num_train_epochs=1.0,
                learning_rate=None,
                max_steps=1,
                dataloader_num_workers=0,
                task="prompt",
            )
            task = {"selection_metric": "eval_f1"}
            actual = train._build_training_arguments(contract, task, args, runtime)
            self.assertEqual(actual.max_steps, 1)
            self.assertEqual(actual.per_device_train_batch_size, 2)
            self.assertTrue(actual.load_best_model_at_end)

    def test_modernbert_yarn_roundtrip_training_and_frequency_validation(self):
        arguments = {
            "original_max_position_embeddings": 32,
            "target_max_position_embeddings": 64,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_implementation": "sdpa",
        }
        config = ModernBertConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32,
            global_attn_every_n_layers=2,
            local_attention=4,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            cls_token_id=1,
            sep_token_id=2,
        )
        configure_modernbert_yarn(config, **arguments)
        with tempfile.TemporaryDirectory() as temporary:
            config.save_pretrained(temporary)
            restored = ModernBertConfig.from_pretrained(
                temporary, local_files_only=True
            )
        restored._attn_implementation = "sdpa"
        model = ModernBertForMaskedLM(restored)
        self.assertEqual(verify_loaded_modernbert_yarn(model, **arguments), 2)
        tokens = torch.tensor([[1, 3, 4, 2]])
        loss = model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.model.embeddings.tok_embeddings.weight.grad)
        model.model.rotary_emb.full_attention_inv_freq.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "YaRN validation failed"):
            verify_loaded_modernbert_yarn(model, **arguments)


if __name__ == "__main__":
    unittest.main()
