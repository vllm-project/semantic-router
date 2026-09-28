"""Actual tiny ModernBERT token training; optional dependencies are not mocked."""

# ruff: noqa: PLC0415

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT))
# Script entrypoints import their siblings from this directory.
from evaluate import load_model  # noqa: E402
from full_training import (  # noqa: E402
    optimizer_groups,
    save_full_checkpoint,
    tensor_receipt,
    validate_method,
    verify_full_checkpoint,
)

DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)


class MethodTests(unittest.TestCase):
    def test_legacy_default_requires_adapter_and_full_rejects_adapter(self):
        validate_method("lora", "adapter", False)
        validate_method("full", None, True)
        for arguments in (
            ("lora", None, False),
            ("lora", "adapter", True),
            ("full", "adapter", False),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_method(*arguments)


@unittest.skipUnless(
    DEPENDENCIES, "Optional Torch/Transformers token training dependencies"
)
class FullTokenTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import (
            ModernBertConfig,
            ModernBertForMaskedLM,
            PreTrainedTokenizerFast,
        )

        torch.set_num_threads(2)
        torch.manual_seed(31)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        entities = [
            "AGE",
            "CREDIT_CARD",
            "DATE_TIME",
            "DOMAIN_NAME",
            "EMAIL_ADDRESS",
            "GPE",
            "IBAN_CODE",
            "IP_ADDRESS",
            "NRP",
            "ORGANIZATION",
            "PERSON",
            "PHONE_NUMBER",
            "STREET_ADDRESS",
            "TITLE",
            "US_DRIVER_LICENSE",
            "US_SSN",
            "ZIP_CODE",
        ]
        labels = ["O"] + [
            f"{prefix}-{entity}" for entity in entities for prefix in ("B", "I")
        ]
        self.label_to_id = {label: index for index, label in enumerate(labels)}
        self.contract = self.root / "contract.json"
        self.contract.write_text(
            json.dumps(
                {"label2id": self.label_to_id, "id2label": dict(enumerate(labels))}
            )
        )
        vocabulary = {
            word: index
            for index, word in enumerate(
                [
                    "[PAD]",
                    "[UNK]",
                    "[BOS]",
                    "[EOS]",
                    "[MASK]",
                    "Alice",
                    "reads",
                    "a",
                    "book",
                    ".",
                    "The",
                    "page",
                    "is",
                    "blank",
                ]
            )
        }
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)]
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            mask_token="[MASK]",
            model_input_names=["input_ids", "attention_mask"],
        )
        config = ModernBertConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            local_attention=8,
            global_attn_every_n_layers=2,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            cls_token_id=2,
            sep_token_id=3,
            attention_dropout=0.0,
            classifier_dropout=0.0,
        )
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        config._attn_implementation = "sdpa"
        self.source = ModernBertForMaskedLM(config)
        with torch.no_grad():
            self.source.head.norm.weight.fill_(3)
        self.source.save_pretrained(self.base)
        tokenizer.save_pretrained(self.base)

    def fresh(self):
        return load_model(self.base, self.contract, trainable=True, fresh_head=True)

    def test_fresh_complete_head_preserves_every_encoder_tensor(self):
        import torch

        model, _, _, _ = self.fresh()
        self.assertEqual(type(model).__name__, "ModernBertForTokenClassification")
        for name, value in self.source.model.state_dict().items():
            self.assertTrue(torch.equal(value, model.model.state_dict()[name]), name)
        self.assertFalse(
            torch.equal(model.head.dense.weight, self.source.head.dense.weight)
        )
        self.assertTrue(
            torch.equal(model.head.norm.weight, torch.ones_like(model.head.norm.weight))
        )
        self.assertTrue(
            torch.equal(model.classifier.bias, torch.zeros_like(model.classifier.bias))
        )
        self.assertEqual(model.classifier.weight.shape, (35, 32))

    def test_real_update_complete_state_and_exact_native_reload_without_peft(self):
        import torch

        model, tokenizer, _, _ = self.fresh()
        before = tensor_receipt(model)
        groups = optimizer_groups(model, 1e-3, 2e-3)
        self.assertEqual([group["name"] for group in groups], ["encoder", "task_head"])
        self.assertEqual(
            sum(len(group["params"]) for group in groups), len(list(model.parameters()))
        )
        optimizer = torch.optim.AdamW(groups)
        batch = tokenizer(
            ["Alice reads a book .", "The page is blank ."],
            return_tensors="pt",
            padding=True,
        )
        labels = torch.zeros_like(batch["input_ids"])
        labels[:, 1] = self.label_to_id["B-PERSON"]
        labels[batch["attention_mask"] == 0] = -100
        model.train()
        loss = model(**batch, labels=labels).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        after = tensor_receipt(model)
        for prefix in (
            "model.embeddings.tok_embeddings.",
            "model.layers.",
            "head.dense.",
            "head.norm.",
            "classifier.",
        ):
            self.assertTrue(
                any(
                    before[name] != value
                    for name, value in after.items()
                    if name.startswith(prefix)
                ),
                prefix,
            )
        model.eval()
        with torch.no_grad():
            expected = model(**batch).logits
        output = self.root / "trained"
        save_full_checkpoint(
            model, tokenizer, output, {"base_revision": "fixture", "test_used": False}
        )
        with patch.dict(sys.modules, {"peft": None}):
            restored, _, _, _ = load_model(output, self.contract)
        restored.eval()
        self.assertEqual(tensor_receipt(restored), after)
        with torch.no_grad():
            torch.testing.assert_close(
                expected, restored(**batch).logits, atol=0, rtol=0
            )

    def test_base_without_fresh_flag_and_inference_fresh_flag_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "token architecture"):
            load_model(self.base, self.contract)
        with self.assertRaisesRegex(ValueError, "only valid for full training"):
            load_model(self.base, self.contract, fresh_head=True)

    def test_missing_encoder_is_never_randomly_repaired(self):
        from safetensors.torch import load_file, save_file

        path = self.base / "model.safetensors"
        weights = load_file(str(path))
        weights.pop("model.layers.0.attn.Wqkv.weight")
        save_file(weights, str(path), metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "encoder tensors"):
            self.fresh()

    @unittest.skipUnless(
        importlib.util.find_spec("peft"), "Optional PEFT compatibility test"
    )
    def test_legacy_adapter_restores_its_complete_saved_token_head(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from train_repair import save_adapter

        model, tokenizer, _, _ = self.fresh()
        adapter = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=2,
                lora_alpha=4,
                target_modules=["Wqkv"],
                modules_to_save=["head", "classifier"],
            ),
        )
        with torch.no_grad():
            adapter.base_model.model.head.modules_to_save["default"].dense.weight.add_(
                0.25
            )
            adapter.base_model.model.classifier.modules_to_save["default"].bias.add_(
                0.1
            )
        adapter.eval()
        batch = tokenizer("Alice reads a book .", return_tensors="pt")
        with torch.no_grad():
            expected = adapter(**batch).logits
        path = self.root / "adapter"
        save_adapter(
            adapter, tokenizer, path, str(self.base), "fixture", self.label_to_id
        )
        restored, _, _, _ = load_model(self.base, self.contract, adapter=path)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                expected, restored(**batch).logits, atol=0, rtol=0
            )

    def test_missing_native_head_and_label_reordering_are_rejected(self):
        from safetensors.torch import load_file, save_file

        model, tokenizer, _, _ = self.fresh()
        path = self.root / "native"
        save_full_checkpoint(model, tokenizer, path, {})
        weights = load_file(str(path / "model.safetensors"))
        weights.pop("head.dense.weight")
        save_file(weights, str(path / "model.safetensors"), metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "complete token head"):
            load_model(path, self.contract)
        config = json.loads((path / "config.json").read_text())
        config["label2id"]["O"] = 1
        (path / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "label order"):
            load_model(path, self.contract)

    def test_storage_rejects_missing_shape_dtype_and_nonfinite_tensors(self):
        import torch
        from safetensors.torch import load_file, save_file

        model, tokenizer, _, _ = self.fresh()
        path = self.root / "native"
        save_full_checkpoint(model, tokenizer, path, {})
        # Keep an independent snapshot: load_file tensors share the checkpoint's
        # mmap, which is invalidated when each corruption fixture truncates it.
        original = {
            name: tensor.clone()
            for name, tensor in load_file(str(path / "model.safetensors")).items()
        }
        key = "classifier.weight"
        for kind in ("missing", "shape", "dtype", "nonfinite"):
            weights = dict(original)
            if kind == "missing":
                weights.pop(key)
            elif kind == "shape":
                weights[key] = weights[key][:1]
            elif kind == "dtype":
                weights[key] = weights[key].to(torch.float16)
            else:
                weights[key] = torch.full_like(weights[key], float("nan"))
            save_file(
                weights, str(path / "model.safetensors"), metadata={"format": "pt"}
            )
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "tensor"):
                verify_full_checkpoint(model, path)

    def test_actual_full_cpu_cli_train_export_and_reload(self):
        self.run_full_cpu_cli()

    def test_actual_document_mean_cli_train_export_and_reload(self):
        self.run_full_cpu_cli("document_mean")

    def test_actual_entity_document_mean_cli_train_export_and_reload(self):
        self.run_full_cpu_cli("entity_document_mean")

    def run_full_cpu_cli(self, normalization=None):
        rows = [
            {
                "id": "positive",
                "full_text": "Alice reads a book .",
                "spans": [
                    {
                        "entity_type": "PERSON",
                        "entity_value": "Alice",
                        "start_position": 0,
                        "end_position": 5,
                    }
                ],
            },
            {"id": "negative", "full_text": "The page .", "spans": []},
        ]
        train = self.root / "train.jsonl"
        train.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = self.root / "run"
        common = [
            "--base",
            str(self.base),
            "--base-id",
            "test/tiny",
            "--base-revision",
            "fixture",
            "--config",
            str(self.contract),
            "--method",
            "full",
        ]
        command = [
            sys.executable,
            str(SCRIPT / "train_repair.py"),
            *common,
            "--fresh-head",
            "--train",
            str(train),
            "--dev",
            str(train),
            "--output",
            str(output),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--accumulate",
            "1",
            "--max-length",
            "64",
            "--learning-rate",
            ".001",
            "--head-learning-rate",
            ".002",
            "--eval-every",
            "2",
            "--evaluation-dtype",
            "float32",
            "--device",
            "cpu",
        ]
        if normalization is not None:
            command.extend(["--loss-normalization", normalization])
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run = json.loads((output / "run.json").read_text())
        self.assertEqual(run["method"], "full")
        self.assertEqual(run["evaluation_dtype"], "float32")
        self.assertEqual(run["loss_normalization"], normalization or "token_mean")
        self.assertTrue(run["fresh_head"])
        self.assertEqual(
            run["initial_artifact"]["files"]["model.safetensors"],
            hashlib.sha256((self.base / "model.safetensors").read_bytes()).hexdigest(),
        )
        self.assertFalse((output / "best-adapter").exists())
        trained, _, _, _ = load_model(output / "last-model", self.contract)
        self.assertNotEqual(run["initial_tensors"], tensor_receipt(trained))
        exported = self.root / "export"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT / "export_repair.py"),
                *common,
                "--checkpoint",
                str(output / "last-model"),
                "--run-manifest",
                str(output / "run.json"),
                "--output",
                str(exported),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        final, _, _, _ = load_model(exported, self.contract)
        self.assertEqual(tensor_receipt(trained), tensor_receipt(final))
        evidence = json.loads((exported / "merge_provenance.json").read_text())
        self.assertTrue(evidence["merge_probe"]["reload_logits_bitwise_equal"])


if __name__ == "__main__":
    unittest.main()
