"""Loading must preserve task parameters, pooling, labels and full input."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.training.model_eval.constants import MODEL_REGISTRY
from src.training.model_eval.model_loading import (
    load_registered_model,
    tokenize_complete,
    validate_label_contract,
    validate_loaded_weights,
    validate_saved_modules,
)


def config(labels):
    return SimpleNamespace(
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        num_labels=len(labels),
        classifier_pooling="cls",
        max_position_embeddings=32768,
    )


def arguments(**changes):
    return SimpleNamespace(
        **dict(
            {
                "use_lora": False,
                "model_id": None,
                "revision": None,
                "dtype": "float32",
                "device": "cpu",
            },
            **changes,
        )
    )


class LoadingContractTest(unittest.TestCase):
    def test_wrong_order_and_missing_head_fail(self):
        with self.assertRaises(ValueError):
            validate_label_contract(config(["unsafe", "safe"]), ["safe", "unsafe"])
        with self.assertRaises(ValueError):
            validate_loaded_weights(
                {"missing_keys": ["head.dense.weight"]}, ["classifier"]
            )
        validate_loaded_weights({"missing_keys": ["classifier.weight"]}, ["classifier"])

    def test_pinned_config_and_precision_reach_the_correct_auto_model(self):
        for role in ("feedback", "fact-check", "intent", "pii"):
            entry = MODEL_REGISTRY[role]
            hf = Mock()
            cfg = config(entry["labels"])
            hf.AutoConfig.from_pretrained.return_value = cfg
            model = Mock(config=cfg)
            cls = (
                hf.AutoModelForTokenClassification
                if role == "pii"
                else hf.AutoModelForSequenceClassification
            )
            cls.from_pretrained.return_value = (model, {})
            with patch.dict(
                sys.modules,
                {"torch": SimpleNamespace(float32="fp32"), "transformers": hf},
            ):
                loaded, _ = load_registered_model(entry, arguments())
            self.assertIs(loaded, model)
            cls.from_pretrained.assert_called_once_with(
                entry["id"],
                revision=entry["revision"],
                config=cfg,
                torch_dtype="fp32",
                output_loading_info=True,
            )
            self.assertEqual(loaded.evaluation_identity["classifier_pooling"], "cls")
            model.to.assert_called_once_with("cpu")

    def test_vela_lora_cannot_silently_resolve_to_a_different_repo(self):
        with (
            patch.dict(sys.modules, {"torch": Mock(), "transformers": Mock()}),
            self.assertRaisesRegex(ValueError, "merged evaluation only"),
        ):
            load_registered_model(MODEL_REGISTRY["feedback"], arguments(use_lora=True))

    def test_over_budget_rejects_without_truncating(self):
        encoded = {"attention_mask": Mock()}
        encoded["attention_mask"].sum.return_value.tolist.return_value = [513]
        tokenizer = Mock(return_value=encoded)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            tokenize_complete(tokenizer, ["long input"], 512, "cpu")
        self.assertFalse(tokenizer.call_args.kwargs["truncation"])

    def test_legacy_adapter_uses_its_declared_base_and_saved_task_head(self):
        entry = {
            "id": "legacy/merged",
            "lora_id": "legacy/adapter",
            "labels": ["safe", "unsafe"],
            "type": "text_classification",
        }
        hf, peft = Mock(), Mock()
        cfg = config(entry["labels"])
        hf.AutoConfig.from_pretrained.return_value = cfg
        base = Mock(config=cfg)
        hf.AutoModelForSequenceClassification.from_pretrained.return_value = (
            base,
            {"missing_keys": ["head.dense.weight", "classifier.weight"]},
        )
        peft.PeftConfig.from_pretrained.return_value = SimpleNamespace(
            base_model_name_or_path="declared/base32k",
            revision="a" * 40,
            modules_to_save=["head", "classifier"],
        )
        peft.PeftModel.from_pretrained.return_value = Mock(config=cfg)
        tensor = Mock(shape=(2, 2))
        tensor.isfinite.return_value.all.return_value.item.return_value = True
        base.state_dict.return_value = {
            "head.dense.weight": tensor,
            "classifier.weight": tensor,
        }
        weights = Mock()
        weights.load_peft_weights.return_value = {
            "base_model.model." + key: value for key, value in base.state_dict().items()
        }
        with patch.dict(
            sys.modules,
            {
                "torch": SimpleNamespace(float32="fp32"),
                "transformers": hf,
                "peft": peft,
                "peft.utils.save_and_load": weights,
            },
        ):
            load_registered_model(entry, arguments(use_lora=True))
        call = hf.AutoModelForSequenceClassification.from_pretrained.call_args
        self.assertEqual(call.args, ("declared/base32k",))
        self.assertEqual(call.kwargs["revision"], "a" * 40)
        peft.PeftModel.from_pretrained.assert_called_once_with(
            base, "legacy/adapter", revision=None
        )

    def test_declared_but_missing_or_corrupt_adapter_head_is_rejected(self):
        tensor = Mock(shape=(2, 2))
        tensor.isfinite.return_value.all.return_value.item.return_value = True
        base = {"head.dense.weight": tensor}
        validate_saved_modules(
            base, {"base_model.model.head.dense.weight": tensor}, ["head"]
        )
        for state in ({}, {"base_model.model.head.dense.weight": Mock(shape=(1, 2))}):
            with self.assertRaisesRegex(ValueError, "exact saved task tensor"):
                validate_saved_modules(base, state, ["head"])
        tensor.isfinite.return_value.all.return_value.item.return_value = False
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_saved_modules(
                base, {"base_model.model.head.dense.weight": tensor}, ["head"]
            )

    def test_cli_selects_legacy_explicitly_and_checks_vela_adapter(self):
        import argparse  # noqa: PLC0415

        from src.training.model_eval.constants import (  # noqa: PLC0415
            COLLECTIONS,
            LANGUAGE_CODES,
            model_registry,
        )

        source = Path(__file__).parents[1] / "mom_collection_eval.py"
        node = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "parse_args"
        )
        namespace = {
            "argparse": argparse,
            "COLLECTIONS": COLLECTIONS,
            "LANGUAGE_CODES": LANGUAGE_CODES,
            "MODEL_REGISTRY": MODEL_REGISTRY,
            "model_registry": model_registry,
            "torch": Mock(),
        }
        exec(compile(ast.Module([node], []), str(source), "exec"), namespace)
        parse = namespace["parse_args"]
        self.assertEqual(parse(["--model", "feedback"]).collection, "served")
        self.assertTrue(
            parse(
                ["--collection", "legacy-mom", "--model", "feedback", "--use_lora"]
            ).use_lora
        )
        with self.assertRaises(SystemExit):
            parse(["--model", "feedback", "--use_lora"])


class NativeRoundtripTest(unittest.TestCase):
    def test_real_adapter_missing_declared_head_fails_before_inference(self):
        try:
            import torch  # noqa: PLC0415
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415
            from safetensors.torch import load_file, save_file  # noqa: PLC0415
            from tokenizers import Tokenizer  # noqa: PLC0415
            from tokenizers.models import WordLevel  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                ModernBertConfig,
                ModernBertForSequenceClassification,
                PreTrainedTokenizerFast,
            )
        except ImportError:
            self.skipTest("optional torch/PEFT dependencies unavailable")
        torch.set_num_threads(2)
        labels = ["safe", "unsafe"]
        cfg = ModernBertConfig(
            vocab_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            local_attention=8,
            id2label=dict(enumerate(labels)),
            label2id={label: i for i, label in enumerate(labels)},
            pad_token_id=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = ModernBertForSequenceClassification(cfg).eval()
            original.save_pretrained(root / "base")
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(
                    WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
                ),
                pad_token="[PAD]",
                unk_token="[UNK]",
                model_input_names=["input_ids", "attention_mask"],
            )
            tokenizer.save_pretrained(root / "base")
            adapter = get_peft_model(
                original,
                LoraConfig(
                    r=2, target_modules=["Wqkv"], modules_to_save=["head", "classifier"]
                ),
            )
            adapter.peft_config["default"].base_model_name_or_path = str(root / "base")
            with torch.no_grad():
                adapter.head.modules_to_save["default"].dense.weight.add_(0.01)
            adapter.save_pretrained(root / "adapter")
            entry = {
                "id": str(root / "base"),
                "lora_id": str(root / "adapter"),
                "type": "text_classification",
                "labels": labels,
            }
            loaded, _ = load_registered_model(entry, arguments(use_lora=True))
            torch.testing.assert_close(
                loaded.head.modules_to_save["default"].dense.weight,
                adapter.head.modules_to_save["default"].dense.weight,
                rtol=0,
                atol=0,
            )
            path = root / "adapter/adapter_model.safetensors"
            state = load_file(path)
            key = next(key for key in state if key.endswith("head.dense.weight"))
            del state[key]
            save_file(state, path)
            with self.assertRaisesRegex(ValueError, "exact saved task tensor"):
                load_registered_model(entry, arguments(use_lora=True))

    def test_real_modernbert_native_heads_and_pooling_survive(self):
        try:
            import torch  # noqa: PLC0415
            from tokenizers import Tokenizer  # noqa: PLC0415
            from tokenizers.models import WordLevel  # noqa: PLC0415
            from tokenizers.pre_tokenizers import Whitespace  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                ModernBertConfig,
                ModernBertForSequenceClassification,
                ModernBertForTokenClassification,
                PreTrainedTokenizerFast,
            )
        except ImportError:
            self.skipTest("optional torch/transformers/tokenizers unavailable")
        torch.set_num_threads(2)
        for token_task, pooling in ((False, "mean"), (False, "cls"), (True, "mean")):
            labels = ["O", "B-PERSON"] if token_task else ["safe", "unsafe"]
            cfg = ModernBertConfig(
                vocab_size=8,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=64,
                local_attention=8,
                global_attn_every_n_layers=2,
                classifier_pooling=pooling,
                id2label=dict(enumerate(labels)),
                label2id={label: i for i, label in enumerate(labels)},
                pad_token_id=0,
                bos_token_id=2,
                eos_token_id=3,
                attn_implementation="sdpa",
            )
            cls = (
                ModernBertForTokenClassification
                if token_task
                else ModernBertForSequenceClassification
            )
            original = cls(cfg).eval()
            backend = Tokenizer(
                WordLevel(
                    {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "hello": 4},
                    unk_token="[UNK]",
                )
            )
            backend.pre_tokenizer = Whitespace()
            with tempfile.TemporaryDirectory() as directory:
                original.save_pretrained(directory)
                PreTrainedTokenizerFast(
                    tokenizer_object=backend,
                    pad_token="[PAD]",
                    unk_token="[UNK]",
                    model_input_names=["input_ids", "attention_mask"],
                ).save_pretrained(directory)
                entry = {
                    "id": directory,
                    "labels": labels,
                    "type": (
                        "token_classification" if token_task else "text_classification"
                    ),
                }
                loaded, tok = load_registered_model(entry, arguments())
                self.assertEqual(loaded.config.classifier_pooling, pooling)
                self.assertEqual(
                    loaded.config._attn_implementation, cfg._attn_implementation
                )
                inputs = tok(
                    ["hello hello", "hello"], padding=True, return_tensors="pt"
                )
                with torch.no_grad():
                    torch.testing.assert_close(
                        original(**inputs).logits,
                        loaded(**inputs).logits,
                        rtol=0,
                        atol=0,
                    )


class EntryPointFailureTest(unittest.TestCase):
    def run_main(self, directory, *, parallel=False, missing=False, success=False):
        source = Path(__file__).parents[1] / "mom_collection_eval.py"
        node = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        roles = ["feedback", "intent"] if parallel else ["feedback"]
        args = SimpleNamespace(
            model=roles, parallel=parallel, device="cpu", output_dir=directory
        )
        stats = {"error": "deliberate loading failure"}
        if success:
            stats = {"accuracy": 1, "f1": 1, "latency": {"p50_ms": 1}}
        futures = [Mock(), Mock()]
        for future in futures:
            future.result.side_effect = RuntimeError("deliberate worker failure")
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.submit.side_effect = futures
        namespace = {
            "parse_args": lambda: args,
            "setup_logging": lambda _: Mock(),
            "Path": Path,
            "json": json,
            "os": SimpleNamespace(cpu_count=lambda: 2),
            "evaluate_single_model": lambda role, _: (
                "unexpected" if missing else role,
                stats,
            ),
            "ProcessPoolExecutor": lambda **_: executor,
            "as_completed": iter,
        }
        exec(compile(ast.Module([node], []), str(source), "exec"), namespace)
        namespace["main"]()

    def test_loading_error_and_missing_result_exit_nonzero_and_are_saved(self):
        for missing in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit) as error:
                    self.run_main(directory, missing=missing)
                self.assertEqual(error.exception.code, 1)
                summary = json.loads((Path(directory) / "summary.json").read_text())
                self.assertIn("error", summary["feedback"])

    def test_failed_parallel_futures_preserve_each_requested_role(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                self.run_main(directory, parallel=True)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(set(summary), {"feedback", "intent"})
            self.assertTrue(
                all("worker failure" in row["error"] for row in summary.values())
            )

    def test_complete_success_returns_normally(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_main(directory, success=True)


if __name__ == "__main__":
    unittest.main()
