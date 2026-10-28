"""Train-only entrypoint never loads development data or selects a checkpoint."""

# ruff: noqa: PLC0415

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.training.model_classifier.sequence_repair import train


class TrainOnlyArgumentTests(unittest.TestCase):
    def test_train_only_accepts_no_development_path(self):
        args = [
            "train",
            "--base",
            "unused",
            "--base-id",
            "example/tiny",
            "--base-revision",
            "fixture",
            "--method",
            "full",
            "--contract",
            "unused",
            "--train",
            "unused",
            "--output",
            "unused",
            "--train-only",
        ]
        with (
            patch.dict(sys.modules, {"torch": MagicMock()}),
            patch.object(sys, "argv", args),
            patch.object(train, "initial_artifact_receipt", return_value={}),
            patch.object(
                train,
                "load_trainable_model",
                side_effect=RuntimeError("reached loader"),
            ) as load,
            self.assertRaisesRegex(RuntimeError, "reached loader"),
        ):
            train.main()
        load.assert_called_once()

    def test_mode_conflicts_fail_before_loading(self):
        args = [
            "train",
            "--base",
            "unused",
            "--base-id",
            "example/tiny",
            "--base-revision",
            "fixture",
            "--method",
            "full",
            "--contract",
            "unused",
            "--train",
            "unused",
            "--output",
            "unused",
        ]
        for flags in (
            [],
            ["--train-only", "--dev", "unused"],
            ["--train-only", "--probe-only"],
            ["--train-only", "--selection", "source-macro-f1"],
            ["--train-only", "--positive-label", "unsafe"],
            ["--train-only", "--selection-false-positive-budget", "0"],
        ):
            with (
                self.subTest(flags=flags),
                patch.dict(sys.modules, {"torch": object()}),
                patch.object(sys, "argv", args + flags),
                patch.object(train, "load_trainable_model") as load,
                self.assertRaises(SystemExit) as error,
            ):
                train.main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name)
        for name in ("torch", "transformers", "numpy", "sklearn")
    ),
    "Optional full training dependencies",
)
class TrainOnlyUpdateTests(unittest.TestCase):
    def test_real_updates_save_only_complete_final_checkpoint(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            ModernBertConfig,
            ModernBertForSequenceClassification,
            PreTrainedTokenizerFast,
        )

        from src.training.model_classifier.sequence_repair.model import load_model

        torch.set_num_threads(1)
        torch.manual_seed(29)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = {"safe": 0, "unsafe": 1}
            config = ModernBertConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=64,
                local_attention=8,
                global_attn_every_n_layers=2,
                classifier_pooling="cls",
                classifier_dropout=0.0,
                attention_dropout=0.0,
                embedding_dropout=0.0,
                mlp_dropout=0.0,
                problem_type="single_label_classification",
                label2id=labels,
                id2label={index: name for name, index in labels.items()},
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                reference_compile=False,
            )
            config._attn_implementation = "sdpa"
            initial = ModernBertForSequenceClassification(config).float()
            initial.save_pretrained(root / "base")
            raw = Tokenizer(
                WordLevel(
                    {
                        "[PAD]": 0,
                        "[BOS]": 1,
                        "[EOS]": 2,
                        "[UNK]": 3,
                        "alpha": 4,
                        "beta": 5,
                    },
                    unk_token="[UNK]",
                )
            )
            raw.pre_tokenizer = Whitespace()
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
                        "classifier_pooling": "cls",
                    }
                )
            )
            train_file = root / "train.jsonl"
            train_file.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": str(i),
                            "group_id": str(i),
                            "source": "synthetic",
                            "text": word,
                            "label": label,
                        }
                    )
                    + "\n"
                    for i, (word, label) in enumerate(
                        [("alpha alpha", "safe"), ("beta beta", "unsafe")]
                    )
                )
            )
            output = root / "run"
            args = [
                "train",
                "--base",
                str(root / "base"),
                "--base-id",
                "example/tiny",
                "--base-revision",
                "fixture",
                "--method",
                "full",
                "--contract",
                str(contract),
                "--train",
                str(train_file),
                "--train-only",
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
                "--eval-every",
                "1",
                "--learning-rate",
                "0.001",
            ]
            tensor_to, module_to, tensor = (
                torch.Tensor.to,
                torch.nn.Module.to,
                torch.tensor,
            )

            def cpu_to(original, value, *args, **kwargs):
                return original(
                    value,
                    *(("cpu", *args[1:]) if args and args[0] == "cuda" else args),
                    **kwargs,
                )

            def cpu_tensor(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return tensor(*args, **kwargs)

            # Only device/autocast/counters are replaced; full HF loading, AdamW,
            # CE, training and native serialization/reload remain real.
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        torch.Tensor,
                        "to",
                        lambda value, *args, **kwargs: cpu_to(
                            tensor_to, value, *args, **kwargs
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        torch.nn.Module,
                        "to",
                        lambda value, *args, **kwargs: cpu_to(
                            module_to, value, *args, **kwargs
                        ),
                    )
                )
                stack.enter_context(patch.object(torch, "tensor", cpu_tensor))
                stack.enter_context(
                    patch.object(
                        torch, "autocast", return_value=contextlib.nullcontext()
                    )
                )
                for name in (
                    "synchronize",
                    "reset_peak_memory_stats",
                    "max_memory_allocated",
                    "max_memory_reserved",
                ):
                    stack.enter_context(patch.object(torch.cuda, name, return_value=0))
                stack.enter_context(patch.object(sys, "argv", args))
                evaluate = stack.enter_context(
                    patch.object(
                        train,
                        "evaluate_records",
                        side_effect=AssertionError("Development evaluation forbidden"),
                    )
                )
                read = stack.enter_context(
                    patch.object(train, "read_records", wraps=train.read_records)
                )
                save = stack.enter_context(
                    patch.object(
                        train,
                        "save_training_checkpoint",
                        wraps=train.save_training_checkpoint,
                    )
                )
                train.main()
            evaluate.assert_not_called()
            self.assertEqual(read.call_count, 1)
            self.assertEqual(save.call_count, 1)
            self.assertEqual(save.call_args.args[2], output / "last-model")
            self.assertFalse((output / "best-model").exists())
            self.assertFalse((output / "selection.json").exists())
            metadata = json.loads((output / "run.json").read_text())
            self.assertTrue(metadata["train_only"])
            self.assertEqual(metadata["dev_rows"], 0)
            self.assertEqual(metadata["dev_files"], [])
            self.assertEqual(metadata["checkpoint_policy"], "final-only")
            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["evaluation_dtype"])
            with (output / "steps.jsonl").open() as stream:
                logs = [json.loads(line) for line in stream]
            self.assertEqual([row["step"] for row in logs], [1, 2])
            final, _, _, _ = load_model(output / "last-model", contract)
            self.assertFalse(
                torch.equal(
                    initial.model.layers[0].attn.Wqkv.weight,
                    final.model.layers[0].attn.Wqkv.weight,
                )
            )
            self.assertFalse(
                torch.equal(initial.classifier.weight, final.classifier.weight)
            )
            self.assertEqual(final.config.classifier_pooling, "cls")


if __name__ == "__main__":
    unittest.main()
