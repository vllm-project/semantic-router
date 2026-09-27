"""The public Hazard loop honors ordered draws with real CPU updates and saving."""

# ruff: noqa: PLC0415

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.model_classifier.safety_classifier import train_vela_hazard as train
from src.training.model_classifier.sequence_repair.data import file_receipts
from src.training.model_classifier.sequence_repair.training_order import (
    ordered_ids_sha256,
)


class TrainingOrderArgumentTests(unittest.TestCase):
    def test_cli_rejects_ambiguous_sampling_before_loading_model(self):
        common = [
            "train",
            "--base",
            "unused",
            "--base-id",
            "example/tiny-base",
            "--base-revision",
            "unused",
            "--method",
            "full",
            "--contract",
            "unused",
            "--train",
            "unused",
            "--dev",
            "unused",
            "--output",
            "unused",
            "--training-order",
            "unused",
        ]
        for flags in [
            ["--source-balanced-sampling"],
            ["--length-balanced-sampling"],
            ["--source-weights", '{"source":1}'],
        ]:
            with self.subTest(flags=flags), patch.dict(
                sys.modules, {"torch": object()}
            ), patch.object(sys, "argv", common + flags), patch.object(
                train, "load_trainable_model"
            ) as load, self.assertRaisesRegex(
                ValueError, "cannot also configure sampling"
            ):
                train.main()
            load.assert_not_called()


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name)
        for name in ("torch", "transformers", "numpy", "sklearn")
    ),
    "Optional full training dependencies",
)
class HazardTrainingOrderTests(unittest.TestCase):
    def test_public_loop_records_actual_order_and_keeps_unconfigured_sampler(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import (
            ModernBertConfig,
            ModernBertForSequenceClassification,
            PreTrainedTokenizerFast,
        )

        from src.training.model_classifier.sequence_repair.model import load_model

        torch.set_num_threads(1)
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = {f"risk_{index}": index for index in range(12)}
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
                    {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]"
                )
            )
            raw.pre_tokenizer = Whitespace()
            raw.post_processor = TemplateProcessing(
                single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)]
            )
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
            datasets = {}
            for split in ("train", "dev"):
                rows = [
                    {
                        "id": f"{split}-{index}",
                        "group_id": f"{split}-{index}",
                        "source": "fixture",
                        "text": f"{split} " + "word " * (index * 2 + 1),
                        "label": "safe" if index == 0 else "unsafe",
                        "targets": [int(index != 0)] * 12,
                        "label_mask": [1] * 12,
                    }
                    for index in range(3 if split == "train" else 2)
                ]
                (root / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                datasets[split] = rows
            ids = ["train-2", "train-0", "train-0", "train-1", "train-2", "train-1"] * 2
            order = root / "order.json"
            order.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "train_files": file_receipts([root / "train.jsonl"]),
                        "eligible_ids_sha256": ordered_ids_sha256(
                            [row["id"] for row in datasets["train"]]
                        ),
                        "steps": 2,
                        "global_batch": 6,
                        "ids": ids,
                        "ids_sha256": ordered_ids_sha256(ids),
                    }
                )
            )
            arguments = [
                "train",
                "--base",
                str(root / "base"),
                "--base-id",
                "example/tiny-base",
                "--base-revision",
                "fixture",
                "--method",
                "full",
                "--contract",
                str(contract),
                "--train",
                str(root / "train.jsonl"),
                "--dev",
                str(root / "dev.jsonl"),
                "--steps",
                "2",
                "--batch-size",
                "2",
                "--accumulate",
                "3",
                "--max-length",
                "64",
                "--microbatch-token-budget",
                "16",
                "--eval-every",
                "2",
                "--evaluation-dtype",
                "float32",
                "--loss-normalization",
                "taxonomy",
                "--learning-rate",
                "0.01",
                "--head-learning-rate",
                "0.02",
            ]
            tensor_to, module_to = torch.Tensor.to, torch.nn.Module.to

            def cpu_tensor(value, *args, **kwargs):
                return tensor_to(
                    value,
                    *(("cpu", *args[1:]) if args and args[0] == "cuda" else args),
                    **kwargs,
                )

            def cpu_module(value, *args, **kwargs):
                return module_to(
                    value,
                    *(("cpu", *args[1:]) if args and args[0] == "cuda" else args),
                    **kwargs,
                )

            tensor = torch.tensor

            def cpu_tensor_factory(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return tensor(*args, **kwargs)

            # Only accelerator placement/autocast is redirected. The public loop,
            # real model, masked loss, AdamW, selector, and full save/reload execute.
            with patch.object(torch.Tensor, "to", cpu_tensor), patch.object(
                torch.nn.Module, "to", cpu_module
            ), patch.object(torch, "tensor", cpu_tensor_factory), patch.object(
                torch, "autocast", return_value=contextlib.nullcontext()
            ):
                for mode in ("ordered", "sampled"):
                    output = root / mode
                    flags = (
                        ["--training-order", str(order)]
                        if mode == "ordered"
                        else ["--source-balanced-sampling"]
                    )
                    with patch.object(
                        sys, "argv", [*arguments, "--output", str(output), *flags]
                    ), patch.object(
                        train, "sample_grouped", wraps=train.sample_grouped
                    ) as sampler:
                        train.main()
                    self.assertEqual(sampler.call_count, 0 if mode == "ordered" else 12)
                    restored, _, _, _ = load_model(output / "last-model", contract)
                    self.assertFalse(
                        torch.equal(
                            initial.model.layers[0].attn.Wqkv.weight,
                            restored.model.layers[0].attn.Wqkv.weight,
                        )
                    )
                    run = json.loads((output / "run.json").read_text())
                    self.assertEqual(run["base_model"], "example/tiny-base")
                    trace_path = output / "actual-training-order.jsonl"
                    if mode == "ordered":
                        trace = [
                            json.loads(line)
                            for line in trace_path.read_text().splitlines()
                        ]
                        self.assertEqual(
                            [key for row in trace for key in row["planned_ids"]], ids
                        )
                        self.assertEqual([row["step"] for row in trace], [1, 2])
                        self.assertNotEqual(trace[0]["microbatch_ids"][0], ids[:2])
                        self.assertEqual(run["examples_per_optimizer_step"], 6)
                        self.assertEqual(
                            json.loads(
                                (output / "training-order-completed.json").read_text()
                            )["completed_draws"],
                            12,
                        )
                    else:
                        self.assertIsNone(run["training_order"])
                        self.assertFalse(trace_path.exists())


if __name__ == "__main__":
    unittest.main()
