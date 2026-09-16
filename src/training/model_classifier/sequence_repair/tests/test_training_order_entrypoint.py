"""Exercise explicit sequence replay through the real tiny-model training CLI."""

# ruff: noqa: PLC0415

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.model_classifier.sequence_repair import train
from src.training.model_classifier.sequence_repair.data import file_receipts
from src.training.model_classifier.sequence_repair.training_order import (
    ordered_ids_sha256,
)


class TrainingOrderArgumentTests(unittest.TestCase):
    def test_conflicting_sampling_fails_before_model_loading(self):
        args = [
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
        for flags in (
            ["--balanced-sampling"],
            ["--length-balanced-sampling"],
            ["--source-balanced-sampling"],
            ["--source-weights", '{"source":1}'],
        ):
            with self.subTest(flags=flags), patch.dict(
                sys.modules, {"torch": object()}
            ), patch.object(sys, "argv", args + flags), patch.object(
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
class SequenceTrainingOrderTests(unittest.TestCase):
    def test_replay_regrouping_and_default_sampler_use_real_updates(self):
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
            labels = {"safe": 0, "unsafe": 1}
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
                        "gamma": 6,
                    },
                    unk_token="[UNK]",
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
                        "text": split + " " + word * (index * 2 + 1),
                        "label": "unsafe" if index == 1 else "safe",
                    }
                    for index, word in enumerate(("alpha ", "beta ", "gamma "))
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
            args = [
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
                "--eval-every",
                "2",
                "--evaluation-dtype",
                "float32",
                "--learning-rate",
                "0.0001",
            ]
            for extra, message in (
                (["--steps", "3"], "optimizer budget"),
                (["--max-length", "6"], "eligible TRAIN IDs"),
            ):
                output = root / "invalid"
                with patch.object(
                    sys,
                    "argv",
                    [
                        *args,
                        "--output",
                        str(output),
                        "--training-order",
                        str(order),
                        *extra,
                    ],
                ), self.assertRaisesRegex(ValueError, message):
                    train.main()
                self.assertFalse(output.exists())
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

            # Placement/autocast and GPU counters only: the model, loss, AdamW,
            # public loop, evaluation and complete checkpoint reload stay real.
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
                states = {}
                for mode in ("ordered", "regrouped", "sampled"):
                    flags = (
                        ["--source-balanced-sampling"]
                        if mode == "sampled"
                        else ["--training-order", str(order)]
                    )
                    if mode == "regrouped":
                        flags += ["--microbatch-token-budget", "16"]
                    output = root / mode
                    with patch.object(
                        sys, "argv", [*args, "--output", str(output), *flags]
                    ), patch.object(
                        train,
                        "sample_source_balanced",
                        wraps=train.sample_source_balanced,
                    ) as sampler:
                        train.main()
                    self.assertEqual(sampler.call_count, 12 if mode == "sampled" else 0)
                    model, _, _, _ = load_model(output / "last-model", contract)
                    self.assertFalse(
                        torch.equal(
                            initial.model.layers[0].attn.Wqkv.weight,
                            model.model.layers[0].attn.Wqkv.weight,
                        )
                    )
                    states[mode] = model.state_dict()
                    metadata = json.loads((output / "run.json").read_text())
                    self.assertEqual(metadata["base_model"], "example/tiny-base")
                    trace_path = output / "actual-training-order.jsonl"
                    if mode == "sampled":
                        self.assertIsNone(metadata["training_order"])
                        self.assertFalse(trace_path.exists())
                        continue
                    trace = [
                        json.loads(line) for line in trace_path.read_text().splitlines()
                    ]
                    self.assertEqual(
                        [key for row in trace for key in row["planned_ids"]], ids
                    )
                    self.assertEqual([row["step"] for row in trace], [1, 2])
                    self.assertEqual(metadata["examples_per_optimizer_step"], 6)
                    self.assertEqual(
                        json.loads(
                            (output / "training-order-completed.json").read_text()
                        )["completed_draws"],
                        12,
                    )
                    if mode == "regrouped":
                        self.assertNotEqual(trace[0]["microbatch_ids"][0], ids[:2])
                for name in states["ordered"]:
                    torch.testing.assert_close(
                        states["ordered"][name],
                        states["regrouped"][name],
                        atol=2e-5,
                        rtol=1e-5,
                    )


if __name__ == "__main__":
    unittest.main()
