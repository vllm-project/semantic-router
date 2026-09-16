"""Real tiny 22-layer forward/backward and complete task-artifact round trips."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    from src.training.model_embeddings.mmbert_32k import newbase_model
    from src.training.model_embeddings.mmbert_32k.newbase_data import file_digest
    from src.training.model_embeddings.mmbert_32k.newbase_model import (
        ExitSpec,
        NewBaseTask,
        state_digest,
    )
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
except ImportError:
    SentenceTransformer = None


@unittest.skipIf(torch is None, "requires torch and transformers")
class NewBaseModelTest(unittest.TestCase):
    def test_explicit_exit_matrix_preserves_old_serialization_and_full_weight(self):
        original = ExitSpec()
        self.assertNotIn("exit_weights", original.to_dict())
        self.assertEqual(ExitSpec.from_dict(original.to_dict()), original)
        weights = [[0.5 / 19] * 5 for _ in range(4)]
        weights[-1][0] = 0.5
        exits = ExitSpec(exit_weights=tuple(tuple(row) for row in weights))
        exits.validate(22, 768)
        self.assertEqual(dict(exits.weighted())[22, 768], 0.5)
        self.assertEqual(ExitSpec.from_dict(exits.to_dict()), exits)
        with self.assertRaisesRegex(ValueError, "complete exit matrix"):
            ExitSpec(exit_weights=((1.0,),)).validate(22, 768)

    def test_continued_task_keeps_all_tensors_outputs_and_explicit_lineage(self):
        for task in ("embedding", "reranker"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                model = self.model(task).eval()
                checkpoint = Path(directory) / "checkpoint"
                model.save(checkpoint)
                (checkpoint / "tokenizer.json").write_text("{}")
                files = {
                    str(p.relative_to(checkpoint)): file_digest(p)
                    for p in checkpoint.rglob("*")
                    if p.is_file()
                }
                before = torch.get_rng_state().clone()
                restored = NewBaseTask.from_task(
                    checkpoint,
                    task,
                    model.exits,
                    expected_files=files,
                    provenance={
                        "parent_task_revision": "a" * 40,
                        "base_revision": "b" * 40,
                    },
                ).eval()
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(
                    state_digest(model.state_dict()),
                    state_digest(restored.state_dict()),
                )
                self.assertEqual(restored.lineage["initialization"], "continued_task")
                with torch.no_grad():
                    left, right = model(*self.inputs()), restored(*self.inputs())
                for key in left:
                    torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, "task checkpoint"):
                    NewBaseTask.from_base(
                        checkpoint, task, model.exits, expected_files=files
                    )

    def test_compile_option_is_changed_only_when_the_config_defines_it(self):
        for existing in (False, True):
            config = SimpleNamespace(model_type="modernbert")
            if existing:
                config.reference_compile = True
            info = {
                "unexpected_keys": [],
                "missing_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
            with (
                self.subTest(existing=existing),
                patch.object(
                    newbase_model.AutoConfig, "from_pretrained", return_value=config
                ),
                patch.object(
                    newbase_model.AutoModel,
                    "from_pretrained",
                    return_value=(object(), info),
                ),
            ):
                newbase_model._load_encoder(Path("unused-config-fixture"))
                self.assertEqual(hasattr(config, "reference_compile"), existing)
                if existing:
                    self.assertIs(config.reference_compile, False)

    def model(self, task):
        config = ModernBertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=22,
            num_attention_heads=2,
            max_position_embeddings=128,
            local_attention=4,
            reference_compile=False,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "sdpa"
        return NewBaseTask(
            ModernBertModel(config),
            task,
            ExitSpec(dimensions=(16, 12, 8, 4, 2)),
            {"test_only": True},
        )

    def inputs(self):
        ids = torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])
        return ids, ids.ne(0).long()

    def test_every_encoder_layer_and_twenty_heads_receive_gradients(self):
        model = self.model("reranker").train()
        outputs = model(*self.inputs())
        self.assertEqual(len(outputs), 20)
        loss = sum(
            weight * torch.nn.functional.softplus(-outputs[key]).mean()
            for key, weight in model.exits.weighted()
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for layer in model.encoder.layers:
            self.assertTrue(
                any(parameter.grad.abs().max() > 0 for parameter in layer.parameters())
            )
        for heads in model.layer_heads.values():
            for head in heads.values():
                self.assertTrue(head[-1].weight.grad.abs().max() > 0)

    def test_both_tasks_save_reload_bitwise_and_preserve_rng(self):
        for task in ("embedding", "reranker"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                model = self.model(task).eval()
                with torch.no_grad():
                    expected = model(*self.inputs())
                checkpoint = Path(directory) / "checkpoint"
                model.save(checkpoint)
                before = torch.get_rng_state().clone()
                restored = NewBaseTask.resume(checkpoint).eval()
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(
                    state_digest(model.state_dict()),
                    state_digest(restored.state_dict()),
                )
                with torch.no_grad():
                    actual = restored(*self.inputs())
                for key, value in expected.items():
                    torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
                (checkpoint / "config.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    NewBaseTask.resume(checkpoint)

    def test_embedding_truncates_then_normalizes_and_excludes_padding(self):
        model = self.model("embedding").eval()
        ids, mask = self.inputs()
        with torch.no_grad():
            batched = model(ids, mask)
            single = model(ids[1:2, :3], mask[1:2, :3])
        for key, values in batched.items():
            torch.testing.assert_close(
                values.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6
            )
            torch.testing.assert_close(values[1:2], single[key], atol=1e-5, rtol=1e-5)

    @unittest.skipIf(SentenceTransformer is None, "requires sentence-transformers")
    def test_standard_embedding_encode_matches_native_and_preserves_weights(self):
        vocabulary = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[UNK]": 3,
            "quiet": 4,
            "observatory": 5,
            "clear": 6,
            "sky": 7,
        }
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            unk_token="[UNK]",
            model_max_length=128,
            model_input_names=["input_ids", "attention_mask"],
        )
        sentences = ["quiet observatory", "clear"]
        batch = tokenizer(sentences, padding=True, return_tensors="pt")
        model = self.model("embedding").eval()
        initial_state = state_digest(model.state_dict())
        with torch.no_grad():
            expected = model(batch["input_ids"], batch["attention_mask"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            model.save(checkpoint, tokenizer)
            # Hub Git snapshots preserve files, not empty module directories.
            (checkpoint / "2_Normalize").rmdir()
            saved = SentenceTransformer(
                str(checkpoint), device="cpu", local_files_only=True
            )
            # No normalize_embeddings argument: the saved module must suffice.
            actual = saved.encode(sentences, convert_to_tensor=True)
            torch.testing.assert_close(actual, expected[22, 16], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(
                actual.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6
            )
            for dimension in model.exits.dimensions:
                prefix = saved.encode(
                    sentences,
                    convert_to_tensor=True,
                    truncate_dim=dimension,
                    normalize_embeddings=True,
                )
                torch.testing.assert_close(
                    prefix, expected[22, dimension], atol=1e-6, rtol=1e-6
                )
            restored = NewBaseTask.resume(checkpoint).eval()
            self.assertEqual(state_digest(restored.state_dict()), initial_state)
            self.assertEqual(state_digest(model.state_dict()), initial_state)
            self.assertEqual(
                state_digest(saved[0].auto_model.state_dict()),
                state_digest(model.encoder.state_dict()),
            )

    def test_missing_encoder_identity_rejected(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "Base lock"),
        ):
            NewBaseTask.from_base(
                Path(directory),
                "embedding",
                ExitSpec(),
                expected_files={},
                provenance={},
            )

    def test_exact_base_values_survive_fresh_task_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            encoder = self.model("embedding").encoder
            del encoder.config.representation_contract
            encoder.save_pretrained(base)
            (base / "tokenizer.json").write_text("{}")
            files = {
                name: file_digest(base / name)
                for name in ("model.safetensors", "config.json", "tokenizer.json")
            }
            task = NewBaseTask.from_base(
                base,
                "reranker",
                ExitSpec(dimensions=(16, 12, 8, 4, 2)),
                expected_files=files,
                provenance={"source_revision": "a" * 40},
            )
            self.assertEqual(
                state_digest(encoder.state_dict()),
                state_digest(task.encoder.state_dict()),
            )
            self.assertEqual(len(task.layer_heads), 4)
            task.save(Path(directory) / "task")
            task_files = {
                name: file_digest(Path(directory) / "task" / name)
                for name in ("model.safetensors", "config.json")
            }
            (Path(directory) / "task" / "tokenizer.json").write_text("{}")
            task_files["tokenizer.json"] = file_digest(
                Path(directory) / "task" / "tokenizer.json"
            )
            with self.assertRaisesRegex(ValueError, "task checkpoint"):
                NewBaseTask.from_base(
                    Path(directory) / "task",
                    "reranker",
                    ExitSpec(dimensions=(16, 12, 8, 4, 2)),
                    expected_files=task_files,
                    provenance={"source_revision": "a" * 40},
                )


if __name__ == "__main__":
    unittest.main()
