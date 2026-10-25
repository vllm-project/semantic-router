"""Portable 2D reranker heads preserve actual weights and reject incomplete state."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from safetensors import SafetensorError
    from safetensors.torch import load_file, save_file
    from transformers import ModernBertConfig, ModernBertModel

    from src.training.model_embeddings.mmbert_32k.representation_contract import (
        set_vela_representation_contract,
    )
    from src.training.model_embeddings.mmbert_32k.reranker_model import (
        Matryoshka2DReranker,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch, transformers and safetensors")
class RerankerHeadStorageTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        backbone = root / "backbone"
        self.checkpoint = root / "checkpoint"
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
        set_vela_representation_contract(config, "reranker")
        ModernBertModel(config).save_pretrained(backbone)
        self.model = Matryoshka2DReranker(
            str(backbone),
            layer_indices=[3, 6, 11, 22],
            dim_indices=[2, 4, 8, 12, 16],
            use_flash_attn=False,
            torch_dtype=torch.float32,
        ).eval()
        # Distinct known values for all 80 tensors, independent of initialization.
        with torch.no_grad():
            for index, tensor in enumerate(self.model.layer_heads.parameters()):
                values = torch.arange(tensor.numel(), dtype=torch.float32)
                tensor.copy_((values.reshape(tensor.shape) + index + 1) / 1024)
        self.model.save_pretrained(str(self.checkpoint))
        self.safe_path = self.checkpoint / "classification_heads.safetensors"
        self.legacy_path = self.checkpoint / "classification_heads.pt"

    def load(self):
        return Matryoshka2DReranker.from_pretrained(
            str(self.checkpoint), use_flash_attn=False, torch_dtype=torch.float32
        )

    def assert_state_equal(self, actual):
        expected = self.model.layer_heads.state_dict()
        self.assertEqual(set(actual), set(expected))
        for name, tensor in actual.items():
            self.assertEqual(tensor.dtype, torch.float32)
            torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)

    def test_all_twenty_heads_and_scores_round_trip_exactly(self):
        self.assertTrue(self.safe_path.is_file())
        self.assertFalse(self.legacy_path.exists())
        stored = load_file(str(self.safe_path))
        self.assertEqual(len(stored), 80)
        self.assert_state_equal(stored)
        loaded = self.load()
        self.assert_state_equal(loaded.layer_heads.state_dict())
        self.assertFalse(loaded.training)
        self.assertEqual(loaded.layer_indices, [3, 6, 11, 22])
        self.assertEqual(loaded.dim_indices, [16, 12, 8, 4, 2])
        self.assertEqual(loaded.pooling_strategy, "cls")
        self.assertEqual(
            loaded.representation_contract, self.model.representation_contract
        )
        config = json.loads((self.checkpoint / "matryoshka_config.json").read_text())
        self.assertEqual(config["hidden_size"], 16)
        self.assertEqual(config["num_layers"], 22)
        ids = torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])
        mask = ids.ne(0).long()
        with torch.inference_mode():
            expected = self.model(ids, mask, return_all_scores=True)["all_scores"]
            actual = loaded(ids, mask, return_all_scores=True)["all_scores"]
        self.assertEqual(len(actual), 20)
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

    def test_safe_format_has_priority_over_different_legacy_weights(self):
        torch.save(
            {
                name: torch.zeros_like(value)
                for name, value in self.model.layer_heads.state_dict().items()
            },
            self.legacy_path,
        )
        with mock.patch("torch.load", side_effect=AssertionError("legacy read")):
            self.assert_state_equal(self.load().layer_heads.state_dict())

    def test_task_auto_map_does_not_replace_the_native_backbone(self):
        path = self.checkpoint / "config.json"
        config = json.loads(path.read_text())
        config["auto_map"] = {"AutoModel": "task_wrapper.TaskModel"}
        path.write_text(json.dumps(config))
        (self.checkpoint / "task_wrapper.py").write_text(
            'raise RuntimeError("task wrapper must not initialize the backbone")\n'
        )
        loaded = self.load()
        self.assertIsInstance(loaded.encoder, ModernBertModel)
        self.assert_state_equal(loaded.layer_heads.state_dict())
        for name, tensor in loaded.encoder.state_dict().items():
            torch.testing.assert_close(
                tensor, self.model.encoder.state_dict()[name], atol=0, rtol=0
            )
        ids = torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])
        mask = ids.ne(0).long()
        with torch.inference_mode():
            expected = self.model(ids, mask, return_all_scores=True)["all_scores"]
            actual = loaded(ids, mask, return_all_scores=True)["all_scores"]
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

    def test_legacy_format_uses_restricted_weights_only_reader(self):
        torch.save(self.model.layer_heads.state_dict(), self.legacy_path)
        self.safe_path.unlink()
        with mock.patch("torch.load", wraps=torch.load) as reader:
            self.assert_state_equal(self.load().layer_heads.state_dict())
        reader.assert_called_once_with(
            str(self.legacy_path), map_location="cpu", weights_only=True
        )

    def test_missing_heads_fail_before_backbone_load(self):
        self.safe_path.unlink()
        with (
            mock.patch.object(Matryoshka2DReranker, "__init__") as constructor,
            self.assertRaisesRegex(FileNotFoundError, "trained reranker heads"),
        ):
            self.load()
        constructor.assert_not_called()

    def test_corrupt_safe_file_never_falls_back_to_valid_legacy(self):
        torch.save(self.model.layer_heads.state_dict(), self.legacy_path)
        self.safe_path.write_bytes(b"invalid safetensors")
        with (
            mock.patch("torch.load", side_effect=AssertionError("legacy read")),
            self.assertRaises(SafetensorError),
        ):
            self.load()

    def test_safe_missing_unexpected_and_wrong_shape_are_rejected(self):
        original = load_file(str(self.safe_path))
        key = next(iter(original))
        invalid_states = [
            {name: tensor for name, tensor in original.items() if name != key},
            {**original, "unknown.weight": torch.zeros(1)},
            {**original, key: original[key].reshape(-1)[:1]},
        ]
        torch.save(self.model.layer_heads.state_dict(), self.legacy_path)
        for state in invalid_states:
            with self.subTest(keys=list(state)):
                save_file(state, str(self.safe_path))
                with self.assertRaises(RuntimeError):
                    self.load()

    def test_legacy_state_is_also_strict(self):
        state = dict(self.model.layer_heads.state_dict())
        del state[next(iter(state))]
        torch.save(state, self.legacy_path)
        self.safe_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.load()

    def test_safe_dtype_is_not_silently_cast_to_fp32(self):
        state = load_file(str(self.safe_path))
        key = next(iter(state))
        state[key] = state[key].half()
        save_file(state, str(self.safe_path))
        with self.assertRaisesRegex(ValueError, "FP32"):
            self.load()

    def test_save_does_not_silently_promote_half_precision_heads(self):
        self.model.layer_heads.half()
        with self.assertRaisesRegex(ValueError, "FP32"):
            self.model.save_pretrained(str(self.checkpoint.parent / "half"))


if __name__ == "__main__":
    unittest.main()
