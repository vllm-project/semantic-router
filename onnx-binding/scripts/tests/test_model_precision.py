"""Match native HF reduced-precision loading without rounding RoPE buffers."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_precision import cast_parameters_preserving_buffers
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch and transformers")
class ModelPrecisionTest(unittest.TestCase):
    def test_modernbert_clone_matches_native_load_parameters_buffers_and_outputs(self):
        torch.manual_seed(19)
        config = ModernBertConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32768,
            local_attention=4,
            attention_dropout=0.0,
            pad_token_id=0,
            reference_compile=False,
        )
        config._attn_implementation = "sdpa"
        original = ModernBertModel(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    native = ModernBertModel.from_pretrained(
                        directory,
                        torch_dtype=dtype,
                        attn_implementation="sdpa",
                        reference_compile=False,
                    ).eval()
                    candidate = cast_parameters_preserving_buffers(
                        copy.deepcopy(original), dtype
                    )
                    for name, parameter in candidate.named_parameters():
                        expected = native.get_parameter(name)
                        self.assertEqual(parameter.dtype, expected.dtype)
                        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
                    for name, buffer in candidate.named_buffers():
                        expected = native.get_buffer(name)
                        self.assertEqual(buffer.dtype, expected.dtype)
                        torch.testing.assert_close(buffer, expected, atol=0, rtol=0)
                    # The previous whole-module cast loses frequency precision;
                    # multiplying the same error by a long position amplifies it.
                    rounded = copy.deepcopy(original).to(dtype)
                    name = next(iter(dict(native.named_buffers())))
                    error = (
                        rounded.get_buffer(name).float() - native.get_buffer(name)
                    ).abs()
                    self.assertGreater(float(error.max() * 32767), 0.0)
                    for length in (2, 65, 513):
                        ids = torch.full((2, length), 3, dtype=torch.long)
                        mask = torch.ones_like(ids)
                        mask[1, -1] = 0
                        ids[1, -1] = 0
                        with torch.inference_mode():
                            expected = native(
                                ids, attention_mask=mask
                            ).last_hidden_state
                            actual = candidate(
                                ids, attention_mask=mask
                            ).last_hidden_state
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_preserves_original_buffer_types_and_tied_parameter_objects(self):
        module = torch.nn.Module()
        module.left = torch.nn.Linear(4, 4)
        module.right = torch.nn.Linear(4, 4)
        module.right.weight = module.left.weight
        module.register_buffer(
            "half_buffer", torch.tensor([0.125], dtype=torch.float16)
        )
        module.register_buffer(
            "double_buffer", torch.tensor([0.1], dtype=torch.float64)
        )
        module.register_buffer("counter", torch.tensor([7], dtype=torch.int64))
        before = {name: buffer.clone() for name, buffer in module.named_buffers()}
        cast_parameters_preserving_buffers(module, torch.bfloat16)
        self.assertIs(module.left.weight, module.right.weight)
        self.assertEqual(module.left.weight.dtype, torch.bfloat16)
        for name, buffer in module.named_buffers():
            self.assertEqual(buffer.dtype, before[name].dtype)
            torch.testing.assert_close(buffer, before[name], atol=0, rtol=0)

    def test_rejects_live_gradients_before_changing_parameters(self):
        module = torch.nn.Linear(3, 1)
        module(torch.ones(1, 3)).sum().backward()
        with self.assertRaisesRegex(ValueError, "independent inference copy"):
            cast_parameters_preserving_buffers(module, torch.float16)
        self.assertEqual(module.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
