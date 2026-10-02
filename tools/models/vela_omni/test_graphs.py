"""Tensor-only tests; run in the pinned export environment (no model downloads)."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from attention import install_export_attention, query_block_attention
from export import graph_nodes, validate_portable_graph
from graphs import FixedClapResize, causal_padding_mask
from onnx import helper
from reference_backend import cpu_flash_reference
from torch.nn import functional
from transformers import Qwen3Config, Qwen3Model
from transformers.masking_utils import create_causal_mask


class PortableGraphTests(unittest.TestCase):
    def test_control_flow_bodies_cannot_hide_custom_operators(self):

        inner = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])], "body", [], []
        )
        outer = helper.make_graph(
            [helper.make_node("Loop", [], [], body=inner)], "outer", [], []
        )
        self.assertEqual(
            [node.op_type for node in graph_nodes(outer)], ["Loop", "Identity"]
        )
        validate_portable_graph(outer, "text")
        outer.node[0].attribute[0].g.node[0].domain = "unreviewed.custom"
        with self.assertRaisesRegex(ValueError, "nonportable"):
            validate_portable_graph(outer, "text")


class BlockAttentionGraph(torch.nn.Module):
    def forward(self, query, key, value, mask):
        return query_block_attention(query, key, value, mask, 8**-0.5)


class BlockAttentionTests(unittest.TestCase):
    def test_export_preserves_dynamic_loop_full_keys_and_masked_rows(self):
        generator = torch.Generator().manual_seed(987)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.onnx"
            example = (
                *(torch.randn(1, 4, 17, 8) for _ in range(3)),
                torch.ones(1, 17, dtype=torch.int64),
            )
            names = ["query", "key", "value", "mask"]
            torch.onnx.export(
                BlockAttentionGraph(),
                example,
                str(path),
                dynamo=False,
                input_names=names,
                output_names=["output"],
                opset_version=17,
                dynamic_axes={name: {2: "sequence"} for name in names[:3]}
                | {"mask": {1: "sequence"}, "output": {2: "sequence"}},
            )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
            for length in (1, 17, 255, 256, 257, 513):
                tensors = tuple(
                    torch.randn(1, 4, length, 8, generator=generator) for _ in range(3)
                )
                for side in ("left", "right"):
                    mask = torch.ones(1, length, dtype=torch.int64)
                    if side == "left":
                        mask[:, : length // 4] = 0
                    elif length > 1:
                        mask[:, -max(1, length // 4) :] = 0
                    expected = functional.scaled_dot_product_attention(
                        *tensors, attn_mask=causal_padding_mask(mask)
                    )
                    actual = session.run(
                        None,
                        {
                            name: value.numpy()
                            for name, value in zip(names, (*tensors, mask), strict=True)
                        },
                    )[0]
                    np.testing.assert_allclose(
                        actual, expected.numpy(), atol=1e-6, rtol=1e-5
                    )

    def test_original_qwen_layers_match_with_bounded_attention(self):

        torch.manual_seed(517)
        config = Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=128,
            attention_dropout=0,
        )
        config._attn_implementation = "sdpa"
        original = Qwen3Model(config).eval()
        bounded = copy.deepcopy(original)
        install_export_attention(bounded)
        ids = torch.randint(0, 128, (1, 513))
        for start, end in ((0, 513), (12, 513), (0, 502)):
            mask = torch.zeros_like(ids)
            mask[:, start:end] = 1
            with torch.inference_mode():
                expected = original(
                    ids, attention_mask=mask, use_cache=False
                ).last_hidden_state
                with cpu_flash_reference():
                    flash = original(
                        ids, attention_mask=mask, use_cache=False
                    ).last_hidden_state
                actual = bounded(
                    ids, attention_mask={"full_attention": mask}, use_cache=False
                ).last_hidden_state
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(flash, expected, atol=2e-6, rtol=2e-5)


class CausalMaskTests(unittest.TestCase):
    def test_export_mask_matches_transformers_with_left_and_right_padding(self):

        config = Qwen3Config()
        config._attn_implementation = "sdpa"
        for mask in (
            torch.tensor([[1, 1, 1, 1]]),
            torch.tensor([[0, 0, 1, 1]]),
            torch.tensor([[1, 1, 0, 0]]),
        ):
            expected = create_causal_mask(
                config,
                input_embeds=torch.zeros(1, 4, 8),
                attention_mask=mask,
                cache_position=torch.arange(4),
                past_key_values=None,
            )
            if expected is None:
                expected = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
            self.assertTrue(torch.equal(causal_padding_mask(mask), expected))


class CubicRewriteTests(unittest.TestCase):
    def test_constant_operator_matches_bicubic_on_structured_and_random_inputs(self):
        generator = torch.Generator().manual_seed(713)
        rewrite = FixedClapResize()
        fixtures = [
            torch.randn(1, 1, 1001, 64, generator=generator),
            torch.arange(1001, dtype=torch.float32)
            .reshape(1, 1, 1001, 1)
            .expand(1, 1, 1001, 64)
            / 1001,
            torch.zeros(1, 1, 1001, 64),
        ]
        fixtures[2][:, :, -1] = 1
        for values in fixtures:
            expected = functional.interpolate(
                values, size=(1024, 64), mode="bicubic", align_corners=True
            )
            expected = (
                expected.reshape(1, 4, 256, 64)
                .permute(0, 1, 3, 2)
                .reshape(1, 1, 256, 256)
            )
            torch.testing.assert_close(rewrite(values), expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
