from __future__ import annotations

import os
import unittest
from unittest import mock

from support import has

HAVE_VLLM = has("vllm") and has("torch")


@unittest.skipUnless(HAVE_VLLM, "needs vLLM")
class GdnBf16InputsTest(unittest.TestCase):
    def setUp(self) -> None:
        from vllm.model_executor.custom_op import op_registry_oot

        self.registry = op_registry_oot
        self.addCleanup(self.registry.pop, "ChunkGatedDeltaRule", None)

    def test_register_is_opt_in_and_idempotent(self) -> None:
        import vllm_sr_plugins
        from vllm_sr_plugins.decision2 import fp32

        with mock.patch.dict(os.environ, {fp32.ENV: ""}):
            vllm_sr_plugins.register()
        self.assertNotIn("ChunkGatedDeltaRule", self.registry)
        with mock.patch.dict(os.environ, {fp32.ENV: "1"}):
            vllm_sr_plugins.register()
            vllm_sr_plugins.register()
        self.assertEqual(
            self.registry["ChunkGatedDeltaRule"].__name__,
            "Bf16InputChunkGatedDeltaRule",
        )

    def test_float32_inputs_reach_the_kernel_as_bfloat16(self) -> None:
        import torch
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            ChunkGatedDeltaRule,
        )

        from vllm_sr_plugins.decision2.fp32 import register_gdn_bf16_inputs

        seen = []

        def kernel(self, q, k, v, g, beta, initial_state, *rest):
            seen.append((q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype, rest[-1]))
            return q * 2, initial_state + 1

        override = register_gdn_bf16_inputs()
        op = override.__new__(override)
        q = torch.randn(1, 5, 2, 4)
        state = torch.zeros(3, 2, 4, 4)
        g = torch.randn(1, 5, 2)
        with mock.patch.object(ChunkGatedDeltaRule, "forward_native", kernel):
            buffer = torch.zeros(q.numel() + 3)
            o, final = op.forward_native(
                q, q, q, g, g, state, True, None, None, None, False, buffer
            )
            self.assertEqual(
                seen[-1],
                (
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.float32,
                    torch.bfloat16,
                    None,
                ),
            )
            self.assertEqual(o.dtype, torch.float32)
            self.assertTrue(torch.equal(o, (q.bfloat16() * 2).float()))
            self.assertTrue(torch.equal(buffer[: q.numel()], o.reshape(-1)))
            self.assertEqual(final.dtype, torch.float32)
            op.forward_native(
                q.bfloat16(), q.bfloat16(), q.bfloat16(), g, g, state, True
            )
            self.assertEqual(seen[-1][0], torch.bfloat16)
            self.assertEqual(seen[-1][4], torch.float32)


if __name__ == "__main__":
    unittest.main()
