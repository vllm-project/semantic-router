"""CPU tests: shape table, masks, fidelity metrics and the reference op sequences vs Transformers."""

from __future__ import annotations

import unittest

from v2.runtime_kernels.shapes import BACKBONES

try:
    import torch
except (
    ImportError
):  # pragma: no cover - CPU environments without torch skip the tensor tests
    torch = None

try:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35
except (
    Exception
):  # noqa: BLE001 - transformers without Qwen3.5 skips the module comparisons
    qwen35 = None


class ShapeTable(unittest.TestCase):
    def test_hybrid_gemm_shapes(self):
        vega = BACKBONES["vega-27b"]
        g = vega.gemms()
        self.assertEqual(vega.conv_dim, 2 * 16 * 128 + 48 * 128)
        self.assertEqual(g["in_proj_qkv"], (10240, 5120, 48))
        self.assertEqual(g["in_proj_merged"], (10240 + 6144 + 2 * 48, 5120, 48))
        self.assertEqual(g["q_proj"], (24 * 256 * 2, 5120, 16))
        self.assertEqual(g["gate_up_merged"], (2 * 17408, 5120, 64))
        eos = BACKBONES["eos-0.8b"]
        self.assertEqual(eos.gdn_layers, 18)
        self.assertEqual(eos.gemms()["k_proj"], (512, 1024, 6))

    def test_dense_backbone_has_no_gdn(self):
        kai = BACKBONES["kai-0.6b"]
        self.assertEqual(kai.gdn_layers, 0)
        self.assertNotIn("in_proj_qkv", kai.gemms())
        self.assertEqual(kai.gemms()["q_proj"], (16 * 128, 1024, 28))


@unittest.skipIf(torch is None, "torch not installed")
class Masks(unittest.TestCase):
    def test_pack_roundtrip(self):
        from v2.runtime_kernels.masks import pack_mask, unpack_mask

        g = torch.Generator().manual_seed(0)
        for n in (1, 31, 32, 33, 100, 384):
            mask = torch.rand(2, n, n, generator=g) > 0.5
            words = pack_mask(mask)
            self.assertEqual(words.dtype, torch.int32)
            self.assertEqual(tuple(words.shape), (2, n, (n + 31) // 32))
            self.assertTrue(torch.equal(unpack_mask(words, n), mask))

    def test_tree_mask_is_ancestral(self):
        from v2.runtime_kernels.masks import tree_mask

        for n in (64, 384, 1000):
            m = tree_mask(torch, n, "cpu")
            self.assertTrue(bool(m.diagonal().all()))
            self.assertFalse(bool(torch.triu(m, 1).any()), "a row sees a later row")
            self.assertTrue(bool(m[:, 0].all()), "every row sees the shared prefix")
            self.assertLess(
                m.sum().item(), n * (n + 1) / 2, "the tree is sparser than causal"
            )


@unittest.skipIf(torch is None, "torch not installed")
class Fidelity(unittest.TestCase):
    def test_ulp_distance(self):
        from v2.runtime_kernels.fidelity import compare, ulp_distance

        a = torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=torch.bfloat16)
        nxt = (a.view(torch.int16) + 1).view(torch.bfloat16)
        self.assertEqual(ulp_distance(torch, a[:2], nxt[:2]).tolist(), [1, 1])
        self.assertEqual(
            ulp_distance(torch, torch.tensor([0.0]), torch.tensor([-0.0])).tolist(), [0]
        )
        r = compare(torch, a, a.clone())
        self.assertEqual(r["match"], 1.0)
        self.assertEqual(r["max_ulp"], 0)
        r = compare(torch, a, nxt)
        self.assertEqual(r["match"], 0.0)


@unittest.skipIf(
    torch is None or qwen35 is None, "torch / transformers Qwen3.5 not installed"
)
class ReferenceMatchesTransformers(unittest.TestCase):
    """The reference replays must equal the Transformers modules (FP32 CPU, no autocast)."""

    def test_head_rmsnorm(self):
        from v2.runtime_kernels import reference as ref

        norm = qwen35.Qwen3_5RMSNorm(256, eps=1e-6)
        with torch.no_grad():
            norm.weight.normal_(0, 0.1)
        x = torch.randn(3, 5, 4, 256)
        self.assertTrue(
            torch.equal(norm(x), ref.head_rmsnorm(torch, x, norm.weight, 1e-6))
        )

    def test_gated_rmsnorm(self):
        from v2.runtime_kernels import reference as ref

        norm = qwen35.Qwen3_5RMSNormGated(128, eps=1e-6)
        with torch.no_grad():
            norm.weight.normal_(1, 0.1)
        core = torch.randn(40, 128).to(torch.bfloat16)
        z = torch.randn(40, 128).to(torch.bfloat16)
        self.assertTrue(
            torch.equal(
                norm(core, z), ref.gated_rmsnorm(torch, core, z, norm.weight, 1e-6)
            )
        )

    def test_add_rmsnorm(self):
        from v2.runtime_kernels import reference as ref

        norm = qwen35.Qwen3_5RMSNorm(64, eps=1e-6)
        with torch.no_grad():
            norm.weight.normal_(0, 0.1)
        res = torch.randn(2, 7, 64)
        delta = torch.randn(2, 7, 64).to(torch.bfloat16)
        hidden, normed = ref.add_rmsnorm(torch, res, delta, norm.weight, 1e-6)
        self.assertTrue(torch.equal(hidden, res + delta))
        self.assertTrue(torch.equal(normed, norm(res + delta).to(torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
