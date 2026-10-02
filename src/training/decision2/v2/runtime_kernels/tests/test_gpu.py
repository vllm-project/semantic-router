"""GPU tests (ROCm/CUDA + Triton): every fused kernel against its reference op sequence.

The element-wise kernels must be bit-identical to the reference at the released shapes;
tree attention must stay within the tolerance of a FlashAttention-style kernel. Skipped
without a GPU. On a node: ``python3 -m v2.runtime_kernels.selftest --out RUN``.
"""

from __future__ import annotations

import unittest

try:
    import torch

    HAVE_GPU = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    HAVE_GPU = False


@unittest.skipUnless(HAVE_GPU, "needs a GPU with Triton")
class ElementwiseBitExact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from v2.runtime_kernels import reference, triton_elementwise
        from v2.runtime_kernels.fidelity import bits

        cls.ref, cls.tk, cls.bits = reference, triton_elementwise, bits
        cls.dev = torch.device("cuda")

    def assertBitEqual(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertTrue(torch.equal(self.bits(torch, a), self.bits(torch, b)))

    def test_add_rmsnorm(self):
        g = torch.Generator(device=self.dev).manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for H in (1024, 2048, 2560, 4096, 5120):
                res = torch.randn(1, 300, H, generator=g, device=self.dev) * 3
                delta = (torch.randn(1, 300, H, generator=g, device=self.dev) * 0.5).to(
                    torch.bfloat16
                )
                w = torch.randn(H, generator=g, device=self.dev) * 0.1
                h_ref, n_ref = self.ref.add_rmsnorm(torch, res, delta, w, 1e-6)
                h, n = self.tk.add_rmsnorm(res, delta, 1.0 + w.float(), 1e-6)
                self.assertBitEqual(h_ref, h)
                self.assertBitEqual(n_ref, n)
                _, n_ref = self.ref.add_rmsnorm(torch, res, None, w, 1e-6)
                self.assertBitEqual(
                    n_ref, self.tk.add_rmsnorm(res, None, 1.0 + w.float(), 1e-6)[1]
                )

    def test_silu_mul_exhaustive(self):
        every = (
            torch.arange(-32768, 32768, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        x = every[torch.isfinite(every)].reshape(1, -1).to(self.dev)
        up = torch.randn_like(x.float()).to(torch.bfloat16)
        self.assertBitEqual(self.ref.silu_mul(torch, x, up), self.tk.silu_mul(x, up))

    def test_sigmoid_gate(self):
        attn = torch.randn(1, 200, 24, 256, device=self.dev).to(torch.bfloat16)
        qp = (torch.randn(1, 200, 24 * 512, device=self.dev) * 3).to(torch.bfloat16)
        gate = qp.view(1, 200, 24, 512)[..., 256:]
        self.assertBitEqual(
            self.ref.sigmoid_gate(torch, attn, gate.reshape(1, 200, -1)),
            self.tk.sigmoid_gate(attn, gate),
        )

    def test_gated_rmsnorm(self):
        core = torch.randn(48 * 300, 128, device=self.dev).to(torch.bfloat16)
        z = (torch.randn(48 * 300, 128, device=self.dev) * 2).to(torch.bfloat16)
        w = 1.0 + torch.randn(128, device=self.dev) * 0.1
        self.assertBitEqual(
            self.ref.gated_rmsnorm(torch, core, z, w, 1e-6),
            self.tk.gated_rmsnorm(core, z, w, 1e-6),
        )

    def test_attn_prep(self):
        T, nh, nkv, hd = 300, 24, 4, 256
        qp = (torch.randn(1, T, nh * hd * 2, device=self.dev) * 2).to(torch.bfloat16)
        kp = (torch.randn(1, T, nkv * hd, device=self.dev) * 2).to(torch.bfloat16)
        vp = torch.randn(1, T, nkv * hd, device=self.dev).to(torch.bfloat16)
        qw, kw = (
            torch.randn(hd, device=self.dev) * 0.1,
            torch.randn(hd, device=self.dev) * 0.1,
        )
        inv_freq = 1.0 / (
            10000000
            ** (torch.arange(0, 64, 2, dtype=torch.float, device=self.dev) / 64)
        )
        cos, sin = self.ref.rotary_cos_sin(
            torch, inv_freq, torch.arange(T, device=self.dev)[None], torch.float32
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q_ref, k_ref, _, _ = self.ref.attn_prep(
                torch, qp, kp, vp, qw, kw, cos, sin, hd, 1e-6
            )
            q, k = self.tk.attn_prep(
                qp, kp, 1.0 + qw.float(), 1.0 + kw.float(), cos, sin, nh, nkv, hd, 1e-6
            )
        self.assertBitEqual(q_ref, q)
        self.assertBitEqual(k_ref, k)

    def test_gdn_prep(self):
        T, nk, nv, dk = 300, 16, 48, 128
        C = (2 * nk + nv) * dk
        mixed = (torch.randn(1, T, C, device=self.dev) * 2).to(torch.bfloat16)
        conv_w = torch.randn(C, 4, device=self.dev) * 0.3
        b = torch.randn(1, T, nv, device=self.dev).to(torch.bfloat16)
        a = (torch.randn(1, T, nv, device=self.dev) * 3).to(torch.bfloat16)
        A_log = torch.log(torch.rand(nv, device=self.dev) * 15 + 0.5)
        dt_bias = torch.randn(nv, device=self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            want = self.ref.gdn_prep(
                torch, mixed, b, a, conv_w, A_log, dt_bias, nk, dk, dk, expand_qk=False
            )
            got = self.tk.gdn_prep(mixed, b, a, conv_w, A_log, dt_bias, nk, dk)
        for w_, g_ in zip(want, got):
            self.assertBitEqual(w_, g_)

    def test_fla_grouped_value_path(self):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        T, nk, nv = 300, 16, 48
        q = torch.randn(1, T, nk, 128, device=self.dev).to(torch.bfloat16)
        k = torch.randn(1, T, nk, 128, device=self.dev).to(torch.bfloat16)
        v = torch.randn(1, T, nv, 128, device=self.dev).to(torch.bfloat16)
        g = -torch.rand(1, T, nv, device=self.dev)
        beta = torch.rand(1, T, nv, device=self.dev).to(torch.bfloat16)
        rep = chunk_gated_delta_rule(
            q.repeat_interleave(3, 2),
            k.repeat_interleave(3, 2),
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
        )[0]
        gva = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]
        self.assertBitEqual(rep, gva)


@unittest.skipUnless(HAVE_GPU, "needs a GPU with Triton")
class TreeAttention(unittest.TestCase):
    def test_against_math_sdpa(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        from v2.runtime_kernels.masks import pack_mask, tree_mask
        from v2.runtime_kernels.tree_attention import tree_attention

        dev = torch.device("cuda")
        N, H, HKV, D = 300, 24, 4, 256
        q = torch.randn(1, H, N, D, device=dev).to(torch.bfloat16)
        k = torch.randn(1, HKV, N, D, device=dev).to(torch.bfloat16)
        v = torch.randn(1, HKV, N, D, device=dev).to(torch.bfloat16)
        mask = tree_mask(torch, N, dev)
        with sdpa_kernel([SDPBackend.MATH]):
            want = torch.nn.functional.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=mask[None, None],
                enable_gqa=True,
            )
        want = want.transpose(1, 2).reshape(1, N, H * D)
        got = tree_attention(
            q, k, v, pack_mask(mask[None]), None, D**-0.5, pv_fp32=True
        ).float()
        rel = (got - want).norm() / want.norm()
        self.assertLess(rel.item(), 5e-3)


if __name__ == "__main__":
    unittest.main()
