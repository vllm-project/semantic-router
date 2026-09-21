from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.artifact import CompatibilitySpec, read_artifact
from src.training.kv_mapper.fit import (
    fit_full_head,
    select_source_layers,
    write_fitted_artifact,
)
from src.training.kv_mapper.ridge import RidgeAccumulator


class RidgeFitTests(unittest.TestCase):
    def test_centered_ridge_recovers_linear_map(self) -> None:
        rng = np.random.RandomState(0)
        n, dx, dy = 4000, 12, 5
        weight = rng.randn(dx, dy)
        bias = rng.randn(dy)
        x = rng.randn(n, dx)
        y = x @ weight + bias
        acc = RidgeAccumulator(dx, dy)
        acc.add(x, y)
        w_hat, b_hat = acc.solve_affine(1e-8)
        self.assertLess(
            np.linalg.norm(w_hat - weight) / np.linalg.norm(weight), 1e-4
        )
        self.assertLess(np.linalg.norm(b_hat - bias), 1e-3)

    def test_empty_ridge_raises(self) -> None:
        acc = RidgeAccumulator(4, 2)
        with self.assertRaises(RuntimeError):
            acc.solve_affine(0.01)

    def test_pearson_selects_linear_source_layer(self) -> None:
        rng = np.random.RandomState(1)
        seq, n_kv, d = 64, 2, 4
        src = [rng.randn(seq, n_kv, d).astype(np.float32) for _ in range(3)]
        tgt = [src[1] * 2.0, src[2] * 3.0]
        top = select_source_layers(src, tgt, k=1)
        self.assertEqual(top[0][0], 1)
        self.assertEqual(top[1][0], 2)

    def test_fit_writes_a1_artifact(self) -> None:
        rng = np.random.RandomState(2)
        n_src, n_tgt, n_kv, d, seq, k = 4, 3, 2, 4, 80, 2
        src = [rng.randn(seq, n_kv, d).astype(np.float32) for _ in range(n_src)]
        topk = [[0, 1], [1, 2], [2, 3]]
        tgt = []
        for t in range(n_tgt):
            x = np.concatenate([src[i].reshape(seq, -1) for i in topk[t]], axis=-1)
            w = rng.randn(x.shape[-1], n_kv * d).astype(np.float32)
            b = rng.randn(n_kv * d).astype(np.float32)
            tgt.append((x @ w + b).reshape(seq, n_kv, d))
        tensors_k = fit_full_head(
            [(src, tgt)],
            topk,
            n_kv_heads=n_kv,
            head_dim=d,
            ridge_alpha=1e-6,
        )
        self.assertIn("target.0.k.W", tensors_k)
        x0 = np.concatenate([src[i].reshape(seq, -1) for i in topk[0]], axis=-1)
        y_hat = x0 @ tensors_k["target.0.k.W"] + tensors_k["target.0.k.b"]
        y_true = tgt[0].reshape(seq, -1)
        rel = np.linalg.norm(y_hat - y_true) / np.linalg.norm(y_true)
        self.assertLess(rel, 0.05)

        v_src = [rng.randn(seq, n_kv, d).astype(np.float32) for _ in range(n_src)]
        v_tgt = [rng.randn(seq, n_kv, d).astype(np.float32) for _ in range(n_tgt)]
        tensors_v = fit_full_head(
            [(v_src, v_tgt)],
            topk,
            n_kv_heads=n_kv,
            head_dim=d,
            ridge_alpha=0.01,
            channel="v",
        )
        tensors = {**tensors_k, **tensors_v}
        compat = CompatibilitySpec(
            source_model="Qwen/Qwen3-14B",
            source_revision="abc123",
            target_model="Qwen/Qwen3-32B",
            target_revision="def456",
            variant="full_head",
            precision="fp16",
            source_tp=1,
            target_tp=1,
            head_order="contiguous",
            num_kv_heads=n_kv,
            head_dim=d,
        )
        layers = {str(t): topk[t] for t in range(n_tgt)}
        with tempfile.TemporaryDirectory() as directory:
            mapper_id = write_fitted_artifact(
                Path(directory),
                compat=compat,
                tensors=tensors,
                topk=k,
                ridge_alpha=1e-6,
                source_layers_per_target={"k": layers, "v": layers},
                pair_slug="synth-14b-32b",
                calibration={"corpus": "synthetic", "seed": 2},
            )
            manifest, got = read_artifact(Path(directory) / mapper_id)
        self.assertEqual(manifest.topk, k)
        self.assertEqual(manifest.ridge_alpha, 1e-6)
        self.assertTrue(manifest.centered_inputs)
        self.assertEqual(manifest.source_layers_per_target["k"]["0"], [0, 1])
        np.testing.assert_allclose(got["target.0.k.W"], tensors["target.0.k.W"])


if __name__ == "__main__":
    unittest.main()
