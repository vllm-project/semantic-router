from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.artifact import read_artifact
from src.training.kv_mapper.collect import ActivationRunMeta, token_fingerprint, write_activation_chunk, write_run_metadata
from src.training.kv_mapper.fit_run import _channel_scores, fit_capture, select_shared_layers
from src.training.kv_mapper.ridge import ols_r2_per_head


class FitRunTests(unittest.TestCase):
    def test_covariance_scores_match_direct_ols(self) -> None:
        rng = np.random.default_rng(8)
        source = rng.normal(size=(3, 40, 2, 4)).astype(np.float32)
        target = rng.normal(size=(2, 40, 2, 4)).astype(np.float32)
        target[0] = source[1] @ np.roll(np.eye(4), 1, axis=0)
        scores = _channel_scores(source, target)
        for i in range(3):
            for j in range(2):
                np.testing.assert_allclose(scores[i, j], ols_r2_per_head(source[i], target[j]), atol=1e-7)

    def test_chunk_to_artifact_with_rotated_source(self) -> None:
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "capture"
            tokens = np.arange(64, dtype=np.int64).reshape(4, 16)
            run.mkdir()
            np.save(run / "tokens.npy", tokens)
            meta = ActivationRunMeta(
                corpus="synthetic", dataset_config="test", dataset_revision="sha",
                source_model="source", source_revision="s", target_model="target", target_revision="t",
                seed=4, seq_len=16, window_stride=0, fitting_token_step=4,
                token_sha256=token_fingerprint(tokens), num_sequences=4,
                source_layers=[0, 1], target_layers=[0], num_kv_heads=2,
                head_dim=3, precision="bf16",
            )
            write_run_metadata(run, meta)
            source = rng.normal(size=(2, 16, 2, 3)).astype(np.float32)
            target = (source[1] @ np.roll(np.eye(3), 1, axis=0) + 0.3).astype(np.float32)
            for index in range(4):
                sl = slice(index * 4, (index + 1) * 4)
                write_activation_chunk(run / "source" / f"{index:06d}.npz", list(source[:, sl]), list(source[:, sl]))
                write_activation_chunk(run / "target" / f"{index:06d}.npz", [target[sl]], [target[sl]])
            artifact = fit_capture(run, Path(directory) / "output", pair_slug="synth", topk=1, ridge_alpha=1e-6)
            manifest, tensors = read_artifact(artifact)
            self.assertEqual(manifest.source_layers_per_target["k"]["0"], [1])
            self.assertEqual(manifest.compatibility.precision, "bf16")
            prediction = source[1].reshape(16, -1) @ tensors["target.0.k.W"] + tensors["target.0.k.b"]
            np.testing.assert_allclose(prediction, target.reshape(16, -1), atol=1e-4)

    def test_shared_selector_rejects_mismatched_rows(self) -> None:
        x = np.zeros((1, 8, 1, 2), dtype=np.float32)
        y = np.zeros((1, 7, 1, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "capture shapes differ"):
            select_shared_layers({"source": (x, x), "target": (y, y)}, 1)


if __name__ == "__main__":
    unittest.main()
