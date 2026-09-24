from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


class ModelEvalRunTests(unittest.TestCase):
    def _model(self):
        try:
            import torch
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest("torch or transformers not installed") from exc
        config = Qwen3Config(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
        )
        return torch, Qwen3ForCausalLM(config).eval()

    def test_cached_continuation_matches_full_forward(self) -> None:
        torch, model = self._model()
        from src.training.kv_mapper.model_eval_run import (
            _cache_pairs,
            _continuation_score,
        )

        with torch.inference_mode():
            prefix = model(torch.tensor([[3, 4, 5, 6]]), use_cache=True)
            cached = _continuation_score(
                model, _cache_pairs(prefix.past_key_values), 7, [8, 9]
            )
            full = model(torch.tensor([[3, 4, 5, 6, 7, 8]]), use_cache=False)
            log_probs = full.logits[0, -2:].float().log_softmax(dim=-1)
            expected = float((log_probs[0, 8] + log_probs[1, 9]) / 2)
        self.assertAlmostEqual(cached, expected, places=5)

    def test_identity_mapping_matches_target_cache(self) -> None:
        torch, model = self._model()
        from src.training.kv_mapper.hooks import attach_pre_rope_hooks, remove_hooks
        from src.training.kv_mapper.model_eval_run import _cache_pairs, _mapped_pairs

        slots, handles = attach_pre_rope_hooks(model, 2, 16)
        try:
            with torch.inference_mode():
                output = model(torch.tensor([[3, 4, 5, 6]]), use_cache=True)
        finally:
            remove_hooks(handles)
        weights = {}
        for layer in range(2):
            for channel in ("k", "v"):
                weights[f"target.{layer}.{channel}.W"] = torch.eye(32)
                weights[f"target.{layer}.{channel}.b"] = torch.zeros(32)
        manifest = SimpleNamespace(
            source_layers_per_target={"k": {"0": [0], "1": [1]}},
            compatibility=SimpleNamespace(num_kv_heads=2, head_dim=16),
        )
        with torch.inference_mode():
            mapped = _mapped_pairs(slots, model, manifest, weights, torch.float32)
        for (mapped_k, mapped_v, _), (true_k, true_v) in zip(
            mapped, _cache_pairs(output.past_key_values)
        ):
            self.assertTrue(torch.equal(mapped_k, true_k))
            self.assertTrue(torch.equal(mapped_v, true_v))


if __name__ == "__main__":
    unittest.main()
