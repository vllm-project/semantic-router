from __future__ import annotations

import unittest

import torch

from d25.omni.model import tiny
from d25.omni.model.attention import (
    apply_attention_mode,
    enable_noncausal_full_attention,
)


def backbone():
    from transformers import Qwen3_5Model

    torch.manual_seed(0)
    config = tiny.config()
    for part in (config, config.text_config, config.vision_config):
        part._attn_implementation = "sdpa"
    return Qwen3_5Model(config).eval()


class NoncausalMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = backbone()
        vocab = self.model.config.text_config.vocab_size
        generator = torch.Generator().manual_seed(1)
        self.ids = torch.randint(20, vocab, (1, 12), generator=generator)
        self.changed = self.ids.clone()
        self.changed[0, 9] = (self.changed[0, 9] + 7) % vocab

    def hidden(self, ids, mask=None, layer=None):
        mask = torch.ones_like(ids) if mask is None else mask
        with torch.no_grad():
            output = self.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                output_hidden_states=True,
            )
        return (
            output.last_hidden_state if layer is None else output.hidden_states[layer]
        )

    def test_causal_prefix_ignores_future_tokens(self) -> None:
        a, b = self.hidden(self.ids), self.hidden(self.changed)
        self.assertTrue(torch.allclose(a[0, :9], b[0, :9], atol=1e-6))
        self.assertFalse(torch.allclose(a[0, 9:], b[0, 9:], atol=1e-4))

    def test_noncausal_full_attention_sees_future_tokens(self) -> None:
        handle = apply_attention_mode(self.model, "noncausal_full_attention")
        try:
            a, b = self.hidden(self.ids), self.hidden(self.changed)
            self.assertFalse(torch.allclose(a[0, :9], b[0, :9], atol=1e-4))
            linear_a, linear_b = self.hidden(self.ids, layer=3), self.hidden(
                self.changed, layer=3
            )
            self.assertTrue(torch.allclose(linear_a[0, :9], linear_b[0, :9], atol=1e-6))
        finally:
            handle.remove()
        restored_a, restored_b = self.hidden(self.ids), self.hidden(self.changed)
        self.assertTrue(torch.allclose(restored_a[0, :9], restored_b[0, :9], atol=1e-6))

    def test_noncausal_left_padding_is_ignored(self) -> None:
        handle = apply_attention_mode(self.model, "noncausal_full_attention")
        try:
            alone = self.hidden(self.ids[:, 4:])[0, -1]
            padded = torch.cat(
                [torch.zeros(1, 4, dtype=torch.long), self.ids[:, 4:]], dim=1
            )
            batch = torch.cat([padded, self.ids], dim=0)
            mask = torch.ones_like(batch)
            mask[0, :4] = 0
            together = self.hidden(batch, mask)[0, -1]
            self.assertTrue(torch.allclose(alone, together, atol=1e-5))
        finally:
            handle.remove()

    def test_hook_rejects_cache_and_non_sdpa(self) -> None:
        handle = enable_noncausal_full_attention(self.model.language_model)
        try:
            with self.assertRaisesRegex(ValueError, "KV cache"):
                self.model(
                    input_ids=self.ids,
                    attention_mask=torch.ones_like(self.ids),
                    use_cache=True,
                )
        finally:
            handle.remove()
        self.model.language_model.config._attn_implementation = "eager"
        with self.assertRaisesRegex(ValueError, "SDPA"):
            enable_noncausal_full_attention(self.model.language_model)


if __name__ == "__main__":
    unittest.main()
