"""MoE backbone support: BOS prompt dispatch, exact LoRA targets, frozen experts."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = nn = None

try:
    from transformers import Gemma4TextConfig, Qwen3_5MoeTextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextModel,
    )
except ImportError:
    Gemma4TextConfig = Qwen3_5MoeTextConfig = None


class Tokenizer:
    bos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(ch) % 97 + 3 for ch in text]


ROW = {
    "id": "r1",
    "state": {"a": 1},
    "task_type": "choice",
    "instructions": "Pick one.",
    "options": [
        {"key": "x", "description": "first"},
        {"key": "y", "description": "second"},
    ],
    "label": 1,
    "family": "f",
}


@unittest.skipIf(torch is None, "torch is unavailable")
class PromptDispatch(unittest.TestCase):
    def test_bos_prompt_shifts_every_position_by_one(self):
        from training.model.decision_model import (
            BOS_PROMPT_VERSION,
            PROMPT_VERSION,
            encode,
            encode_bos,
            encoder_for,
        )

        plain = encode(ROW, Tokenizer(), 4096)
        bos = encode_bos(ROW, Tokenizer(), 4096)
        self.assertEqual(bos["ids"], [2, *plain["ids"]])
        self.assertEqual(
            bos["candidate_positions"], [p + 1 for p in plain["candidate_positions"]]
        )
        self.assertEqual(bos["query_position"], plain["query_position"] + 1)
        self.assertEqual(bos["prompt_sha256"], plain["prompt_sha256"])
        self.assertNotEqual(bos["token_ids_sha256"], plain["token_ids_sha256"])
        self.assertIs(encoder_for({"prompt_version": PROMPT_VERSION}), encode)
        self.assertIs(encoder_for({"prompt_version": BOS_PROMPT_VERSION}), encode_bos)
        with self.assertRaises(ValueError):
            encoder_for({"prompt_version": "other"})

    def test_bos_prompt_keeps_the_no_truncation_limit(self):
        from training.model.decision_model import encode, encode_bos

        length = len(encode(ROW, Tokenizer(), 4096)["ids"])
        encode_bos(ROW, Tokenizer(), length + 1)
        with self.assertRaises(ValueError):
            encode_bos(ROW, Tokenizer(), length)


def tiny_gemma():
    config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        num_experts=4,
        top_k_experts=2,
        enable_moe_block=True,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        attention_k_eq_v=True,
    )
    return Gemma4TextModel(config)


def tiny_qwen_moe():
    config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    return Qwen3_5MoeTextModel(config)


@unittest.skipIf(Gemma4TextConfig is None, "Transformers MoE models are unavailable")
class MoETargets(unittest.TestCase):
    def assert_frozen_experts(self, backbone, targets):
        from peft import LoraConfig, get_peft_model

        self.assertTrue(targets)
        self.assertFalse(
            any(
                ".experts" in t or ".router" in t or t.endswith("mlp.gate")
                for t in targets
            )
        )
        peft = get_peft_model(
            backbone,
            LoraConfig(r=2, lora_alpha=4, target_modules=targets, bias="none"),
        )
        trainable = [n for n, p in peft.named_parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 2 * len(targets))
        self.assertFalse(any("experts" in n or "router" in n for n in trainable))

    def test_gemma_targets_attention_and_dense_mlp_only(self):
        from training.model.decision_model import moe_summary
        from training.model.lora import select_target_modules

        backbone = tiny_gemma()
        targets = select_target_modules(backbone)
        by_layer = [
            sorted(t.split(".", 2)[2] for t in targets if t.startswith(f"layers.{i}."))
            for i in range(2)
        ]
        self.assertIn("self_attn.v_proj", by_layer[0])
        self.assertNotIn("self_attn.v_proj", by_layer[1])
        for layer in by_layer:
            self.assertTrue(
                {
                    "mlp.down_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "self_attn.k_proj",
                    "self_attn.o_proj",
                    "self_attn.q_proj",
                }
                <= set(layer)
            )
        summary = moe_summary(backbone)
        self.assertEqual(summary["num_experts"], 4)
        self.assertEqual(summary["experts_per_token"], 2)
        self.assertEqual(
            summary["active_text_parameters_per_token"],
            summary["text_parameters"] - summary["routed_expert_parameters"] // 2,
        )
        self.assert_frozen_experts(backbone, targets)

    def test_qwen_moe_targets_attention_and_shared_expert_only(self):
        from training.model.lora import select_target_modules

        backbone = tiny_qwen_moe()
        targets = select_target_modules(backbone)
        self.assertIn("layers.0.linear_attn.in_proj_qkv", targets)
        self.assertIn("layers.1.self_attn.q_proj", targets)
        self.assertIn("layers.1.mlp.shared_expert.down_proj", targets)
        self.assertNotIn("layers.1.mlp.shared_expert_gate", targets)
        self.assertEqual(len(targets), 5 + 3 + 4 + 3)
        self.assert_frozen_experts(backbone, targets)


if __name__ == "__main__":
    unittest.main()
