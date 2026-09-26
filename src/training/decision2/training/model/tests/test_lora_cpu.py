"""Optional tiny PEFT save/reload/optimizer-resume check on CPU only."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from torch import nn
except ImportError:
    torch = nn = PeftModel = load_file = None


@unittest.skipIf(torch is None, "CPU torch, PEFT, and safetensors are unavailable")
class LoRACpuRoundTrip(unittest.TestCase):
    def test_adapter_head_and_optimizer_resume_match_uninterrupted_step(self):
        from training.model.decision_model import DecisionModel
        from training.model.lora import adapter_parameters, attach_lora

        class Config:
            layer_types = ["full_attention"]
            num_hidden_layers = 1

            def to_dict(self):
                return {
                    "layer_types": self.layer_types,
                    "num_hidden_layers": self.num_hidden_layers,
                }

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(self.self_attn, name, nn.Linear(4, 4, bias=False))
                self.mlp = nn.Module()
                for name in ("gate_proj", "up_proj", "down_proj"):
                    setattr(self.mlp, name, nn.Linear(4, 4, bias=False))

            def forward(self, x):
                return sum(
                    getattr(self.self_attn, name)(x)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                ) + sum(
                    getattr(self.mlp, name)(x)
                    for name in ("gate_proj", "up_proj", "down_proj")
                )

        class TinyText(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Config()
                self.layers = nn.ModuleList([Block()])

            def forward(self, x):
                return self.layers[0](x)

        class Tokenizer:
            def save_pretrained(self, path):
                (Path(path) / "tokenizer_config.json").write_text("{}")

        torch.manual_seed(7)
        original_base = TinyText()
        initial_state = {
            name: tensor.clone() for name, tensor in original_base.state_dict().items()
        }
        first = DecisionModel(original_base, nn.Linear(4, 1), {"base_revision": "test"})
        fingerprint = {
            "source_name": "tiny",
            "files_sha256": {"model.safetensors": "a" * 64},
        }
        attach_lora(
            first,
            rank=2,
            alpha=4,
            dropout=0.0,
            source_kind="decision1",
            source_fingerprint=fingerprint,
        )

        def optimizer(model):
            return torch.optim.AdamW(
                [
                    {"params": adapter_parameters(model)},
                    {"params": list(model.head.parameters())},
                ],
                lr=0.01,
            )

        def update(model, opt):
            opt.zero_grad(set_to_none=True)
            x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
            loss = model.head(model.backbone(x)).square().mean()
            loss.backward()
            opt.step()

        first_optimizer = optimizer(first)
        update(first, first_optimizer)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            first.save(checkpoint, Tokenizer())
            torch.save(first_optimizer.state_dict(), checkpoint / "optimizer.pt")
            update(first, first_optimizer)

            reloaded_base = TinyText()
            reloaded_base.load_state_dict(initial_state)
            second = DecisionModel(
                reloaded_base, nn.Linear(4, 1), {"base_revision": "test"}
            )
            second.backbone = PeftModel.from_pretrained(
                reloaded_base,
                checkpoint / "adapter",
                is_trainable=True,
                local_files_only=True,
            )
            second.head.load_state_dict(
                load_file(str(checkpoint / "decision_head.safetensors"))
            )
            second.metadata = json.loads(
                (checkpoint / "decision_config.json").read_text()
            )
            second_optimizer = optimizer(second)
            second_optimizer.load_state_dict(
                torch.load(checkpoint / "optimizer.pt", weights_only=True)
            )
            update(second, second_optimizer)
            for (name_a, param_a), (name_b, param_b) in zip(
                first.named_parameters(), second.named_parameters()
            ):
                self.assertEqual(name_a, name_b)
                self.assertTrue(
                    torch.allclose(param_a, param_b, atol=1e-7, rtol=1e-7), name_a
                )
            x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
            second.eval()
            with torch.no_grad():
                before_merge = second.head(second.backbone(x))
            second.merge_lora({"model_sha256": "c" * 64})
            with torch.no_grad():
                after_merge = second.head(second.backbone(x))
            self.assertTrue(
                torch.allclose(before_merge, after_merge, atol=1e-6, rtol=1e-6)
            )
            self.assertEqual(second.metadata["checkpoint_format"], "full")
            self.assertEqual(
                second.metadata["lora_origin"]["adapter"]["model_sha256"], "c" * 64
            )


if __name__ == "__main__":
    unittest.main()
