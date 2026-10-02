"""Backbone shapes of the released Decision 2.0 packages (read from their ``backbone/config.json``).

Vega-27B is the pinned Qwen3.8-27B base plus a rank-64 LoRA on every Linear;
the others are full checkpoints. Kai-0.6B is a dense Qwen3 backbone (every
layer full attention, full RoPE); the others are Qwen3.5-family hybrids with
three Gated DeltaNet layers per full (gated) attention layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Backbone:
    name: str
    hidden: int
    intermediate: int
    layers: int
    full_layers: int
    heads: int
    kv_heads: int
    head_dim: int
    gdn_k_heads: int = 0
    gdn_v_heads: int = 0
    gdn_k_dim: int = 128
    gdn_v_dim: int = 128
    conv_kernel: int = 4
    rotary_dim: int = 0
    gated_attention: bool = True
    lora_rank: int = 0

    @property
    def gdn_layers(self) -> int:
        return self.layers - self.full_layers

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_k_heads * self.gdn_k_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_v_heads * self.gdn_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    def gemms(self) -> dict[str, tuple[int, int, int]]:
        """Per-layer GEMMs as name -> (N, K, count per forward); M is the row count.

        Names follow the reference modules; ``*_merged`` rows are the
        concatenations a fused path would use (open-jev-fast merges 9 GEMMs per
        layer into 4).
        """
        h, i = self.hidden, self.intermediate
        q_out = self.heads * self.head_dim * (2 if self.gated_attention else 1)
        kv_out = self.kv_heads * self.head_dim
        out = {
            "gate_proj": (i, h, self.layers),
            "up_proj": (i, h, self.layers),
            "down_proj": (h, i, self.layers),
            "gate_up_merged": (2 * i, h, self.layers),
            "q_proj": (q_out, h, self.full_layers),
            "k_proj": (kv_out, h, self.full_layers),
            "v_proj": (kv_out, h, self.full_layers),
            "o_proj": (h, self.heads * self.head_dim, self.full_layers),
            "qkv_merged": (q_out + 2 * kv_out, h, self.full_layers),
        }
        if self.gdn_layers:
            nv = self.gdn_v_heads
            out.update(
                {
                    "in_proj_qkv": (self.conv_dim, h, self.gdn_layers),
                    "in_proj_z": (self.gdn_value_dim, h, self.gdn_layers),
                    "in_proj_b": (nv, h, self.gdn_layers),
                    "in_proj_a": (nv, h, self.gdn_layers),
                    "out_proj": (h, self.gdn_value_dim, self.gdn_layers),
                    "in_proj_merged": (
                        self.conv_dim + self.gdn_value_dim + 2 * nv,
                        h,
                        self.gdn_layers,
                    ),
                }
            )
        return out


BACKBONES = {
    "kai-0.6b": Backbone(
        "kai-0.6b",
        1024,
        3072,
        28,
        28,
        16,
        8,
        128,
        rotary_dim=128,
        gated_attention=False,
    ),
    "eos-0.8b": Backbone(
        "eos-0.8b", 1024, 3584, 24, 6, 8, 2, 256, 16, 16, rotary_dim=64
    ),
    "sol-2b": Backbone("sol-2b", 2048, 6144, 24, 6, 8, 2, 256, 16, 16, rotary_dim=64),
    "nox-4b": Backbone("nox-4b", 2560, 9216, 32, 8, 16, 4, 256, 16, 32, rotary_dim=64),
    "lux-9b": Backbone("lux-9b", 4096, 12288, 32, 8, 16, 4, 256, 16, 32, rotary_dim=64),
    "vega-27b": Backbone(
        "vega-27b", 5120, 17408, 64, 16, 24, 4, 256, 16, 48, rotary_dim=64, lora_rank=64
    ),
}

ROWS = (64, 128, 256, 384, 512, 768, 1024, 2048, 4096)
PRIORITY = ("vega-27b", "eos-0.8b", "nox-4b")
