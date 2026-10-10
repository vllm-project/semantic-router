"""Per-GPU memory plan for Omni FSDP2 training on one node of 8 x MI325X (256 GB HBM each).

Model (Qwen3.8-27B as trained here): 25,624,600,064 language parameters (64 layers, hidden 5120,
vocab 248,320; ``lm_head`` and MTP dropped), 460,730,096 vision parameters, a 255 x 5120 readout.
FSDP2 keeps FP32 master parameters, FP32 gradients and FP32 AdamW moments sharded over the 8 ranks,
all-gathers BF16 parameters one unit at a time, and checkpoints every decoder layer and ViT block.

``python -m d25.omni.k8s.train.memory_plan`` prints the plan; the default case is the O-graft-lowlr
arm (everything trainable) with a 32,768-token microbatch budget:

| Item | GB per GPU |
| --- | ---: |
| FP32 parameters, gradients, AdamW moments (16 B x 26.09B / 8) | 52.2 |
| BF16 unsharded units in flight (root incl. embeddings, 3 decoder layers) | 4.8 |
| Embedding gradient before reduce-scatter (BF16 + FP32 copy) | 7.6 |
| Layer-boundary activations (64 x tokens x 5120 x 2 B) | 21.5 |
| Recompute peak of one decoder layer (forward + backward) | 12.2 |
| ViT activations (4 images x 6,400 patches, trainable encoder) | 1.6 |
| Runtime, RCCL buffers, allocator slack | 16.0 |
| **Peak** | **~116** |

The frozen-encoder arms save about 0.6 GB. Activation terms scale linearly with the token budget:
65,536 tokens gives about 150 GB, still below 60% of HBM. The plan assumes SDPA uses a fused
(flash or memory-efficient) kernel for the 16 gated-attention layers (head dim 256) with a padding
mask; a fallback to the math kernel would add ``rows x 24 x L^2 x 6 B`` per layer (about 39 GB for
one 16k-token row), which the first pilot must rule out by reading ``gpu_peak_gb`` in
``training.jsonl``. Maximum row: 4 images x 1,600 visual tokens + text up to 8,192 tokens < 16,384.

Host memory: rank 0 stages the full FP32 state dict (104 GB) when loading and saving; with eight
ranks' processors, rows and image decoding and a 64 GiB ``/dev/shm`` the pod peaks near 330 GB, so
the Job requests 640 GiB. Disk per run: one resume point (FP32 parameters and moments, about
313 GB) plus 52 GB per saved BF16 checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

GB = 1e9
LANGUAGE_PARAMETERS = 25_624_600_064
VISION_PARAMETERS = 460_730_096
ENCODER_PARAMETERS = 411_500_000
READOUT_PARAMETERS = 255 * 5120
EMBEDDING_PARAMETERS = 248_320 * 5120
HIDDEN = 5120
LAYERS = 64
FFN = 17_408
ATTENTION_HEADS = 24
HBM_GB = 256
PATCHES_PER_IMAGE = 6_400
VISION_HIDDEN = 1152
VISION_BLOCKS = 27


@dataclass(frozen=True)
class Plan:
    rows: dict[str, float]

    @property
    def peak(self) -> float:
        return sum(self.rows.values())


def plan(
    token_budget: int = 32_768,
    world: int = 8,
    encoder_trainable: bool = True,
    images: int = 4,
    layers_in_flight: int = 3,
    overhead_gb: float = 16.0,
) -> Plan:
    total = LANGUAGE_PARAMETERS + VISION_PARAMETERS + READOUT_PARAMETERS
    trainable = total - (0 if encoder_trainable else ENCODER_PARAMETERS)
    layer = (LANGUAGE_PARAMETERS - EMBEDDING_PARAMETERS) / LAYERS
    per_token_layer = (HIDDEN * 4 + 12_288 + 2_048 + 6_144 + 3 * FFN) * 2 * 2
    vision = (
        images * PATCHES_PER_IMAGE * VISION_HIDDEN * 2 * VISION_BLOCKS
        if encoder_trainable
        else 0
    )
    rows = {
        "fp32 parameters": total * 4 / world,
        "fp32 gradients": trainable * 4 / world,
        "adamw moments": trainable * 8 / world,
        "bf16 units in flight": (EMBEDDING_PARAMETERS + layers_in_flight * layer) * 2,
        "embedding gradient": EMBEDDING_PARAMETERS * 6,
        "layer-boundary activations": LAYERS * token_budget * HIDDEN * 2,
        "one-layer recompute": token_budget * per_token_layer,
        "vit activations": vision,
    }
    rows = {name: value / GB for name, value in rows.items()}
    rows["runtime and slack"] = overhead_gb
    return Plan(rows)


def math_attention_penalty_gb(row_tokens: int, rows: int = 1) -> float:
    return rows * ATTENTION_HEADS * row_tokens**2 * 6 / GB


def max_token_budget(limit_fraction: float = 0.8, **kwargs) -> int:
    budget = 4096
    while plan(token_budget=budget * 2, **kwargs).peak <= limit_fraction * HBM_GB:
        budget *= 2
    return budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--token-budget", type=int, default=32_768)
    parser.add_argument("--frozen-encoder", action="store_true")
    args = parser.parse_args()
    result = plan(args.token_budget, encoder_trainable=not args.frozen_encoder)
    for name, value in result.rows.items():
        print(f"{name:28s} {value:7.1f} GB")
    print(f"{'peak':28s} {result.peak:7.1f} GB of {HBM_GB} GB")
    print(
        f"math-attention fallback for one 16k row: +{math_attention_penalty_gb(16_384):.1f} GB per layer"
    )
    print(
        f"largest power-of-two budget under 80% HBM: {max_token_budget(encoder_trainable=not args.frozen_encoder)}"
    )


if __name__ == "__main__":
    main()
