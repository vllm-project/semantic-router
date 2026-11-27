"""Compare fused ``attn_prep`` to the eager Qwen3.5 / Qwen3 reference.

    python3 tools/attn_prep_parity.py --device cuda:0 --output OUT.json

The reference is ``attn_prep_ref``: Zero-centered (or Qwen3) RMSNorm on each
head, then partial (or full) rotary, matching the native decoder path. Does
not enable the CUDA fused slot. Enable only if every representative shape is
bit-exact and model parity still passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from kernel_parity import compare_tensors  # noqa: E402
from vllm_srun.accel.kernels import attn_prep_ref  # noqa: E402

DEFAULT_SHAPES = (
    # B, T, heads, kv_heads, head_dim, rotary_dim, gated
    "1,16,8,2,128,32,1",
    "2,32,8,2,128,32,1",
    "1,8,4,2,128,128,0",
    "1,17,8,2,256,64,1",
)


def parse_shape(spec: str) -> tuple[int, ...]:
    parts = [int(piece) for piece in spec.split(",")]
    if len(parts) != 7:
        raise SystemExit(f"shape {spec!r} must be B,T,H,Hkv,D,ROT,gated")
    _batch, _length, heads, kv_heads, head_dim, rotary_dim, gated = parts
    if min(parts[:6]) < 1 or gated not in (0, 1):
        raise SystemExit("shape dimensions must be positive and gated must be 0 or 1")
    if head_dim not in (128, 256) or heads % kv_heads:
        raise SystemExit(
            "head width must be 128 or 256 and heads must be divisible by kv_heads"
        )
    if rotary_dim > head_dim or rotary_dim % 2:
        raise SystemExit("rotary width must be even and no larger than the head width")
    return tuple(parts)


def tensors(
    batch: int,
    length: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    gated: bool,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(0)
    q_width = heads * (2 * head_dim if gated else head_dim)
    return {
        "q_proj": torch.randn(
            batch, length, q_width, device=device, dtype=torch.bfloat16
        ),
        "k_proj": torch.randn(
            batch, length, kv_heads * head_dim, device=device, dtype=torch.bfloat16
        ),
        "qw1": torch.randn(head_dim, device=device, dtype=torch.float32),
        "kw1": torch.randn(head_dim, device=device, dtype=torch.float32),
        "cos": torch.randn(
            batch, length, rotary_dim, device=device, dtype=torch.bfloat16
        ),
        "sin": torch.randn(
            batch, length, rotary_dim, device=device, dtype=torch.bfloat16
        ),
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "gated": gated,
        "zero_centred": gated,
        "eps": 1e-6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", action="append", dest="shapes")
    args = parser.parse_args()
    kind, _, index = args.device.partition(":")
    if kind != "cuda" or not torch.cuda.is_available():
        raise SystemExit("attn_prep_parity requires a CUDA device")
    device = torch.device("cuda", int(index or 0))
    from vllm_srun.accel import triton_gfx942 as fused

    rows = []
    all_exact = True
    for spec in args.shapes or DEFAULT_SHAPES:
        batch, length, heads, kv_heads, head_dim, rotary_dim, gated_i = parse_shape(
            spec
        )
        gated = bool(gated_i)
        values = tensors(
            batch, length, heads, kv_heads, head_dim, rotary_dim, gated, device
        )
        ref_q, ref_k = attn_prep_ref(
            values["q_proj"],
            values["k_proj"],
            values["qw1"],
            values["kw1"],
            values["cos"],
            values["sin"],
            heads,
            kv_heads,
            head_dim,
            values["eps"],
            gated=gated,
            zero_centred=gated,
        )
        fused_q, fused_k = fused.attn_prep(
            values["q_proj"].contiguous(),
            values["k_proj"].contiguous(),
            values["qw1"],
            values["kw1"],
            values["cos"].contiguous(),
            values["sin"].contiguous(),
            heads,
            kv_heads,
            head_dim,
            values["eps"],
            gated=gated,
            zero_centred=gated,
        )
        torch.cuda.synchronize(device)
        outputs = {
            "q": compare_tensors(fused_q, ref_q),
            "k": compare_tensors(fused_k, ref_k),
        }
        exact = all(item["exact"] for item in outputs.values())
        all_exact = all_exact and exact
        row = {"shape": spec, "outputs": outputs, "exact": exact}
        print(json.dumps(row), flush=True)
        rows.append(row)

    result = {
        "schema": "model-runtime-attn-prep-parity/1",
        "device": str(device),
        "enable": False,
        "note": (
            "Does not change CUDAAccelerator registration. "
            "Passing attn_prep alone does not enable the fused decoder path."
        ),
        "passed": all_exact,
        "shapes": rows,
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"passed": all_exact}))
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
