"""Compare fused ``gdn_prep`` to the released GatedDeltaNet prep (all five outputs).

    python3 tools/gdn_prep_parity.py --device cuda:0 --output OUT.json
        [--no-causal-conv1d] [--shape B,T,C,KW,K_HEADS,HEAD_DIM ...]

The released path is ``gdn_prep_ref``: depthwise causal conv + SiLU, split q/k/v,
sigmoid ``beta``, and ``g = -exp(A_log) * softplus(a + dt_bias)``. Acceptance
requires ``causal-conv1d`` (``validation=released-reference``). ``--no-causal-conv1d``
is a torch-only diagnostic and never sets ``passed``. Convolution-only is not
enough: q, k, v, g, and beta are all scored.

Does not enable the CUDA fused slot. Enable only if every representative shape
is bit-exact and model parity still passes.

Shapes default to the H200 fused-kernel probe plus a slightly longer sequence.
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
from vllm_srun.accel.gpu import _optional, _wrap_conv  # noqa: E402
from vllm_srun.accel.kernels import gdn_prep_ref  # noqa: E402

DEFAULT_SHAPES = (
    "1,16,512,4,2,64",
    "2,32,512,4,2,64",
    "1,64,512,4,2,64",
    "1,65,2048,4,4,128",
)
OUTPUTS = ("q", "k", "v", "g", "beta")


def conv_reference(no_causal_conv1d: bool) -> tuple[Any, str, str]:
    """Released-reference is causal-conv1d. Torch-only is diagnostic, not acceptance."""
    if no_causal_conv1d:
        return None, "torch-reference", "torch-diagnostic"
    raw = _optional("causal_conv1d", "causal_conv1d_fn")
    if raw is None:
        raise SystemExit(
            "gdn_prep_parity requires causal-conv1d; pass --no-causal-conv1d "
            "only for a torch-reference diagnostic (that is not released-reference validation)"
        )
    return _wrap_conv(raw), "causal-conv1d", "released-reference"


def parse_shape(spec: str) -> tuple[int, ...]:
    parts = [int(piece) for piece in spec.split(",")]
    if len(parts) != 6:
        raise SystemExit(f"shape {spec!r} must be B,T,C,KW,K_HEADS,HEAD_DIM")
    _batch, _length, channels, kernel, k_heads, head_dim = parts
    if min(parts) < 1 or kernel not in (2, 3, 4):
        raise SystemExit(
            "shape dimensions must be positive; convolution width must be 2, 3 or 4"
        )
    if channels <= 2 * k_heads * head_dim or channels % head_dim:
        raise SystemExit("channels must contain complete query, key and value heads")
    return tuple(parts)


def tensors(
    batch: int,
    length: int,
    channels: int,
    kernel: int,
    k_heads: int,
    head_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(0)
    nv = (channels - 2 * k_heads * head_dim) // head_dim
    return {
        "mixed": torch.randn(
            batch, length, channels, device=device, dtype=torch.bfloat16
        ),
        "b": torch.randn(batch, length, nv, device=device, dtype=torch.bfloat16),
        "a": torch.randn(batch, length, nv, device=device, dtype=torch.bfloat16),
        "conv_w": torch.randn(channels, kernel, device=device, dtype=torch.float32),
        "A_log": torch.randn(nv, device=device, dtype=torch.float32),
        "dt_bias": torch.randn(nv, device=device, dtype=torch.float32),
        "k_heads": k_heads,
        "head_dim": head_dim,
    }


def released(values: dict[str, Any], conv_fn: Any) -> tuple[torch.Tensor, ...]:
    return gdn_prep_ref(
        values["mixed"],
        values["b"],
        values["a"],
        values["conv_w"],
        values["A_log"],
        values["dt_bias"],
        values["k_heads"],
        values["head_dim"],
        conv_fn=conv_fn,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-causal-conv1d", action="store_true")
    parser.add_argument("--shape", action="append", dest="shapes")
    args = parser.parse_args()
    kind, _, index = args.device.partition(":")
    if kind != "cuda" or not torch.cuda.is_available():
        raise SystemExit("gdn_prep_parity requires a CUDA device")
    device = torch.device("cuda", int(index or 0))
    from vllm_srun.accel import triton_gfx942 as fused

    conv_fn, conv_source, validation = conv_reference(args.no_causal_conv1d)

    rows = []
    all_exact = True
    for spec in args.shapes or DEFAULT_SHAPES:
        batch, length, channels, kernel, k_heads, head_dim = parse_shape(spec)
        values = tensors(batch, length, channels, kernel, k_heads, head_dim, device)
        ref = released(values, conv_fn)
        fused_out = fused.gdn_prep(
            values["mixed"].contiguous(),
            values["b"].contiguous(),
            values["a"].contiguous(),
            values["conv_w"],
            values["A_log"],
            values["dt_bias"],
            k_heads,
            head_dim,
        )
        torch.cuda.synchronize(device)
        outputs = {
            name: compare_tensors(fused_out[i], ref[i])
            for i, name in enumerate(OUTPUTS)
        }
        exact = all(item["exact"] for item in outputs.values())
        all_exact = all_exact and exact
        row = {"shape": spec, "conv": conv_source, "outputs": outputs, "exact": exact}
        print(
            json.dumps({"shape": spec, "exact": exact, "outputs": outputs}), flush=True
        )
        rows.append(row)

    accepted = all_exact and validation == "released-reference"
    result = {
        "schema": "model-runtime-gdn-prep-parity/1",
        "device": str(device),
        "conv": conv_source,
        "validation": validation,
        "enable": False,
        "note": (
            "Does not change CUDAAccelerator registration. "
            "Registration is per slot, but the fused decoder path requires "
            "several slots together; passing gdn_prep alone will not activate "
            "that whole path. Enable gdn_prep only after exact=true on "
            "representative shapes and a model-level gpu_parity pass."
        ),
        "passed": accepted,
        "exact_vs_reference": all_exact,
        "shapes": rows,
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": accepted,
                "conv": conv_source,
                "validation": validation,
                "exact_vs_reference": all_exact,
            }
        )
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
