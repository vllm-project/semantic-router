"""Direct CUDA causal-conv1d versus torch-reference comparison; no kernel selector.

    python tools/causal_conv1d_parity.py --output conv-parity.json

No model downloads or package code are used. Inputs are generated on CPU from
a fixed seed. The small witness stores complete raw tensor bits; larger cases
store hashes and differing output samples. A non-exact result is evidence,
not a probe failure: exit 1 means nonfinite or nondeterministic outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_parity import compare_tensors
from vllm_srun.accel.gpu import _wrap_conv
from vllm_srun.accel.kernels import causal_conv1d_ref

# B,C,T,KW. Eos: (2*16 + 16)*128 = 6144; Nox: (2*16 + 32)*128 = 8192.
DEFAULT_SHAPES = ("1,8,16,4", "1,6144,16,4", "2,6144,65,4", "1,8192,129,4")
MODES = {
    "bf16-autocast": (torch.bfloat16, torch.float32, True),
    "bf16-resident": (torch.bfloat16, torch.bfloat16, False),
    "float32": (torch.float32, torch.float32, False),
}
INTEGER_DTYPES = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def bits(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu().contiguous()
    width = value.element_size() * 8
    return value.view(INTEGER_DTYPES[value.dtype]).to(torch.int64) & ((1 << width) - 1)


def tensor_record(tensor: torch.Tensor, include_bits: bool) -> dict:
    value = tensor.detach().cpu().contiguous()
    record = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    if include_bits:
        record["bits"] = bits(value).reshape(-1).tolist()
    return record


def comparison(left: torch.Tensor, right: torch.Tensor, sample_count: int = 4) -> dict:
    """Compare bytes and monotonic IEEE encodings, including distinct signed zeros."""
    result = compare_tensors(left, right)
    a, b = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    ab, bb = bits(a).reshape(-1), bits(b).reshape(-1)
    width = a.element_size() * 8
    sign, mask = 1 << (width - 1), (1 << width) - 1

    def ordered(value):
        return torch.where(value & sign != 0, mask - value, value + sign)

    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result["finite"] = finite
    result["max_ulp_distance"] = (
        int((ordered(ab) - ordered(bb)).abs().max()) if finite and ab.numel() else None
    )
    indices = (ab != bb).nonzero().flatten()[:sample_count].tolist()

    def number(value):
        value = float(value)
        return value if math.isfinite(value) else None

    result["samples"] = [
        {
            "flat_index": index,
            "causal_conv1d": number(a.reshape(-1)[index]),
            "torch_reference": number(b.reshape(-1)[index]),
            "causal_bits": int(ab[index]),
            "reference_bits": int(bb[index]),
        }
        for index in indices
    ]
    return result


def parse_shape(spec: str) -> tuple[int, int, int, int]:
    values = tuple(int(part) for part in spec.split(","))
    if len(values) != 4 or min(values) < 1 or values[3] not in (2, 3, 4):
        raise ValueError("shape must be positive B,C,T,KW, with KW in 2,3,4")
    if values[1] % 8:
        raise ValueError("channels must be a multiple of 8 for the transpose layout")
    return values


def inputs(shape, mode, layout, seed, device):
    batch, channels, length, kernel = shape
    input_dtype, weight_dtype, _ = MODES[mode]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Match the projection's [B,T,C].transpose(1,2) layout before convolution.
    hidden = (
        torch.randn(batch, length, channels, generator=generator)
        .to(input_dtype)
        .to(device)
        .transpose(1, 2)
    )
    if layout == "contiguous":
        hidden = hidden.contiguous()
    weight = (torch.randn(channels, kernel, generator=generator) * 0.02).to(
        device=device, dtype=weight_dtype
    )
    return hidden, weight


def run_case(conv, shape, mode, layout, activation, seed, repeats, raw_limit, device):
    hidden, weight = inputs(shape, mode, layout, seed, device)
    use_autocast = MODES[mode][2]

    def run(fn):
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_autocast
            else nullcontext()
        )
        with torch.inference_mode(), context:
            return fn(hidden, weight, bias=None, activation=activation)

    # These are the two actual implementations, irrespective of exact/consent policy.
    causal, reference = run(conv), run(causal_conv1d_ref)
    torch.cuda.synchronize(device)
    result = comparison(causal, reference)
    deterministic = {"causal_conv1d": True, "torch_reference": True}
    for _ in range(repeats - 1):
        deterministic["causal_conv1d"] &= compare_tensors(causal, run(conv))["exact"]
        deterministic["torch_reference"] &= compare_tensors(
            reference, run(causal_conv1d_ref)
        )["exact"]
    raw = hidden.numel() <= raw_limit
    return {
        "shape_bctk": list(shape),
        "mode": mode,
        "autocast_dtype": "bfloat16" if use_autocast else None,
        "layout": layout,
        "input_stride": list(hidden.stride()),
        "activation": activation,
        "bias": None,
        "seed": seed,
        "weight_scale": 0.02,
        "repeats": repeats,
        "deterministic": deterministic,
        "comparison": result,
        "tensors": {
            "input": tensor_record(hidden, raw),
            "weight": tensor_record(weight, raw),
            "causal_conv1d": tensor_record(causal, raw),
            "torch_reference": tensor_record(reference, raw),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", action="append")
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument(
        "--layout", action="append", choices=("contiguous", "transpose")
    )
    parser.add_argument("--activation", action="append", choices=("none", "silu"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--raw-limit", type=int, default=256)
    args = parser.parse_args()
    if args.repeats < 2 or args.raw_limit < 0:
        parser.error("repeats must be >=2 and raw-limit must be >=0")
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip:
        parser.error("requires an NVIDIA CUDA device")
    from causal_conv1d import causal_conv1d_fn

    properties = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver = None
    conv = _wrap_conv(causal_conv1d_fn)
    rows = []
    for spec in args.shape or DEFAULT_SHAPES:
        shape = parse_shape(spec)
        for mode in args.mode or MODES:
            for layout in args.layout or ("contiguous", "transpose"):
                for activation in args.activation or ("none", "silu"):
                    row = run_case(
                        conv,
                        shape,
                        mode,
                        layout,
                        None if activation == "none" else activation,
                        args.seed,
                        args.repeats,
                        args.raw_limit,
                        device,
                    )
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "shape_bctk",
                                    "mode",
                                    "layout",
                                    "activation",
                                    "comparison",
                                )
                            }
                        ),
                        flush=True,
                    )
    result = {
        "schema": "cuda-causal-conv1d-exactness/1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "device": properties.name,
        "arch": f"sm_{properties.major}{properties.minor}",
        "torch": torch.__version__,
        "driver_versions": driver,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "causal_conv1d": importlib.metadata.version("causal-conv1d"),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "calls": ["_wrap_conv(causal_conv1d_fn)", "causal_conv1d_ref"],
        "kernel_selector_used": False,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_source_sha256": {
            fn.__name__: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            for fn in (_wrap_conv, causal_conv1d_ref)
        },
        "byteorder": sys.byteorder,
        "input_generation": "CPU float32 torch.randn, then dtype cast and device copy; weight scale 0.02",
        "bit_encoding": "unsigned IEEE encoding; logical row-major order; native byte-order SHA256",
        "ulp_distance": "monotonic IEEE encoding distance; -0 and +0 are adjacent",
        "non_exact_observed": any(not r["comparison"]["exact"] for r in rows),
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return (
        0
        if all(
            r["comparison"]["finite"] and all(r["deterministic"].values()) for r in rows
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
