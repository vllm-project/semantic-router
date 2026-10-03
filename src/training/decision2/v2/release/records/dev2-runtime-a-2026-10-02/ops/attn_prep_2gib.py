"""attn_prep with a query projection above 2 GiB: a released fast_kernels.py against a new one, on one GPU.

    python3 attn_prep_2gib.py --old OLD/fast_kernels.py --new NEW/fast_kernels.py --output OUT.json [--rows 9]

Nox-4B's attention widths: 16 query heads (query and gate, head dim 256), 4 key heads, RoPE over 64 dims, BF16
projections of ``rows`` x 16,384 tokens. With 9 rows the query projection holds 2.25 GiB and the key projection
0.28 GiB: Triton marks a pointer argument ``tt.pointer_range = 32`` only for a tensor of at most 2 GiB, so the two
pointers of the released kernel's runtime branch differ there. Records whether each kernel compiles and runs on the
full batch and compares every row that runs with the released kernel run on a copy of that row alone (every tensor
below 2 GiB), bit for bit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=9)
    args = parser.parse_args()
    import torch
    from triton.backends.amd.compiler import HIPBackend

    old, new = load(args.old, "fast_kernels_old"), load(args.new, "fast_kernels_new")
    rows, T, heads, kv_heads, D, rot, eps = args.rows, 16384, 16, 4, 256, 64, 1e-6
    g = torch.Generator(device="cuda").manual_seed(20261003)
    qp = torch.randn(rows, T, heads * 2 * D, generator=g, device="cuda").to(
        torch.bfloat16
    )
    kp = torch.randn(rows, T, kv_heads * D, generator=g, device="cuda").to(
        torch.bfloat16
    )
    qw1 = 1.0 + 0.1 * torch.randn(D, generator=g, device="cuda")
    kw1 = 1.0 + 0.1 * torch.randn(D, generator=g, device="cuda")
    inv_freq = 1.0 / (
        10000000.0 ** (torch.arange(0, rot, 2, device="cuda").float() / rot)
    )
    angles = torch.outer(torch.arange(T, device="cuda").float(), inv_freq)
    emb = torch.cat([angles, angles], dim=-1)[None]
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
    call = (qw1, kw1, cos, sin, heads, kv_heads, D, eps)
    out = {
        "schema": "dev2-attn-prep-2gib/1",
        "shapes": {
            "rows": rows,
            "tokens": T,
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": D,
            "rotary": rot,
        },
        "within_2gib": {
            "q_proj": HIPBackend.is_within_2gb(qp),
            "k_proj": HIPBackend.is_within_2gb(kp),
        },
        "bytes": {
            "q_proj": qp.untyped_storage().size(),
            "k_proj": kp.untyped_storage().size(),
        },
        "torch": torch.__version__,
        "device": torch.cuda.get_device_properties(0).gcnArchName,
    }
    results = {}
    for label, module in (("old", old), ("new", new)):
        try:
            q, k = module.attn_prep(qp, kp, *call)
            torch.cuda.synchronize()
            results[label] = (q, k)
            out[label] = {"full_batch": "ran"}
        except Exception as error:
            out[label] = {
                "full_batch": "failed",
                "error": f"{type(error).__name__}: {str(error).splitlines()[0][:300]}",
                "traceback_tail": traceback.format_exc().splitlines()[-3:],
            }
    equal = {}
    for b in range(rows):
        q1, k1 = old.attn_prep(qp[b : b + 1].clone(), kp[b : b + 1].clone(), *call)
        for label, (q, k) in results.items():
            same = torch.equal(q[b : b + 1], q1) and torch.equal(k[b : b + 1], k1)
            equal.setdefault(label, []).append(same)
    out["rows_equal_to_old_per_row"] = {label: sum(v) for label, v in equal.items()}
    out["passed"] = (
        out["new"]["full_batch"] == "ran"
        and out["rows_equal_to_old_per_row"].get("new") == rows
        and out["within_2gib"] == {"q_proj": False, "k_proj": True}
    )
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "within_2gib",
                    "old",
                    "new",
                    "rows_equal_to_old_per_row",
                    "passed",
                )
            }
        )
    )
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
