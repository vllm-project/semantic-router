"""Tree attention (Triton) against PyTorch SDPA on ROCm at head dim 256, gate included.

    python3 -m v2.runtime_kernels.bench_tree --out RUN [--models eos-0.8b,nox-4b,vega-27b]
        [--rows 128,256,384,512,1024]

Masks: ``causal`` (one unpadded sequence: the shipped runtime's common case, where SDPA can
use its flash kernel with ``is_causal``) and ``tree`` (a packed prefix tree: shared prefix,
three question segments, two or three candidate tails each; every row sees its ancestors and
itself). References: SDPA with the boolean mask (on this ROCm build only the math backend
accepts a mask) and, for ``causal``, flash SDPA with ``is_causal``; each followed by the
reference sigmoid gate. Ours: ``tree_attention`` with the gate fused (P@V in BF16, and in
FP32). Writes RUN/tree.json.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="eos-0.8b,nox-4b,vega-27b")
    ap.add_argument("--rows", default="128,256,384,512,1024")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from . import reference as ref
    from .fidelity import compare
    from .shapes import BACKBONES
    from .timing import time_call, time_graph
    from .masks import pack_mask, tree_mask
    from .tree_attention import tree_attention

    dev = torch.device("cuda")
    report: dict[str, Any] = {"device": torch.cuda.get_device_name(0), "shapes": {}}
    for name in args.models.split(","):
        bb = BACKBONES[name]
        H, HKV, D = bb.heads, bb.kv_heads, bb.head_dim
        scale = D**-0.5
        for N in [int(r) for r in args.rows.split(",")]:
            g = torch.Generator(device=dev).manual_seed(N + H)
            q = torch.randn(1, H, N, D, generator=g, device=dev).to(torch.bfloat16)
            k = torch.randn(1, HKV, N, D, generator=g, device=dev).to(torch.bfloat16)
            v = (
                torch.randn(1, N, HKV, D, generator=g, device=dev)
                .to(torch.bfloat16)
                .transpose(1, 2)
            )
            qp = (torch.randn(1, N, H, 2 * D, generator=g, device=dev) * 2).to(
                torch.bfloat16
            )
            gate_view = qp[..., D:]
            for kind in ("causal", "tree"):
                key = f"{name}/N{N}/{kind}"
                mask = (
                    torch.ones(N, N, dtype=torch.bool, device=dev).tril()
                    if kind == "causal"
                    else tree_mask(torch, N, dev)
                )
                bits = pack_mask(mask[None])
                rec: dict[str, Any] = {
                    "mask_density_vs_causal": mask.sum().item() / (N * (N + 1) / 2)
                }

                def gated(attn_bthd: Any) -> Any:
                    return ref.sigmoid_gate(
                        torch, attn_bthd, gate_view.reshape(1, N, -1)
                    )

                def sdpa_math() -> Any:
                    with sdpa_kernel([SDPBackend.MATH]):
                        o = torch.nn.functional.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=mask[None, None],
                            scale=scale,
                            enable_gqa=True,
                        )
                    return gated(o.transpose(1, 2).contiguous())

                def sdpa_flash() -> Any:
                    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                        o = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, is_causal=True, scale=scale, enable_gqa=True
                        )
                    return gated(o.transpose(1, 2).contiguous())

                variants = {"sdpa_mask": sdpa_math}
                if kind == "causal":
                    variants["sdpa_flash_causal"] = sdpa_flash
                variants["triton"] = lambda: tree_attention(
                    q, k, v, bits, gate_view, scale
                )
                variants["triton_pv_fp32"] = lambda: tree_attention(
                    q, k, v, bits, gate_view, scale, pv_fp32=True
                )
                outs = {}
                for vname, fn in variants.items():
                    try:
                        outs[vname] = fn()
                        rec[vname] = {
                            "eager": time_call(fn, torch, iters=args.iters),
                            "graph": time_graph(fn, torch),
                        }
                    except Exception:  # noqa: BLE001
                        rec[vname] = {"error": traceback.format_exc()[-1500:]}
                for vname, out in outs.items():
                    if vname != "sdpa_mask" and "sdpa_mask" in outs:
                        rec[vname]["fidelity_vs_sdpa_mask"] = compare(
                            torch, outs["sdpa_mask"], out
                        )
                    if vname.startswith("triton") and "sdpa_flash_causal" in outs:
                        rec[vname]["fidelity_vs_flash"] = compare(
                            torch, outs["sdpa_flash_causal"], out
                        )
                report["shapes"][key] = rec
                print(
                    json.dumps(
                        {
                            "key": key,
                            **{
                                v: (
                                    round(r["eager"]["median_us"], 1),
                                    round(
                                        (r.get("graph") or {}).get("median_us", -1), 1
                                    ),
                                )
                                for v, r in rec.items()
                                if isinstance(r, dict) and "eager" in r
                            },
                        }
                    ),
                    flush=True,
                )
                (args.out / "tree.json").write_text(
                    json.dumps(report, indent=1, sort_keys=True)
                )
    print(json.dumps({"done": str(args.out / "tree.json")}))


if __name__ == "__main__":
    main()
