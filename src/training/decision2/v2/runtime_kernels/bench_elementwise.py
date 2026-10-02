"""Per-kernel timing of the six fused element-wise kernels at the released backbones' shapes.

    python3 -m v2.runtime_kernels.bench_elementwise --out RUN [--models eos-0.8b,nox-4b,vega-27b]
        [--rows 128,512,2048] [--compile-rows 512] [--iters 200]

For every backbone x row count (one sequence of M tokens) and kernel it times
the reference op sequence in eager mode (``time_call``: synchronised, median
of ``--iters``) and inside one HIP graph (``time_graph``: launch gaps
removed), the Triton kernel, ``torch.compile`` of the reference (Inductor, at
``--compile-rows`` only, default and with ``emulate_precision_casts``) and
AITER where a counterpart exists, plus each output's fidelity against the
reference on the same inputs. Inputs are random with activation-like scales;
real-activation fidelity is measured by ``capture``. Writes RUN/elementwise.json.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Callable


def make_inputs(torch: Any, bb: Any, M: int, dev: Any) -> dict[str, Any]:
    g = torch.Generator(device=dev).manual_seed(1234 + M)

    def rn(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=g, device=dev) * scale).to(dtype)

    H, I = bb.hidden, bb.intermediate
    x: dict[str, Any] = {
        "res": rn(1, M, H, scale=3.0),
        "delta": rn(1, M, H, scale=0.5, dtype=torch.bfloat16),
        "norm_w": rn(H, scale=0.1),
        "gate": rn(1, M, I, scale=2.0, dtype=torch.bfloat16),
        "up": rn(1, M, I, dtype=torch.bfloat16),
    }
    x["gate_up"] = torch.cat([x["gate"], x["up"]], -1)
    nh, nkv, hd = bb.heads, bb.kv_heads, bb.head_dim
    qw = nh * hd * (2 if bb.gated_attention else 1)
    x.update(
        qp=rn(1, M, qw, scale=2.0, dtype=torch.bfloat16),
        kp=rn(1, M, nkv * hd, scale=2.0, dtype=torch.bfloat16),
        vp=rn(1, M, nkv * hd, dtype=torch.bfloat16),
        qn_w=rn(hd, scale=0.1),
        kn_w=rn(hd, scale=0.1),
        attn=rn(1, M, nh, hd, dtype=torch.bfloat16),
    )
    rot = bb.rotary_dim
    inv_freq = 1.0 / (
        10000000 ** (torch.arange(0, rot, 2, dtype=torch.float, device=dev) / rot)
    )
    from . import reference as ref

    x["cos"], x["sin"] = ref.rotary_cos_sin(
        torch, inv_freq, torch.arange(M, device=dev)[None], torch.float32
    )
    if bb.gdn_layers:
        nk, nv, dk = bb.gdn_k_heads, bb.gdn_v_heads, bb.gdn_k_dim
        C = bb.conv_dim
        x.update(
            mixed=rn(1, M, C, scale=2.0, dtype=torch.bfloat16),
            conv_w=rn(C, bb.conv_kernel, scale=0.3),
            b=rn(1, M, nv, dtype=torch.bfloat16),
            a=rn(1, M, nv, scale=3.0, dtype=torch.bfloat16),
            A_log=torch.log(torch.rand(nv, generator=g, device=dev) * 15 + 0.5),
            dt_bias=rn(nv),
            core=rn(M * nv, dk, dtype=torch.bfloat16),
            z=rn(M * nv, dk, scale=2.0, dtype=torch.bfloat16),
            gn_w=1.0 + rn(dk, scale=0.1),
        )
    return x


def kernels(
    torch: Any, bb: Any, x: dict[str, Any]
) -> dict[str, dict[str, Callable[[], Any]]]:
    """name -> {"ref": fn, "triton": fn, ...}; every fn returns the comparable output tensor(s)."""
    from . import reference as ref
    from . import triton_elementwise as tk

    eps = 1e-6
    w1 = 1.0 + x["norm_w"].float()
    qw1, kw1 = 1.0 + x["qn_w"].float(), 1.0 + x["kn_w"].float()
    nh, nkv, hd = bb.heads, bb.kv_heads, bb.head_dim
    out: dict[str, dict[str, Callable[[], Any]]] = {
        "add_rmsnorm": {
            "ref": lambda: ref.add_rmsnorm(
                torch, x["res"], x["delta"], x["norm_w"], eps
            )[1],
            "triton": lambda: tk.add_rmsnorm(x["res"], x["delta"], w1, eps)[1],
        },
        "bf16_cast": {
            "ref": lambda: ref.add_rmsnorm(torch, x["res"], None, x["norm_w"], eps)[
                0
            ].to(torch.bfloat16),
        },
        "silu_mul": {
            "ref": lambda: ref.silu_mul(torch, x["gate"], x["up"]),
            "triton": lambda: tk.silu_mul(x["gate"], x["up"]),
            "triton_merged": lambda: tk.silu_mul(x["gate_up"]),
        },
        "attn_prep": {
            "ref": lambda: ref.attn_prep(
                torch,
                x["qp"],
                x["kp"],
                x["vp"],
                x["qn_w"],
                x["kn_w"],
                x["cos"],
                x["sin"],
                hd,
                eps,
            )[:2],
            "triton": lambda: tk.attn_prep(
                x["qp"], x["kp"], qw1, kw1, x["cos"], x["sin"], nh, nkv, hd, eps
            ),
        },
        "sigmoid_gate": {
            "ref": lambda: ref.sigmoid_gate(
                torch,
                x["attn"],
                x["qp"]
                .view(*x["qp"].shape[:2], nh, 2 * hd)[..., hd:]
                .reshape(*x["qp"].shape[:2], -1),
            ),
            "triton": lambda: tk.sigmoid_gate(
                x["attn"], x["qp"].view(*x["qp"].shape[:2], nh, 2 * hd)[..., hd:]
            ),
        },
    }
    if not bb.gated_attention:
        out.pop("sigmoid_gate")
    if bb.gdn_layers:
        nk, dk = bb.gdn_k_heads, bb.gdn_k_dim
        out["gdn_prep"] = {
            "ref": lambda: ref.gdn_prep(
                torch,
                x["mixed"],
                x["b"],
                x["a"],
                x["conv_w"],
                x["A_log"],
                x["dt_bias"],
                nk,
                dk,
                dk,
            ),
            "ref_gva": lambda: ref.gdn_prep(
                torch,
                x["mixed"],
                x["b"],
                x["a"],
                x["conv_w"],
                x["A_log"],
                x["dt_bias"],
                nk,
                dk,
                dk,
                expand_qk=False,
            ),
            "triton": lambda: tk.gdn_prep(
                x["mixed"],
                x["b"],
                x["a"],
                x["conv_w"],
                x["A_log"],
                x["dt_bias"],
                nk,
                dk,
            ),
        }
        out["gated_rmsnorm"] = {
            "ref": lambda: ref.gated_rmsnorm(torch, x["core"], x["z"], x["gn_w"], eps),
            "triton": lambda: tk.gated_rmsnorm(x["core"], x["z"], x["gn_w"], eps),
        }
    return out


def aiter_variants(
    torch: Any, bb: Any, x: dict[str, Any]
) -> dict[str, Callable[[], Any]]:
    """AITER counterparts (BF16 residual for add+RMSNorm: not the runtime's FP32 residual)."""
    import aiter

    H = bb.hidden
    rows = x["res"].numel() // H
    res_bf16 = x["res"].reshape(rows, H).to(torch.bfloat16)
    delta = x["delta"].reshape(rows, H)
    w1_bf16 = (1.0 + x["norm_w"].float()).to(torch.bfloat16)
    out_n = torch.empty_like(delta)
    out_r = torch.empty_like(delta)
    gu = x["gate_up"].reshape(rows, -1)
    out_s = torch.empty(rows, gu.shape[-1] // 2, dtype=torch.bfloat16, device=gu.device)

    def add_norm():
        aiter.rmsnorm2d_fwd_with_add(out_n, delta, res_bf16, out_r, w1_bf16, 1e-6)
        return out_n

    def silu():
        aiter.silu_and_mul(out_s, gu)
        return out_s

    return {"add_rmsnorm": add_norm, "silu_mul": silu}


def flatten(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) else result


def compare_outputs(torch: Any, reference: Any, result: Any) -> Any:
    """``compare`` per output; tuples are compared element by element up to the shorter length."""
    from .fidelity import compare

    if isinstance(reference, tuple) and isinstance(result, tuple):
        return [compare(torch, r, o) for r, o in zip(reference, result)]
    return compare(torch, flatten(reference), flatten(result))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="eos-0.8b,nox-4b,vega-27b")
    ap.add_argument("--rows", default="128,512,2048")
    ap.add_argument("--compile-rows", default="512")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    import torch

    from .fidelity import compare
    from .shapes import BACKBONES
    from .timing import time_call, time_graph

    dev = torch.device("cuda")
    compile_rows = {int(r) for r in args.compile_rows.split(",") if r}
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "rows": {},
        "errors": [],
    }
    try:
        import aiter  # noqa: F401

        have_aiter = True
    except Exception as exc:  # noqa: BLE001
        have_aiter = False
        report["errors"].append(f"aiter import: {exc!r}"[:300])
    for name in args.models.split(","):
        bb = BACKBONES[name]
        for M in [int(r) for r in args.rows.split(",")]:
            key = f"{name}/M{M}"
            x = make_inputs(torch, bb, M, dev)
            entry: dict[str, Any] = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ks = kernels(torch, bb, x)
                extra = aiter_variants(torch, bb, x) if have_aiter else {}
                for kname, variants in ks.items():
                    rec: dict[str, Any] = {}
                    reference = None
                    for vname, fn in variants.items():
                        try:
                            res = fn()
                            rec[vname] = {
                                "eager": time_call(fn, torch, iters=args.iters),
                                "graph": time_graph(fn, torch),
                            }
                            if vname.startswith("ref"):
                                # the last ref* variant (e.g. ref_gva: q/k at the k-head count) is the comparand
                                reference = res
                            elif reference is not None:
                                rec[vname]["fidelity"] = compare_outputs(
                                    torch, reference, res
                                )
                        except Exception:  # noqa: BLE001
                            rec[vname] = {"error": traceback.format_exc()[-1500:]}
                    if kname in extra:
                        try:
                            fn = extra[kname]
                            res = fn()
                            rec["aiter"] = {
                                "eager": time_call(fn, torch, iters=args.iters),
                                "graph": time_graph(fn, torch),
                                "fidelity": compare(
                                    torch, flatten(reference).reshape(res.shape), res
                                ),
                            }
                        except Exception:  # noqa: BLE001
                            rec["aiter"] = {"error": traceback.format_exc()[-1500:]}
                    if M in compile_rows and "ref" in variants:
                        for label, emulate in (
                            ("compile", False),
                            ("compile_emulate", True),
                        ):
                            try:
                                import torch._inductor.config as icfg

                                icfg.emulate_precision_casts = emulate
                                torch._dynamo.reset()
                                compiled = torch.compile(variants["ref"], dynamic=False)
                                res = compiled()
                                rec[label] = {
                                    "eager": time_call(
                                        compiled, torch, iters=args.iters
                                    ),
                                    "graph": time_graph(compiled, torch),
                                    "fidelity": compare_outputs(
                                        torch, variants["ref"](), res
                                    ),
                                }
                            except Exception:  # noqa: BLE001
                                rec[label] = {"error": traceback.format_exc()[-1500:]}
                    entry[kname] = rec
                    print(
                        json.dumps(
                            {
                                "key": key,
                                "kernel": kname,
                                **{
                                    v: (
                                        round(r["eager"]["median_us"], 1),
                                        round(
                                            (r.get("graph") or {}).get("median_us", -1),
                                            1,
                                        ),
                                    )
                                    for v, r in rec.items()
                                    if "eager" in r
                                },
                            }
                        ),
                        flush=True,
                    )
            report["rows"][key] = entry
            del x
            torch.cuda.empty_cache()
            (args.out / "elementwise.json").write_text(
                json.dumps(report, indent=1, sort_keys=True)
            )
    print(json.dumps({"done": str(args.out / "elementwise.json")}))


if __name__ == "__main__":
    main()
