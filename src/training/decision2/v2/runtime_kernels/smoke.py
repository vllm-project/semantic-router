"""Environment and exactness smoke test on one GPU (node side, no model weights).

    python3 -m v2.runtime_kernels.smoke --out RUN

Records the stack, which ``causal_conv1d_fn`` Transformers dispatches to,
exhaustive BF16 SiLU / sigmoid exactness of the Triton formulas, each fused
kernel against its reference on random inputs, FLA's GVA path against the
reference's repeated q/k, the SDPA backends available at head dim 256 and the
AITER entry points. Writes RUN/smoke.json.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import traceback
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    import triton

    from . import reference as ref
    from . import triton_elementwise as tk
    from .fidelity import compare
    from .timing import time_call

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    out: dict = {
        "device": props.name,
        "arch": getattr(props, "gcnArchName", None),
        "cus": props.multi_processor_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
    }
    try:
        import fla
        import transformers

        out["fla"] = fla.__version__
        out["transformers"] = transformers.__version__
    except Exception as exc:  # noqa: BLE001
        out["import_error"] = repr(exc)

    # which causal_conv1d_fn does Transformers call?
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    impl = None
    for cell in m.causal_conv1d_fn.__closure__ or ():
        value = cell.cell_contents
        if callable(value) and getattr(value, "__module__", "").startswith(
            "causal_conv1d"
        ):
            impl = f"{value.__module__}.{value.__name__}"
    out["causal_conv1d_impl"] = impl
    chunk_impl = None
    for cell in m.torch_chunk_gated_delta_rule.__closure__ or ():
        value = cell.cell_contents
        if callable(value) and getattr(value, "__module__", "").startswith("fla"):
            chunk_impl = f"{value.__module__}.{value.__name__}"
    out["chunk_gated_delta_rule_impl"] = chunk_impl

    # exhaustive BF16 exactness of SiLU and sigmoid
    every = (
        torch.arange(-32768, 32768, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .to(dev)
    )
    finite = torch.isfinite(every)
    x = every[finite].reshape(1, -1).contiguous()
    ones = torch.ones_like(x)
    silu_ref = torch.nn.functional.silu(x)
    silu_tk = tk.silu_mul(x, ones)
    out["silu_exhaustive"] = compare(torch, silu_ref, silu_tk)
    sig_ref = torch.sigmoid(x)
    n = (x.shape[1] // 64) * 64
    xg = x[:, :n].reshape(1, n // 64, 1, 64)
    sig_tk = tk.sigmoid_gate(torch.ones_like(xg), xg).reshape(1, n)
    out["sigmoid_exhaustive"] = compare(torch, sig_ref[:, :n], sig_tk)

    results = {}
    torch.manual_seed(0)
    T, H = 512, 5120
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # add + RMSNorm
        res = torch.randn(1, T, H, device=dev) * 3
        delta = (torch.randn(1, T, H, device=dev) * 0.5).to(torch.bfloat16)
        w = torch.randn(H, device=dev) * 0.1
        h_ref, n_ref = ref.add_rmsnorm(torch, res, delta, w, 1e-6)
        h_tk, n_tk = tk.add_rmsnorm(res, delta, 1.0 + w.float(), 1e-6)
        results["add_rmsnorm.hidden"] = compare(torch, h_ref, h_tk)
        results["add_rmsnorm.normed"] = compare(torch, n_ref, n_tk)
        # SiLU * mul
        gate = (torch.randn(1, T, 17408, device=dev) * 2).to(torch.bfloat16)
        up = (torch.randn(1, T, 17408, device=dev)).to(torch.bfloat16)
        results["silu_mul"] = compare(
            torch, ref.silu_mul(torch, gate, up), tk.silu_mul(gate, up)
        )
        merged = torch.cat([gate, up], -1)
        results["silu_mul.merged"] = compare(
            torch, ref.silu_mul(torch, gate, up), tk.silu_mul(merged)
        )
        # gated RMSNorm
        core = torch.randn(T * 48, 128, device=dev).to(torch.bfloat16)
        z = (torch.randn(T * 48, 128, device=dev) * 2).to(torch.bfloat16)
        wg = 1.0 + torch.randn(128, device=dev) * 0.1
        results["gated_rmsnorm"] = compare(
            torch,
            ref.gated_rmsnorm(torch, core, z, wg, 1e-6),
            tk.gated_rmsnorm(core, z, wg, 1e-6),
        )
        # attention prep (27B shapes)
        nh, nkv, hd = 24, 4, 256
        qp = (torch.randn(1, T, nh * hd * 2, device=dev) * 2).to(torch.bfloat16)
        kp = (torch.randn(1, T, nkv * hd, device=dev) * 2).to(torch.bfloat16)
        vp = torch.randn(1, T, nkv * hd, device=dev).to(torch.bfloat16)
        qw = torch.randn(hd, device=dev) * 0.1
        kw = torch.randn(hd, device=dev) * 0.1
        inv_freq = 1.0 / (
            10000000 ** (torch.arange(0, 64, 2, dtype=torch.float, device=dev) / 64)
        )
        pos = torch.arange(T, device=dev)[None]
        cos, sin = ref.rotary_cos_sin(torch, inv_freq, pos, torch.float32)
        q_ref, k_ref, v_ref, g_ref = ref.attn_prep(
            torch, qp, kp, vp, qw, kw, cos, sin, hd, 1e-6
        )
        q_tk, k_tk = tk.attn_prep(
            qp, kp, 1.0 + qw.float(), 1.0 + kw.float(), cos, sin, nh, nkv, hd, 1e-6
        )
        results["attn_prep.q"] = compare(torch, q_ref, q_tk)
        results["attn_prep.k"] = compare(torch, k_ref, k_tk)
        # sigmoid gate, reading the gate in place from q_proj_out
        attn = torch.randn(1, T, nh, hd, device=dev).to(torch.bfloat16)
        gate_view = qp.view(1, T, nh, 2 * hd)[..., hd:]
        results["sigmoid_gate"] = compare(
            torch,
            ref.sigmoid_gate(torch, attn, g_ref),
            tk.sigmoid_gate(attn, gate_view),
        )
        # Gated DeltaNet prep (27B shapes: 16 k heads, 48 v heads)
        nk, nv, dk = 16, 48, 128
        C = (2 * nk + nv) * dk
        mixed = (torch.randn(1, T, C, device=dev) * 2).to(torch.bfloat16)
        conv_w = torch.randn(C, 4, device=dev) * 0.3
        bb = torch.randn(1, T, nv, device=dev).to(torch.bfloat16)
        aa = (torch.randn(1, T, nv, device=dev) * 3).to(torch.bfloat16)
        A_log = torch.log(torch.rand(nv, device=dev) * 15 + 0.5)
        dt_bias = torch.randn(nv, device=dev)
        try:
            conv_ref = ref.causal_conv1d_silu(torch, mixed, conv_w)
            qr, kr, vr, gr, br = ref.gdn_prep(
                torch,
                mixed,
                bb,
                aa,
                conv_w,
                A_log,
                dt_bias,
                nk,
                dk,
                dk,
                expand_qk=False,
            )
            qt, kt, vt, gt, bt = tk.gdn_prep(
                mixed, bb, aa, conv_w, A_log, dt_bias, nk, dk
            )
            results["gdn_prep.v"] = compare(torch, vr, vt)
            results["gdn_prep.q"] = compare(torch, qr, qt)
            results["gdn_prep.k"] = compare(torch, kr, kt)
            results["gdn_prep.g"] = compare(torch, gr, gt)
            results["gdn_prep.beta"] = compare(torch, br, bt)
            results["gdn_prep.conv_vs_fallback"] = compare(
                torch,
                conv_ref,
                torch.nn.functional.silu(
                    torch.nn.functional.conv1d(
                        mixed.transpose(1, 2).float(),
                        conv_w.unsqueeze(1),
                        None,
                        padding=3,
                        groups=C,
                    )[..., :T]
                )
                .to(torch.bfloat16)
                .transpose(1, 2),
            )
        except Exception:  # noqa: BLE001
            results["gdn_prep.error"] = traceback.format_exc()[-2000:]

        # FLA chunk: GVA path vs repeated q/k (the reference)
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            q16 = torch.nn.functional.normalize(
                torch.randn(1, T, nk, dk, device=dev), dim=-1
            ).to(torch.bfloat16)
            k16 = torch.nn.functional.normalize(
                torch.randn(1, T, nk, dk, device=dev), dim=-1
            ).to(torch.bfloat16)
            vv = torch.randn(1, T, nv, dk, device=dev).to(torch.bfloat16)
            gg = -torch.rand(1, T, nv, device=dev) * 0.5
            be = torch.rand(1, T, nv, device=dev).to(torch.bfloat16)
            o_rep, _ = chunk_gated_delta_rule(
                q16.repeat_interleave(nv // nk, 2),
                k16.repeat_interleave(nv // nk, 2),
                vv,
                gg,
                be,
                use_qk_l2norm_in_kernel=True,
            )
            o_gva, _ = chunk_gated_delta_rule(
                q16, k16, vv, gg, be, use_qk_l2norm_in_kernel=True
            )
            results["fla.gva_vs_repeat"] = compare(torch, o_rep, o_gva)
            out["fla_chunk_us_T512_H48"] = time_call(
                lambda: chunk_gated_delta_rule(
                    q16.repeat_interleave(3, 2),
                    k16.repeat_interleave(3, 2),
                    vv,
                    gg,
                    be,
                    use_qk_l2norm_in_kernel=True,
                ),
                torch,
                5,
                50,
            )
        except Exception:  # noqa: BLE001
            results["fla.error"] = traceback.format_exc()[-2000:]
    out["kernels"] = results

    # SDPA backends at head dim 256
    sdpa = {}
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 24, 384, 256, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 384, 256, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 384, 256, device=dev, dtype=torch.bfloat16)
    mask = torch.ones(384, 384, dtype=torch.bool, device=dev).tril()[None, None]
    for name, backend in [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]:
        for label, kwargs in [
            ("causal", {"is_causal": True}),
            ("boolmask", {"attn_mask": mask}),
        ]:
            try:
                with sdpa_kernel([backend]):
                    t = time_call(
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, enable_gqa=True, **kwargs
                        ),
                        torch,
                        5,
                        50,
                    )
                sdpa[f"{name}.{label}"] = t["median_us"]
            except Exception as exc:  # noqa: BLE001
                sdpa[f"{name}.{label}"] = f"unavailable: {str(exc)[:160]}"
    out["sdpa_hd256_T384"] = sdpa

    try:
        import aiter

        names = sorted(
            n
            for n in dir(aiter)
            if any(
                s in n.lower()
                for s in (
                    "rmsnorm",
                    "silu",
                    "rope",
                    "gated",
                    "sigmoid",
                    "layernorm",
                    "act_and_mul",
                )
            )
        )
        out["aiter"] = {"file": aiter.__file__, "names": names[:200]}
    except Exception as exc:  # noqa: BLE001
        out["aiter"] = f"unavailable: {str(exc)[:300]}"
    try:
        out["rocprofv3"] = subprocess.run(
            ["rocprofv3", "--version"], capture_output=True, text=True, timeout=60
        ).stdout[-300:]
    except Exception as exc:  # noqa: BLE001
        out["rocprofv3"] = repr(exc)
    (args.out / "smoke.json").write_text(
        json.dumps(out, indent=1, sort_keys=True, default=str)
    )
    print(json.dumps(out, indent=1, sort_keys=True, default=str)[:20000])


if __name__ == "__main__":
    main()
