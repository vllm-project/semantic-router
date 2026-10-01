"""Isolate the FLA gated-delta kernels at a padded batch shape (random tensors, no model).

    python3 -m v2.eval.ix1.fla_probe --batch B --length T --heads H [--dim 128]

Runs ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` (as Transformers calls it: BF16 q/k/v,
FP32 g, L2-normalized q/k in the kernel) on [B, T, H, D] inputs, then again one batch element at
a time, and reports the largest difference and any non-finite output. The two must agree:
batch elements are independent.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch


def inputs(batch, length, heads, dim, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, length, heads, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)
    beta = torch.rand((batch, length, heads), device="cuda", generator=g).to(
        torch.bfloat16
    )
    gate = -torch.rand((batch, length, heads), device="cuda", generator=g) * 0.1
    return q, k, v, gate, beta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v, gate, beta = inputs(args.batch, args.length, args.heads, args.dim)
    elements = args.batch * args.length * args.heads * args.dim
    report = {**vars(args), "elements": elements, "over_int32": elements > 2**31 - 1}
    print(json.dumps({"event": "start", **report}), flush=True)
    with torch.inference_mode():
        full, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=gate,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            output_final_state=False,
        )
        torch.cuda.synchronize()
        report["full_nonfinite"] = int((~torch.isfinite(full)).sum())
        worst, per_element = 0.0, []
        for b in range(args.batch):
            one, _ = chunk_gated_delta_rule(
                q[b : b + 1],
                k[b : b + 1],
                v[b : b + 1],
                g=gate[b : b + 1],
                beta=beta[b : b + 1],
                use_qk_l2norm_in_kernel=True,
                output_final_state=False,
            )
            diff = (
                (full[b : b + 1].float() - one.float())
                .abs()
                .nan_to_num(float("inf"))
                .max()
                .item()
            )
            worst = max(worst, diff)
            per_element.append(diff)
        report["max_abs_diff_vs_per_element"] = worst
        report["diff_by_element"] = per_element
        report["first_element_past_int32"] = -(
            -(2**31) // (args.length * args.heads * args.dim)
        )
    print(json.dumps({"event": "result", **report}), flush=True)
    sys.exit(0 if worst == 0.0 and report["full_nonfinite"] == 0 else 1)


if __name__ == "__main__":
    main()
