"""Are concatenated projections (one GEMM over stacked weights) bit-identical to the separate GEMMs, and faster?

python -m d25.vega.latency.gemmcat --hidden 5120 --groups 17408,17408 10240,6144,48,48 12288,1024,1024
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F


def timeit(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument(
        "--groups",
        nargs="+",
        default=["17408,17408", "10240,6144,48,48", "12288,1024,1024"],
    )
    ap.add_argument(
        "--tokens", nargs="+", type=int, default=[64, 342, 700, 1351, 2900, 5763]
    )
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda:0"
    out = []
    for group in args.groups:
        sizes = [int(s) for s in group.split(",")]
        ws = [
            torch.randn(n, args.hidden, device=dev, dtype=torch.bfloat16) * 0.02
            for n in sizes
        ]
        cat = torch.cat(ws, 0)
        for m in args.tokens:
            x = torch.randn(m, args.hidden, device=dev, dtype=torch.bfloat16)
            sep = [F.linear(x, w) for w in ws]
            whole = F.linear(x, cat)
            exact = all(
                torch.equal(s, p) for s, p in zip(sep, torch.split(whole, sizes, -1))
            )
            t_sep = timeit(lambda: [F.linear(x, w) for w in ws])
            t_cat = timeit(lambda: F.linear(x, cat))
            row = {
                "group": group,
                "tokens": m,
                "exact": exact,
                "separate_us": round(t_sep, 1),
                "concat_us": round(t_cat, 1),
            }
            out.append(row)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
