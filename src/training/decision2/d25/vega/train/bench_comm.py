"""RCCL collective and GEMM throughput on one node (run under ``d25.vega.train.launch``).

python -m d25.vega.train.launch --nproc 8 d25.vega.train.bench_comm --out /data/d25/vega/train/bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def timed(fn, iters: int) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    began = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - began) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    results = {"world": world, "collectives": [], "gemm": []}
    for mb in (64, 512, 2048):
        for dtype in (torch.bfloat16, torch.float32):
            numel = mb * 2**20 // torch.tensor([], dtype=dtype).element_size()
            numel -= numel % world
            full = torch.empty(numel, dtype=dtype, device=device)
            shard = torch.empty(numel // world, dtype=dtype, device=device)
            t_ag = timed(lambda: dist.all_gather_into_tensor(full, shard), args.iters)
            t_rs = timed(lambda: dist.reduce_scatter_tensor(shard, full), args.iters)
            t_ar = timed(lambda: dist.all_reduce(full), args.iters)
            size = numel * full.element_size()
            factor = (world - 1) / world
            results["collectives"].append(
                {
                    "bytes": size,
                    "dtype": str(dtype).replace("torch.", ""),
                    "all_gather_busbw_GBps": size * factor / t_ag / 1e9,
                    "reduce_scatter_busbw_GBps": size * factor / t_rs / 1e9,
                    "all_reduce_busbw_GBps": 2 * size * factor / t_ar / 1e9,
                }
            )
            del full, shard
    for m, k, n in (
        (8192, 5120, 17408),
        (16384, 5120, 17408),
        (16384, 17408, 5120),
        (16384, 5120, 10240),
    ):
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        t = timed(lambda: a @ b, args.iters * 3)
        results["gemm"].append(
            {"m": m, "k": k, "n": n, "tflops": 2 * m * k * n / t / 1e12}
        )
    torch.cuda.synchronize()
    if rank == 0:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)
        print(json.dumps(results, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
