"""Decision 2.0 Sol 2B, Eos 0.8B and Kai 0.6B as small-size anchors on Vega's frozen text proxies.

    python -m d25.family.anchors run --name sol2 --proxy /data/d25/omni/family/proxy/pv1 --out DIR --gpus N

Vega's calibration anchors stop at 9B/4B (Lux, Nox, Kev 9B). The nano, lite and edge gates pair against our
own Decision 2.0 model of the same size, measured here through its released package (the same
``Decision2Engine`` Vega uses for Lux and Nox) on the O part of the proxies, one process per GPU, then
scored with Vega's proxy scorer. ``scores.json`` carries ``O_proxy``, the input ``d25.family.gate
--ref-o-proxy`` expects.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXTRA = {
    "sol2": (
        "vllm-sr/Decision-2.0-Sol-2B",
        "64235bef55dad29387dd16da7c90e038bf2f0972",
        "decision-2.0-sol-2b",
    ),
    "eos2": (
        "vllm-sr/Decision-2.0-Eos-0.8B",
        "3594047d69f476f1d01cf84c593e213fc3a4dfe0",
        "decision-2.0-eos-0.8b",
    ),
    "kai2": (
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764",
        "decision-2.0-kai-0.6b",
    ),
}
MODELS = "/data/d25/omni/family/anchors/models"


def register() -> None:
    from d25.vega.eval.proxy import anchors as A

    for name, (repo, revision, board) in EXTRA.items():
        A.REPOS[name], A.RELEASE[name], A.BOARD[name] = repo, revision, board
    A.MODELS = MODELS
    original = A.make_engine

    def make_engine(name, device="cuda:0", options=None):
        if name in EXTRA:
            return A.Decision2Engine(name, device)
        return original(name, device, options)

    A.make_engine = make_engine


def worker(args) -> None:
    import torch

    # The Decision 2.0 runtime launches kernels on the current device; with several workers per pod the
    # current device must be the worker's own GPU, or kernels fault on another GPU's memory.
    torch.cuda.set_device(torch.device(args.device))
    register()
    from d25.vega.eval.proxy import run

    run.main(
        ["--anchor", args.name, "--rows", *args.rows, "--out", args.out]
        + [
            "--shard",
            str(args.shard),
            "--shards",
            str(args.shards),
            "--device",
            args.device,
        ]
    )


def run_all(args) -> None:
    out = Path(args.out)
    rows = sorted(str(p) for p in (Path(args.proxy) / "o").glob("*.jsonl.gz"))
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "d25.family.anchors", "worker", "--name", args.name]
            + ["--rows", *rows, "--out", str(out)]
            + ["--shard", str(i), "--shards", str(args.gpus), "--device", f"cuda:{i}"]
        )
        for i in range(args.gpus)
    ]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"workers failed with exit codes {codes}")
    from d25.vega.eval.proxy.score import load_results, score_all

    results = load_results([str(p) for p in sorted(out.glob("results.shard*.jsonl"))])
    scores = score_all(args.proxy, results)
    summary = {
        "anchor": args.name,
        "repo": EXTRA[args.name][0],
        "revision": EXTRA[args.name][1],
        "O_proxy": scores["O_proxy"],
        "O_families": scores.get("O_families"),
        "coverage": scores.get("coverage"),
    }
    (out / "scores.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True, choices=sorted(EXTRA))
    r.add_argument("--proxy", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gpus", type=int, default=1)
    w = sub.add_parser("worker")
    w.add_argument("--name", required=True, choices=sorted(EXTRA))
    w.add_argument("--rows", nargs="+", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--shard", type=int, default=0)
    w.add_argument("--shards", type=int, default=1)
    w.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    (run_all if args.cmd == "run" else worker)(args)


if __name__ == "__main__":
    main()
