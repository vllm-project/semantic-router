"""Evaluate one checkpoint on several vision suites with one shard per GPU, then merge and summarise.

    python -m d25.omni.eval.eval_ckpt --ckpt CKPT --out OUT --gpus 8 \
        --suite public=/data/d25/omni/suite/vision-0.3.1 --suite cvbench=/data/d25/omni/proxy/cvbench-proxy

Each suite runs ``d25.omni.eval.run_suite`` (resumable shards) into ``OUT/<name>/`` and is merged
there; ``OUT/summary.json`` collects every suite's ``scores.json``. A finished suite (its
``scores.json`` newer than every shard file) is skipped, so the step can be re-run after a restart.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def shard_command(args, name: str, suite: str, shard: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "d25.omni.eval.run_suite",
        "run",
        "--ckpt",
        args.ckpt,
        "--suite",
        suite,
        "--out",
        str(Path(args.out) / name),
        "--shard",
        str(shard),
        "--num-shards",
        str(args.gpus),
        "--device",
        f"cuda:{shard}",
        "--max-pixels",
        str(args.max_pixels),
    ]


def run_suite(args, name: str, suite: str) -> dict:
    out = Path(args.out) / name
    scores = out / "scores.json"
    if scores.exists():
        return json.loads(scores.read_text())
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    procs = []
    for shard in range(args.gpus):
        log = open(out / f"shard-{shard:02d}.log", "a")
        procs.append(
            (
                subprocess.Popen(
                    shard_command(args, name, suite, shard),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
                log,
            )
        )
    codes = [proc.wait() for proc, _ in procs]
    for _, log in procs:
        log.close()
    if any(codes):
        raise SystemExit(f"{name}: shard exit codes {codes}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "d25.omni.eval.run_suite",
            "merge",
            "--ckpt",
            args.ckpt,
            "--suite",
            suite,
            "--out",
            str(out),
            "--num-shards",
            str(args.gpus),
        ],
        check=True,
    )
    result = json.loads(scores.read_text())
    result.setdefault("seconds", round(time.time() - began, 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=1_638_400)
    parser.add_argument("--suite", action="append", required=True, help="NAME=DIR")
    args = parser.parse_args()
    summary = {"ckpt": args.ckpt, "suites": {}}
    for spec in args.suite:
        name, suite = spec.split("=", 1)
        if not Path(suite, "rows.jsonl.gz").exists():
            summary["suites"][name] = {"missing": suite}
            continue
        summary["suites"][name] = run_suite(args, name, suite)
    path = Path(args.out) / "summary.json"
    path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                k: (
                    v.get("public", v.get("skill", v.get("missing")))
                    if isinstance(v, dict)
                    else v
                )
                for k, v in summary["suites"].items()
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
