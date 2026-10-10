"""d3 teacher labels for the family: text rows, image rows, and the labelled training mixture.

    python -m d25.family.teacher text --ckpt CKPT --rows A.jsonl.gz [...] --out OUT --gpus 8
    python -m d25.family.teacher image --ckpt CKPT --rows A.jsonl.gz [...] --out OUT --gpus 8
    python -m d25.family.teacher mix --mix-dir M2-v5 --probs 'OUT*/probs.jsonl' --out DIR

``text`` runs the Vega engine (``d25.vega.eval.run_rows``) and ``image`` the Omni engine
(``d25.omni.eval.run_rows``) over training-format rows with a fixed shard count, so a run resumes
correctly on any number of GPUs: shards are worked in waves of ``--gpus`` until ``OUT/probs.jsonl``
covers every row. ``mix`` attaches the probabilities as ``meta.teachers.<name>`` and sets
``target = g * gold + (1 - g) * teacher`` with ``d25.vega.data.teacher_mix`` (the rule that built
M2T-v5 from pplx-decider-v1.1).
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


def _complete(out: Path) -> bool:
    """Both engines write ``summary.json`` only after a merge; the Vega one also counts misses."""
    summary = out / "summary.json"
    if not (out / "probs.jsonl").exists() or not summary.exists():
        return False
    return json.loads(summary.read_text()).get("missing", 0) == 0


def _run(cmd: list[str]) -> None:
    print(json.dumps({"event": "run", "cmd": cmd[:6]}), flush=True)
    code = subprocess.call(cmd)
    if code:
        raise SystemExit(code)


def text(args) -> None:
    out = Path(args.out)
    devices = "0" if args.gpus <= 1 else f"0-{args.gpus - 1}"
    for _ in range(-(-args.num_shards // args.gpus) + 2):
        if _complete(out):
            break
        _run(
            [
                sys.executable,
                "-m",
                "d25.vega.eval.run_rows",
                "run",
                "--ckpt",
                args.ckpt,
                "--rows",
                *args.rows,
                "--out",
                str(out),
                "--devices",
                devices,
                "--num-shards",
                str(args.num_shards),
            ]
        )
    if not _complete(out):
        raise SystemExit(f"{out}: labels incomplete")


def image(args) -> None:
    out = Path(args.out)
    if _complete(out):
        return
    common = ["--ckpt", args.ckpt, "--rows", *args.rows, "--out", str(out)]
    common += ["--num-shards", str(args.num_shards)]
    for start in range(0, args.num_shards, args.gpus):
        wave = list(range(start, min(start + args.gpus, args.num_shards)))
        procs = [
            subprocess.Popen(
                [sys.executable, "-m", "d25.omni.eval.run_rows", "run"]
                + common
                + ["--shard", str(k), "--device", f"cuda:{i}"]
            )
            for i, k in enumerate(wave)
        ]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(max(codes))
    _run([sys.executable, "-m", "d25.omni.eval.run_rows", "merge"] + common)


def mix(args) -> None:
    from d25.vega.data import teacher_mix

    files = sorted({Path(p) for pattern in args.probs for p in glob.glob(pattern)})
    if not files:
        raise SystemExit(f"no probs files match {args.probs}")
    manifest = teacher_mix.build(
        Path(args.mix_dir), {args.name: files}, Path(args.out), args.gold_weight
    )
    print(json.dumps({"counts": manifest["counts"], "rows": manifest["rows"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("text", "image"):
        p = sub.add_parser(name)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--rows", nargs="+", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--gpus", type=int, default=8)
        p.add_argument("--num-shards", type=int, default=8)
    m = sub.add_parser("mix")
    m.add_argument("--mix-dir", required=True)
    m.add_argument("--probs", nargs="+", required=True, help="glob(s) of probs.jsonl")
    m.add_argument("--out", required=True)
    m.add_argument("--name", default="d3")
    m.add_argument("--gold-weight", type=float, default=0.5)
    args = parser.parse_args()
    {"text": text, "image": image, "mix": mix}[args.cmd](args)


if __name__ == "__main__":
    main()
