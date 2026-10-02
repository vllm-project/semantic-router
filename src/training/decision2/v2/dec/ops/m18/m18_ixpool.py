"""Node side (decoder M18; copy of M17's m17_ixpool.py): IX1 parity gates and full runs of the M17 candidates over M17's GPUs, the 4B
Index-first worker's pool (dec/ops/4bif/pool.py at afa88b47f) with M17 names for its logs, panel record and void.
M18's GPUs are handed to the harness's lease form first (owner track=eval-ix1, status=released, purpose naming dec-m18).

    python3 m18_ixpool.py --mirror DIR --panel panel-8 --gpus "0 1 2 3" NAME [NAME ...]

Every NAME is a ``launch.sh`` DIAGNOSTIC entry of the mirror. In NAME order: the 86-request parity gate on one GPU
(``launch.sh parity``), then each shard of the panel as its own ``launch.sh run --only k`` with shard k on the GPU it
is dispatched to. A GPU is taken only if its lease owner file is absent, is a released eval-ix1 lease, or names one
of these NAMEs (never another job's active lease, even between that job's shards), and rocm-smi shows it idle (the
harness checks again and refuses a busy GPU). An attempt the harness refuses before anything ran (its parity
reference pass or shard directory never started) is moved to ``void/`` with its write-once launcher record and
dispatched again; a parity gate or shard that ran and failed stops its NAME (never rerun); the others go on. The
panel is recorded in ``runs/NAME/m18-panel``; progress goes to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

R = Path("/data/dev2/private/eval/index021/ix1")
LEASES = Path("/data/dev2/leases")


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def allowed(gpu: int, names: list[str]) -> bool:
    path = LEASES / f"gpu{gpu}.lock" / "owner"
    text = path.read_text() if path.is_file() else ""
    if not text.strip():
        return True
    lines = text.splitlines()
    if "track=eval-ix1" in lines and any(
        l.startswith("status=released") for l in lines
    ):
        return True
    return any(
        re.search(rf"^purpose=IX1 .* {re.escape(n)}$", text, re.M) for n in names
    )


def idle(gpus: list[int]) -> set[int]:
    out = subprocess.run(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    cards = json.loads(out)
    return {
        g
        for g in gpus
        if float(cards[f"card{g}"]["GPU use (%)"]) <= 5
        and int(cards[f"card{g}"]["VRAM Total Used Memory (B)"]) <= 2 * 2**30
    }


def void(paths: list[Path], tag: str) -> Path:
    target = R / "void" / f"m18-{tag}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    target.mkdir(parents=True)
    for path in paths:
        if path.exists():
            path.rename(target / path.name)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split()]
    src = args.mirror / "src/training/decision2"
    launch = src / "v2/eval/ix1/launch.sh"
    panel = R / args.panel
    n = len(list(panel.glob("shard-*-of-*.jsonl.gz")))
    assert n and (panel / f"shard-0-of-{n}.jsonl.gz").is_file(), panel
    names = args.names
    parity: dict[str, str] = {}
    shards = {name: {k: "pending" for k in range(n)} for name in names}
    mine: dict[int, tuple] = {}
    for name in names:
        done = R / "parity" / name / "parity.json"
        if done.is_file():
            parity[name] = "pass" if json.loads(done.read_text())["pass"] else "fail"
        elif (R / "parity" / name).exists():
            parity[name] = "fail"
            log(
                f"{name}: an earlier parity attempt left {R / 'parity' / name}; void it first"
            )
        else:
            parity[name] = "pending"
        run = R / "runs" / name
        for k in range(n):
            if (run / f"shard-{k}" / "end_epoch").is_file():
                code = (run / f"shard-{k}" / "exit_code").read_text().strip()
                shards[name][k] = "done" if code == "0" else "fail"
            elif (run / f"shard-{k}").exists():
                shards[name][k] = "fail"
                log(
                    f"{name} shard {k} exists without an end; resume it with launch.sh resume"
                )
    log(f"pool {gpus} for {names} over {args.panel} ({n} shards)")
    while True:
        for g, job in list(mine.items()):
            name, what = job[0], job[1]
            if what == "parity":
                if job[2].poll() is None:
                    continue
                done = R / "parity" / name / "parity.json"
                ok = (
                    job[2].returncode == 0
                    and done.is_file()
                    and json.loads(done.read_text())["pass"]
                )
                if (
                    not ok
                    and not (R / "parity" / name / "ref" / "start_epoch").is_file()
                ):
                    parity[name] = "pending"
                    log(
                        f"{name} parity on gpu{g} refused before its reference pass; voided into "
                        f"{void([R / 'parity' / name], name + '-parity')}"
                    )
                else:
                    parity[name] = "pass" if ok else "fail"
                    log(
                        f"{name} parity on gpu{g}: {'PASS' if ok else 'FAILED'} (exit {job[2].returncode})"
                    )
            else:
                w = R / "runs" / name / f"shard-{what}"
                if not (w / "end_epoch").is_file():
                    continue
                code = (w / "exit_code").read_text().strip()
                shards[name][what] = "done" if code == "0" else "fail"
                log(f"{name} shard {what} on gpu{g}: exit {code}")
            del mine[g]
        active = [
            m for m in names if parity[m] != "fail" and "fail" not in shards[m].values()
        ]
        if (
            not any(
                s in ("pending", "running") for m in active for s in shards[m].values()
            )
            and not mine
        ):
            break
        free = [g for g in gpus if g not in mine and allowed(g, names)]
        free = [g for g in free if g in idle(free)] if free else []
        for name in active:
            if not free:
                break
            if parity[name] == "pending":
                g = free.pop(0)
                rows = panel / "compat-86.gold-free.jsonl.gz"
                proc = subprocess.Popen(
                    [
                        "bash",
                        str(launch),
                        "parity",
                        "--src",
                        str(args.mirror),
                        "--model",
                        name,
                        "--gpu",
                        str(g),
                        "--run",
                        str(R / "parity" / name),
                        "--rows",
                        str(rows),
                    ],
                    cwd=src,
                    stdout=open(R / "logs" / f"m18-pool-parity-{name}.log", "a"),
                    stderr=subprocess.STDOUT,
                )
                parity[name] = "running"
                mine[g] = (name, "parity", proc)
                log(f"{name} parity started on gpu{g}")
                continue
            if parity[name] != "pass":
                continue
            run = R / "runs" / name
            run.mkdir(parents=True, exist_ok=True)
            (run / "m18-panel").write_text(args.panel + "\n")
            for k in range(n):
                if not free:
                    break
                if shards[name][k] != "pending":
                    continue
                g = free.pop(0)
                order = [g if i == k else gpus[0] for i in range(n)]
                result = subprocess.run(
                    [
                        "bash",
                        str(launch),
                        "run",
                        "--src",
                        str(args.mirror),
                        "--model",
                        name,
                        "--gpus",
                        " ".join(map(str, order)),
                        "--only",
                        str(k),
                        "--run",
                        str(run),
                        "--rows-dir",
                        str(panel),
                        "--cache",
                        str(R / "parity" / name / "cache-frozen"),
                    ],
                    cwd=src,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    why = (result.stderr or result.stdout).strip()[-300:]
                    if (run / f"shard-{k}" / "launched").is_file():
                        shards[name][k] = "fail"
                        log(
                            f"{name} shard {k} launch on gpu{g} FAILED after its start: {why}"
                        )
                    else:
                        where = void(
                            [run / f"launcher-run-only-{k}.json"], f"{name}-shard{k}"
                        )
                        log(
                            f"{name} shard {k} refused on gpu{g} ({why}); launcher record voided into {where}"
                        )
                    break
                shards[name][k] = "running"
                mine[g] = (name, k)
                log(f"{name} shard {k} started on gpu{g}")
        time.sleep(30)
    for name in names:
        log(f"{name}: parity {parity[name]}, shards {shards[name]}")


if __name__ == "__main__":
    main()
