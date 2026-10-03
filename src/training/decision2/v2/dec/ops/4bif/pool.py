"""Node side: IX1 parity gates and full runs of several diagnostic packages over GPUs shared with other tracks.

    python3 pool.py --mirror DIR --panel panel-8 --gpus "1 2 3 4 5 6 7" NAME [NAME ...]

Every NAME is a ``launch.sh`` DIAGNOSTIC entry of the mirror. In NAME order: the 86-request parity gate on one GPU
(``launch.sh parity``), then each shard of the panel as its own ``launch.sh run --only k`` with shard k on the GPU it
is dispatched to. A GPU is taken only if its lease owner file is absent, is a released eval-ix1 lease, names one of
these NAMEs, or is an abandoned eval-ix1 lease (its job on this GPU finished STALE seconds ago or more: every shard
the run's launcher records placed on this GPU ended, or its parity gate wrote parity.json; the GPU idle over
IDLE_POLLS consecutive polls; a chain starts its next job within a minute), and rocm-smi shows it idle (the harness
checks again and refuses a busy GPU). A restarted pool adopts its own parity gates and shards whose containers still
run.
Another job's active lease is never taken, even between its shards; a taken-over owner file is kept as
``owner.prev-4bif-<UTC>``. An attempt the harness refuses before anything ran (its parity
reference pass or shard directory never started) is moved to ``void/`` with its write-once launcher record and
dispatched again; a parity gate or shard that ran and failed stops its NAME (never rerun); the others go on. The
panel is recorded in ``runs/NAME/4bif-panel``; progress goes to stdout.
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


STALE = 15 * 60
IDLE_POLLS = 3


def shards_on(run: Path, gpu: int) -> list[Path]:
    """The shard directories of an IX1 run that its launcher records placed on this GPU."""
    found = []
    for record in run.glob("launcher-run*.json"):
        try:
            gpus = [int(g) for g in json.loads(record.read_text())["gpus"]]
        except (OSError, ValueError, KeyError):
            continue
        only = record.stem.removeprefix("launcher-run").removeprefix("-only-")
        ks = [int(k) for k in only.split("_")] if only else range(len(gpus))
        found += [run / f"shard-{k}" for k in ks if k < len(gpus) and gpus[k] == gpu]
    return found


def abandoned(text: str, gpu: int) -> bool:
    """An eval-ix1 lease whose job on this GPU ended at least STALE seconds ago."""
    if "track=eval-ix1" not in text.splitlines():
        return False
    found = re.search(r"^run_dir=(.+)$", text, re.M)
    if not found:
        return False
    run = Path(found.group(1))
    if (run / "parity.json").is_file():
        ends = [run / "parity.json"]
    else:
        ends = [w / "end_epoch" for w in shards_on(run, gpu)]
        if not ends or not all(e.is_file() for e in ends):
            return False
    return time.time() - max(e.stat().st_mtime for e in ends) >= STALE


def containers() -> set[str]:
    out = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    ).stdout
    return set(out.split())


def parity_alive(name: str) -> bool:
    """A launch.sh parity process of NAME (started by an earlier pool) still runs on this node."""
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = cmdline.read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            any(a.endswith(b"launch.sh") for a in args)
            and b"parity" in args
            and name.encode() in args
        ):
            return True
    return False


def slug(name: str) -> str:
    return name.lower().replace(".", "_")


def allowed(gpu: int, names: list[str], streak: dict[int, int]) -> str | None:
    path = LEASES / f"gpu{gpu}.lock" / "owner"
    text = path.read_text() if path.is_file() else ""
    if not text.strip():
        return "free"
    lines = text.splitlines()
    if "track=eval-ix1" in lines and any(
        l.startswith("status=released") for l in lines
    ):
        return "released"
    if any(re.search(rf"^purpose=IX1 .* {re.escape(n)}$", text, re.M) for n in names):
        return "ours"
    if streak.get(gpu, 0) >= IDLE_POLLS and abandoned(text, gpu):
        return "abandoned"
    return None


def keep_owner(gpu: int) -> None:
    path = LEASES / f"gpu{gpu}.lock" / "owner"
    if path.is_file():
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        (path.parent / f"owner.prev-4bif-{stamp}").write_text(path.read_text())
        log(
            f"gpu{gpu}: taking over an abandoned eval-ix1 lease (kept as owner.prev-4bif-{stamp})"
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
    target = R / "void" / f"4bif-{tag}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
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
    streak: dict[int, int] = {}
    running = containers()
    for name in names:
        done = R / "parity" / name / "parity.json"
        live = [c for c in running if c.startswith(f"ix1-parity-{slug(name)}-g")]
        if done.is_file():
            parity[name] = "pass" if json.loads(done.read_text())["pass"] else "fail"
        elif live or parity_alive(name):
            g = int(live[0].rsplit("-g", 1)[1].split("-")[0]) if live else -1
            parity[name] = "running"
            mine[g] = (name, "parity", None)
            log(f"{name}: adopting its running parity gate on gpu{g}")
        elif (R / "parity" / name).exists():
            parity[name] = "fail"
            log(
                f"{name}: an earlier parity attempt left {R / 'parity' / name}; void it first"
            )
        else:
            parity[name] = "pending"
        run = R / "runs" / name
        for k in range(n):
            live = [c for c in running if c.startswith(f"ix1-{slug(name)}-s{k}-g")]
            if (run / f"shard-{k}" / "end_epoch").is_file():
                code = (run / f"shard-{k}" / "exit_code").read_text().strip()
                shards[name][k] = "done" if code == "0" else "fail"
            elif live:
                g = int(live[0].rsplit("-g", 1)[1])
                shards[name][k] = "running"
                mine[g] = (name, k)
                log(f"{name}: adopting its running shard {k} on gpu{g}")
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
                done = R / "parity" / name / "parity.json"
                if job[2] is None:
                    if parity_alive(name):
                        continue
                    code = 0 if done.is_file() else 1
                elif job[2].poll() is None:
                    continue
                else:
                    code = job[2].returncode
                ok = (
                    code == 0
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
                        f"{name} parity on gpu{g}: {'PASS' if ok else 'FAILED'} (exit {code})"
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
        now_idle = idle([g for g in gpus if g not in mine])
        for g in gpus:
            streak[g] = streak.get(g, 0) + 1 if g in now_idle else 0
        why = {g: allowed(g, names, streak) for g in now_idle}
        free = [g for g in gpus if why.get(g)]
        for name in active:
            if not free:
                break
            if parity[name] == "pending":
                g = free.pop(0)
                if why[g] == "abandoned":
                    keep_owner(g)
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
                    stdout=open(R / "logs" / f"4bif-pool-parity-{name}.log", "a"),
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
            (run / "4bif-panel").write_text(args.panel + "\n")
            for k in range(n):
                if not free:
                    break
                if shards[name][k] != "pending":
                    continue
                g = free.pop(0)
                if why[g] == "abandoned":
                    keep_owner(g)
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
