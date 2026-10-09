"""Single-node launcher for the Vega trainer and tools.

Starts ``--nproc`` worker processes of ``python -m <module> <args>`` with the torch.distributed
environment on loopback. Unlike torchrun it forwards SIGTERM to the workers and waits for them as
long as they need (the pod's grace period bounds it), so a preempted run can finish its update and
write a checkpoint. If any worker fails, the others are stopped and the failing exit code returned.

    python -m d25.vega.train.launch --nproc 8 -- d25.vega.train.train --run ... [trainer args]
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--log-dir",
        help="Append each worker's output to <dir>/rank<r>.log (rank 0 also echoed)",
    )
    parser.add_argument("module")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    port = options.port or free_port()
    procs: list[subprocess.Popen] = []
    pumps: list[threading.Thread] = []
    forwarded = {"signal": None}
    log_dir = Path(options.log_dir) if options.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    def pump(stream, path: Path, echo: bool) -> None:
        with path.open("ab") as handle:
            for line in iter(stream.readline, b""):
                handle.write(line)
                handle.flush()
                if echo:
                    sys.stdout.buffer.write(line)
                    sys.stdout.flush()

    def forward(signum, _frame):
        if forwarded["signal"] is None:
            forwarded["signal"] = signum
            print(
                f"[launch] received signal {signum}; forwarding to {len(procs)} workers",
                flush=True,
            )
            for proc in procs:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    args = options.args[1:] if options.args[:1] == ["--"] else options.args
    for rank in range(options.nproc):
        env = dict(os.environ)
        env.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE=str(options.nproc),
            LOCAL_WORLD_SIZE=str(options.nproc),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
        if log_dir:
            proc = subprocess.Popen(
                [sys.executable, "-m", options.module, *args],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            thread = threading.Thread(
                target=pump,
                args=(proc.stdout, log_dir / f"rank{rank}.log", rank == 0),
                daemon=True,
            )
            thread.start()
            pumps.append(thread)
        else:
            proc = subprocess.Popen(
                [sys.executable, "-m", options.module, *args], env=env
            )
        procs.append(proc)
    failed = None
    while True:
        codes = [proc.poll() for proc in procs]
        if all(code is not None for code in codes):
            break
        bad = [(rank, code) for rank, code in enumerate(codes) if code not in (None, 0)]
        if bad and failed is None and forwarded["signal"] is None:
            failed = bad[0]
            print(
                f"[launch] worker {failed[0]} exited with {failed[1]}; stopping the others",
                flush=True,
            )
            for proc in procs:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
            deadline = time.time() + 120
            while time.time() < deadline and any(proc.poll() is None for proc in procs):
                time.sleep(1)
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
        time.sleep(1)
    for thread in pumps:
        thread.join(timeout=10)
    codes = [proc.returncode for proc in procs]
    print(f"[launch] worker exit codes {codes}", flush=True)
    if failed is not None:
        return failed[1] if failed[1] > 0 else 1
    nonzero = [code for code in codes if code != 0]
    if not nonzero:
        return 0
    positive = [code for code in nonzero if code > 0]
    return max(positive) if positive else 1


if __name__ == "__main__":
    sys.exit(main())
