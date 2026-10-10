"""Unattended per-node Omni runner: a file queue of items on node disk, worked through in order.

    python -m d25.omni.runner --node 03 --gpus 8        # inside the runner Job's pod

Queue ``/data/d25/omni/queue/<NN>/``: ``pending/ running/ done/ failed/`` hold item JSON files
(``<order>-<name>.json``); ``state/<name>/`` holds per-step logs and markers; ``runner.log`` and
``heartbeat.json`` show progress. The runner resumes the item in ``running/`` or takes the
lexicographically first ready file in ``pending/``. Anyone may add items at any time; a restarted pod
resumes at the first step without a ``.done`` marker (steps must be idempotent or resumable).

Item keys: ``name``; ``steps`` [{"run": bash, "timeout_h": float, "creates": path (skip when it
exists)}]; ``wait_for`` [paths that must exist]; ``defer`` ("skip": later ready items run while this
one waits; otherwise it waits in place up to ``wait_timeout_h``); ``max_attempts`` (default 2).

Step environment: ``D25_CODE`` (code dir of the tag ``/data/d25/omni/src/current`` points to when the
item starts; it is pinned for the item), ``D25_NODE``, ``D25_GPUS``, ``D25_ITEM``, ``D25_STATE``,
``D25_RESULTS``, ``PYTHONPATH`` (FLA overlay, Omni overlays, code), and the pod's HF variables.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("D25_OMNI_ROOT", "/data/d25/omni"))
CURRENT = ROOT / "src" / "current"
RESULTS = ROOT / "results"
FLA_OVERLAY = "/data/d25/shared/pylib/fla052"
POLL_S = 30


def now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


class Runner:
    def __init__(self, node: str, gpus: int):
        self.node, self.gpus = node, gpus
        self.queue = ROOT / "queue" / node
        for sub in ("pending", "running", "done", "failed", "state"):
            (self.queue / sub).mkdir(parents=True, exist_ok=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.current: dict = {}
        self.child: subprocess.Popen | None = None
        self.stopping = False
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def log(self, message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        with open(self.queue / "runner.log", "a") as handle:
            handle.write(line + "\n")

    def event(self, **fields) -> None:
        with open(RESULTS / "events.jsonl", "a") as handle:
            handle.write(
                json.dumps({"time": now(), "node": self.node, **fields}) + "\n"
            )

    def _heartbeat(self) -> None:
        while True:
            beat = {"time": now(), "pid": os.getpid(), **self.current}
            tmp = self.queue / "heartbeat.json.tmp"
            tmp.write_text(json.dumps(beat, indent=1))
            tmp.replace(self.queue / "heartbeat.json")
            time.sleep(30)

    @staticmethod
    def name_of(item: Path) -> str:
        try:
            return json.loads(item.read_text()).get("name") or item.stem
        except (OSError, ValueError):
            return item.stem

    @staticmethod
    def ready(spec: dict) -> bool:
        return all(Path(p).exists() for p in spec.get("wait_for") or [])

    def next_item(self) -> Path | None:
        running = sorted((self.queue / "running").glob("*.json"))
        if running:
            return running[0]
        for item in sorted((self.queue / "pending").glob("*.json")):
            try:
                spec = json.loads(item.read_text())
            except (OSError, ValueError) as error:
                self.finish(item, "failed", f"unreadable spec: {error}")
                continue
            if self.ready(spec):
                return self.claim(item)
            if spec.get("defer") != "skip":
                waited = time.time() - item.stat().st_mtime
                if waited > float(spec.get("wait_timeout_h", 24)) * 3600:
                    self.finish(item, "failed", "inputs never appeared")
                    continue
                return None
        return None

    def claim(self, item: Path) -> Path:
        target = self.queue / "running" / item.name
        item.replace(target)
        return target

    def finish(self, item: Path, outcome: str, reason: str = "") -> None:
        target = self.queue / outcome / item.name
        shutil.move(str(item), target)
        if reason:
            (self.queue / outcome / f"{item.name}.reason.txt").write_text(reason + "\n")
        self.log(f"{outcome.upper()} {item.name} {reason}".rstrip())
        self.event(item=item.name, outcome=outcome, reason=reason)

    def env(self, code: Path, name: str, state: Path) -> dict[str, str]:
        env = dict(os.environ)
        overlays = [FLA_OVERLAY] + sorted(
            str(p) for p in (ROOT / "pylib").glob("gpu-*")
        )
        env.update(
            D25_CODE=str(code),
            D25_NODE=self.node,
            D25_GPUS=str(self.gpus),
            D25_ITEM=name,
            D25_STATE=str(state),
            D25_RESULTS=str(RESULTS),
            PYTHONPATH=":".join(
                overlays + [str(code / "src" / "training" / "decision2")]
            ),
        )
        return env

    def run_step(self, run: str, env: dict, log: Path, timeout_h: float) -> int:
        with open(log, "a") as handle:
            handle.write(f"\n===== {now()} start\n")
            handle.flush()
            self.child = subprocess.Popen(
                ["bash", "-c", run],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=env["D25_CODE"],
                start_new_session=True,
            )
            try:
                code = self.child.wait(timeout=timeout_h * 3600)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
                code = 124
            handle.write(f"===== {now()} exit {code}\n")
        self.child = None
        return code

    def run_item(self, item: Path) -> None:
        spec = json.loads(item.read_text())
        name = spec.get("name") or item.stem
        state = self.queue / "state" / name
        state.mkdir(parents=True, exist_ok=True)
        pin = state / "code"
        if not pin.exists():
            pin.write_text(str(CURRENT.resolve()) + "\n")
        code = Path(pin.read_text().strip())
        env = self.env(code, name, state)
        attempts_file = state / "attempts"
        attempts = int(attempts_file.read_text()) if attempts_file.exists() else 0
        for index, step in enumerate(spec.get("steps") or []):
            done = state / f"step-{index:02d}.done"
            creates = step.get("creates")
            if done.exists() or (creates and Path(creates).exists()):
                continue
            self.current = {"item": name, "step": index, "since": now()}
            self.log(f"RUN {name} step {index}: {step['run'][:160]}")
            rc = self.run_step(
                step["run"],
                env,
                state / f"step-{index:02d}.log",
                float(step.get("timeout_h", 12)),
            )
            if self.stopping:
                return
            if rc != 0:
                attempts += 1
                attempts_file.write_text(str(attempts))
                if attempts >= int(spec.get("max_attempts", 2)) or step.get("fatal"):
                    self.current = {}
                    self.finish(
                        item,
                        "failed",
                        f"step {index} exit {rc} after {attempts} attempt(s)",
                    )
                    return
                self.log(f"RETRY {name} step {index} (exit {rc}, attempt {attempts})")
                return
            done.write_text(now() + "\n")
        self.current = {}
        self.finish(item, "done")

    def loop(self) -> None:
        self.log(
            f"runner start node {self.node} gpus {self.gpus} code {CURRENT.resolve()}"
        )
        while not self.stopping:
            item = self.next_item()
            if item is None:
                self.current = {"idle": True}
                time.sleep(POLL_S)
                continue
            self.run_item(item)

    def stop(self, *_args) -> None:
        self.stopping = True
        if self.child is not None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--node", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    runner = Runner(args.node, args.gpus)
    signal.signal(signal.SIGTERM, runner.stop)
    signal.signal(signal.SIGINT, runner.stop)
    runner.loop()
    # A non-zero exit makes the Job restart the pod, which resumes the running item.
    raise SystemExit(143)


if __name__ == "__main__":
    main()
