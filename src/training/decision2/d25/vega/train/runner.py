"""Unattended per-node arm runner: a queue of arm specs on node disk, worked through in order.

    python -m d25.vega.train.runner --node 06          # inside the runner Job's pod (8 GPUs)

Queue ``/data/d25/vega/queue/<NN>/``: ``pending/ running/ done/ failed/`` hold arm spec JSON files
(``<order>-<arm>.json``); ``state/<arm>/`` holds markers; ``runner.log`` and ``heartbeat.json`` show
progress. The runner takes the lexicographically first pending spec (or resumes the one in
``running/``), runs it and moves it to ``done/`` or ``failed/`` (+ ``<file>.reason.txt``). Anyone may
drop new specs into ``pending/`` at any time; a restarted pod resumes where it stopped (the trainer
resumes from its latest DCP checkpoint; finished exports/evals are skipped through markers).

Per arm: wait for data (poll every 2 min up to ``wait_timeout_h``), optional ``prepare`` steps, train
(8 GPUs; pauses after each intermediate export so it can be evaluated, then resumes), evaluate every
export with ws-measure's ``d25.vega.eval.ckpt_eval`` (missing/failing evals leave a failure marker
and the arm continues), copy results to ``/data/d25/vega/results/{<arm>/<step>/,index/}``, append
events to ``/data/d25/vega/results/events.jsonl``, best-effort upload to the private HF dataset
``vllm-sr/d25-vega-results`` under ``<NN>/``.

Spec keys (defaults = Perplexity's recipe): name, train (path or list of candidate paths; the first
ready one is pinned), dev (default: <train dir>/dev.jsonl.gz), init ("base" or a checkpoint dir),
init_kind, attention_mode, lr, readout_lr, warmup_ratio, schedule, min_lr_ratio, epochs, seed,
rows_per_update, max_length, teacher, teacher_weight, brier_weight, export_fractions,
evals {"intermediate": [...], "final": [...]}, wait_for, wait_timeout_h, prepare
[{"run": cmd, "creates": path}], save_every, keep_dcp, token_budget, trainer_args, max_attempts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/d25/vega")
CURRENT = ROOT / "src" / "current"
RESULTS = ROOT / "results"
HF_RESULTS_REPO = "vllm-sr/d25-vega-results"
POLL_DATA_S = 120
DEFAULTS = {
    "init": "base",
    "init_kind": None,
    "attention_mode": "noncausal_full_attention",
    "lr": 2e-6,
    "readout_lr": None,
    "warmup_ratio": 0.15,
    "schedule": "cosine",
    "min_lr_ratio": 0.1,
    "epochs": 1,
    "seed": 20260920,
    "rows_per_update": 256,
    "max_length": 8192,
    "teacher": None,
    "teacher_weight": 0.0,
    "brier_weight": 0.0,
    "export_fractions": [0.3333, 0.6667, 1.0],
    "evals": {"intermediate": ["proxy"], "final": ["proxy", "full"]},
    "wait_for": [],
    "wait_timeout_h": 12,
    "prepare": [],
    "save_every": 1000,
    "keep_dcp": 2,
    "token_budget": 49152,
    "trainer_args": [],
    "max_attempts": 3,
    "eval_timeout_h": {"proxy": 3, "full": 10},
}
STOPPING = {"flag": False}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Runner:
    def __init__(self, node: str, gpus: int):
        self.node = node
        self.gpus = gpus
        self.queue = ROOT / "queue" / node
        for name in ("pending", "running", "done", "failed", "state"):
            (self.queue / name).mkdir(parents=True, exist_ok=True)
        (RESULTS / "index").mkdir(parents=True, exist_ok=True)
        self.status = {"phase": "starting", "item": None}
        self.child: subprocess.Popen | None = None
        threading.Thread(target=self._heartbeat, daemon=True).start()

    # -- bookkeeping -------------------------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{now()} [runner {self.node}] {message}"
        print(line, flush=True)
        with (self.queue / "runner.log").open("a") as handle:
            handle.write(line + "\n")

    def event(self, **fields) -> None:
        record = {"time": now(), "node": self.node, **fields}
        with (RESULTS / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.log(f"event {json.dumps(fields)[:400]}")

    def _heartbeat(self) -> None:
        while True:
            try:
                info = {
                    "time": now(),
                    "node": self.node,
                    "pid": os.getpid(),
                    **self.status,
                }
                item = self.status.get("item")
                if item:
                    tail = ROOT / "ckpt" / item / "logs" / "train.jsonl"
                    if tail.exists():
                        with tail.open("rb") as handle:
                            handle.seek(max(0, tail.stat().st_size - 4000))
                            lines = handle.read().decode(errors="ignore").splitlines()
                        for line in reversed(lines):
                            try:
                                last = json.loads(line)
                                info["last_step"] = {
                                    k: last.get(k)
                                    for k in (
                                        "step",
                                        "loss",
                                        "tokens_per_s",
                                        "peak_alloc_gib",
                                        "time",
                                    )
                                }
                                break
                            except ValueError:
                                continue
                tmp = self.queue / "heartbeat.json.tmp"
                tmp.write_text(json.dumps(info, indent=1))
                os.replace(tmp, self.queue / "heartbeat.json")
            except Exception:  # noqa: BLE001
                pass
            time.sleep(60)

    # -- queue ---------------------------------------------------------------------------------
    def next_item(self) -> Path | None:
        running = sorted(self.queue.glob("running/*.json"))
        if running:
            return running[0]
        for path in sorted(self.queue.glob("pending/*.json")):
            target = self.queue / "running" / path.name
            try:
                os.rename(path, target)
            except FileNotFoundError:
                continue
            return target
        return None

    def finish(self, item: Path, outcome: str, reason: str = "") -> None:
        target = self.queue / outcome / item.name
        if target.exists():
            target = target.with_name(f"{item.stem}.{int(time.time())}.json")
        os.replace(item, target)
        if reason:
            target.with_name(target.name + ".reason.txt").write_text(reason + "\n")
        self.log(f"{item.name} -> {outcome} {reason}")

    # -- helpers -------------------------------------------------------------------------------
    @staticmethod
    def code_dir() -> Path:
        return CURRENT.resolve()

    def env(self, code: Path) -> dict[str, str]:
        env = dict(os.environ)
        overlay = env.get("D25_FLA_OVERLAY", "/data/d25/shared/pylib/fla052")
        env["PYTHONPATH"] = f"{overlay}:{code}"
        env["D25_CODE_DIR"] = str(code)
        return env

    def run(
        self,
        cmd: list[str] | str,
        code: Path,
        log_path: Path,
        timeout: float | None = None,
    ) -> int:
        """Run a child (process group), forwarding SIGTERM; returns its exit code (-9 on timeout)."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        shell = isinstance(cmd, str)
        with log_path.open("ab") as out:
            self.child = subprocess.Popen(
                cmd,
                cwd=str(code),
                env=self.env(code),
                stdout=out,
                stderr=subprocess.STDOUT,
                shell=shell,
                executable="/bin/bash" if shell else None,
                start_new_session=True,
            )
            began = time.time()
            while True:
                code_ = self.child.poll()
                if code_ is not None:
                    break
                if timeout and time.time() - began > timeout:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    try:
                        self.child.wait(300)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid, signal.SIGKILL)
                        self.child.wait()
                    code_ = -9
                    break
                time.sleep(5)
        self.child = None
        return code_

    @staticmethod
    def ready(path: str) -> bool:
        p = Path(path)
        if p.is_dir():
            return (p / "VERIFIED").exists() or (p / "READY").exists()
        return p.exists()

    def upload(self, local: Path, remote: str, code: Path) -> None:
        script = (
            "import os,sys\nfrom huggingface_hub import HfApi\napi=HfApi()\n"
            f"api.create_repo({HF_RESULTS_REPO!r}, repo_type='dataset', private=True, exist_ok=True)\n"
            f"api.upload_file(path_or_fileobj={str(local)!r}, path_in_repo={remote!r}, repo_id={HF_RESULTS_REPO!r}, "
            "repo_type='dataset', commit_message='d25 vega runner result')\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=self.env(code),
                timeout=300,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.log(f"HF upload of {remote} failed: {result.stderr[-300:]}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"HF upload of {remote} failed: {exc}")

    @staticmethod
    def item_name(item: Path) -> str:
        try:
            return str(json.loads(item.read_text()).get("name", item.stem))
        except Exception:  # noqa: BLE001
            return item.stem

    def maybe_upgrade(self) -> None:
        """Re-exec the runner if ``current`` carries a different runner.py that imports cleanly."""
        try:
            new = self.code_dir() / "d25" / "vega" / "train" / "runner.py"
            if not new.exists() or new.read_bytes() == Path(__file__).read_bytes():
                return
            check = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import d25.vega.train.runner, d25.vega.train.train",
                ],
                env=self.env(self.code_dir()),
                cwd=str(self.code_dir()),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if check.returncode != 0:
                self.log(
                    f"not upgrading: new runner fails to import: {check.stderr[-300:]}"
                )
                return
            self.log(f"upgrading runner to {self.code_dir().name} (re-exec)")
            env = self.env(CURRENT)
            os.chdir(str(CURRENT))
            os.execve(
                sys.executable,
                [
                    sys.executable,
                    "-m",
                    "d25.vega.train.runner",
                    "--node",
                    self.node,
                    "--gpus",
                    str(self.gpus),
                ],
                env,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"upgrade check failed: {exc}")

    def run_tool(self, spec: dict, state_dir: Path) -> tuple[str, str]:
        """``kind: tool`` items: run shell steps in order (markers make them restartable)."""
        code = self.code_dir()
        for i, step in enumerate(spec.get("steps", [])):
            marker = state_dir / f"step-{i}.done"
            if marker.exists():
                continue
            self.status["phase"] = f"tool step {i}"
            rc = self.run(
                step["run"],
                code,
                state_dir / f"step-{i}.log",
                timeout=float(step.get("timeout_h", 4)) * 3600,
            )
            if rc != 0:
                return (
                    "failed",
                    f"tool step {i} exited {rc} (see {state_dir}/step-{i}.log)",
                )
            marker.write_text(now() + "\n")
        return "done", ""

    def parity(self, name: str, export: Path, dev: str | None, state_dir: Path) -> None:
        """Export reload parity (1 GPU) on the first export of an arm; result next to the export."""
        marker = state_dir / f"parity-{export.name}"
        if (
            marker.with_suffix(".done").exists()
            or marker.with_suffix(".failed").exists()
            or not dev
        ):
            return
        if not (export / "parity_rows.jsonl").exists():
            return
        code = self.code_dir()
        self.status["phase"] = f"parity {export.name}"
        out = export.parent / f"{export.name}.parity.json"
        rc = self.run(
            [
                sys.executable,
                "-m",
                "d25.vega.train.export",
                "parity",
                "--ckpt",
                str(export),
                "--dev",
                dev,
                "--rows",
                "32",
                "--out",
                str(out),
                "--engine",
            ],
            code,
            export.parent / "logs" / f"parity-{export.name}.log",
            timeout=3600,
        )
        summary = {}
        if rc == 0 and out.exists():
            data = json.loads(out.read_text())
            summary = {k: v for k, v in data.items() if isinstance(v, dict)}
            marker.with_suffix(".done").write_text(now() + "\n")
        else:
            marker.with_suffix(".failed").write_text(f"{now()} exit {rc}\n")
        self.event(arm=name, kind="parity", step=export.name, exit=rc, **summary)

    # -- arm -----------------------------------------------------------------------------------
    def run_item(self, item: Path) -> tuple[str, str]:
        try:
            spec = {**DEFAULTS, **json.loads(item.read_text())}
            name = spec["name"]
        except Exception as exc:  # noqa: BLE001
            return "failed", f"unreadable spec: {exc}"
        if spec.get("kind") == "tool":
            if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,62}", name):
                return "failed", f"invalid item name {name!r}"
            state_dir = self.queue / "state" / name
            state_dir.mkdir(parents=True, exist_ok=True)
            self.status.update(item=name, phase="tool")
            return self.run_tool(spec, state_dir)
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,62}", name):
            return "failed", f"invalid arm name {name!r}"
        state_dir = self.queue / "state" / name
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "state.json"
        state = (
            json.loads(state_file.read_text())
            if state_file.exists()
            else {"attempts": 0, "crashes": 0}
        )

        def save() -> None:
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            os.replace(tmp, state_file)

        self.status.update(item=name, phase="waiting-for-data")
        out = ROOT / "ckpt" / name
        logs = out / "logs"
        if not state.get("started"):
            state["started"] = now()
            save()
            self.event(arm=name, kind="arm_started", spec=item.name)
        # data
        if not state.get("train"):
            candidates = (
                spec["train"] if isinstance(spec["train"], list) else [spec["train"]]
            )
            deadline = time.time() + float(spec["wait_timeout_h"]) * 3600
            while True:
                chosen = next((c for c in candidates if self.ready(c)), None)
                waits_ok = all(Path(p).exists() for p in spec["wait_for"])
                if chosen and waits_ok:
                    break
                if time.time() > deadline:
                    return (
                        "failed",
                        f"data not ready after {spec['wait_timeout_h']} h: {candidates} {spec['wait_for']}",
                    )
                if STOPPING["flag"]:
                    raise SystemExit(143)
                self.status["phase"] = f"waiting-for-data {candidates[0]}"
                time.sleep(POLL_DATA_S)
            state["train"] = chosen
            dev = spec.get("dev")
            if isinstance(dev, list):
                dev = next((d for d in dev if Path(d).exists()), None)
            if (
                dev is None
                and Path(chosen).is_dir()
                and (Path(chosen) / "dev.jsonl.gz").exists()
            ):
                dev = str(Path(chosen) / "dev.jsonl.gz")
            state["dev"] = dev
            save()
            self.event(arm=name, kind="data_ready", train=chosen, dev=dev)
        # code (pinned per arm so a resumed run uses the same trainer)
        if not state.get("code") or not Path(state["code"]).exists():
            state["code"] = str(self.code_dir())
            save()
        code = Path(state["code"])
        # prepare steps
        for i, step in enumerate(spec["prepare"]):
            marker = state_dir / f"prepare-{i}.done"
            if marker.exists() or (
                step.get("creates") and Path(step["creates"]).exists()
            ):
                continue
            self.status["phase"] = f"prepare {i}"
            rc = self.run(
                step["run"],
                code,
                logs / f"prepare-{i}.log",
                timeout=float(step.get("timeout_h", 4)) * 3600,
            )
            if rc != 0:
                return (
                    "failed",
                    f"prepare step {i} exited {rc} (see {logs}/prepare-{i}.log)",
                )
            marker.write_text(now() + "\n")
        # train / evaluate loop
        init = spec["init"]
        if init == "base":
            init = os.environ.get("D25_BASE_DIR", "")
            if not init:
                return "failed", "init 'base' but D25_BASE_DIR is not set on this node"
        init_kind = spec["init_kind"] or ("base" if spec["init"] == "base" else "warm")
        args = [
            "--run",
            name,
            "--train",
            state["train"],
            "--output",
            str(out),
            "--init",
            init,
            "--init-kind",
            init_kind,
            "--attention-mode",
            spec["attention_mode"],
            "--lr",
            str(spec["lr"]),
            "--readout-lr",
            str(spec["readout_lr"] if spec["readout_lr"] is not None else spec["lr"]),
            "--warmup-ratio",
            str(spec["warmup_ratio"]),
            "--schedule",
            spec["schedule"],
            "--min-lr-ratio",
            str(spec["min_lr_ratio"]),
            "--epochs",
            str(spec["epochs"]),
            "--seed",
            str(spec["seed"]),
            "--rows-per-update",
            str(spec["rows_per_update"]),
            "--max-length",
            str(spec["max_length"]),
            "--token-budget",
            str(spec["token_budget"]),
            "--reshard-after-forward",
            "off",
            "--ac",
            "full",
            "--save-every",
            str(spec["save_every"]),
            "--keep-dcp",
            str(spec["keep_dcp"]),
            "--export-fractions",
            ",".join(str(f) for f in spec["export_fractions"]),
            "--pause-after-export",
            "--dev-every",
            "200",
            "--dev-rows",
            "1000",
        ]
        if state.get("dev"):
            args += ["--dev", state["dev"]]
        if spec["teacher"]:
            args += [
                "--teacher",
                spec["teacher"],
                "--teacher-weight",
                str(spec["teacher_weight"]),
            ]
        if spec["brier_weight"]:
            args += ["--brier-weight", str(spec["brier_weight"])]
        args += [str(a) for a in spec["trainer_args"]]
        while not (state_dir / "train.done").exists():
            self.evaluate_exports(
                name, spec, out, state_dir, final_only=False, dev=state.get("dev")
            )
            self.maybe_upgrade()
            self.status["phase"] = "training"
            cmd = [
                sys.executable,
                "-m",
                "d25.vega.train.launch",
                "--nproc",
                str(self.gpus),
                "--log-dir",
                str(logs),
                "d25.vega.train.train",
                *args,
            ]
            self.log(
                f"{name}: launching trainer (attempt {state['attempts'] + 1}, code {code.name})"
            )
            rc = self.run(cmd, code, logs / "launcher.log")
            self.log(f"{name}: trainer exited {rc}")
            if STOPPING["flag"]:
                raise SystemExit(143)
            if rc == 0:
                (state_dir / "train.done").write_text(now() + "\n")
                self.event(arm=name, kind="train_finished")
                break
            if rc == 10:
                self.event(arm=name, kind="train_paused_for_eval")
                continue
            if rc == 3:
                return (
                    "failed",
                    "sanity: train loss after 200 updates not below the loss at update 10 (events.jsonl)",
                )
            if rc == 4:
                return "failed", "non-finite loss or gradient (events.jsonl)"
            state["attempts"] += 1
            save()
            if state["attempts"] >= int(spec["max_attempts"]):
                return (
                    "failed",
                    f"trainer failed {state['attempts']} times (last exit {rc}); see {logs}",
                )
            self.event(
                arm=name, kind="train_retry", exit=rc, attempts=state["attempts"]
            )
            time.sleep(60)
        self.evaluate_exports(
            name, spec, out, state_dir, final_only=False, dev=state.get("dev")
        )
        return "done", ""

    def evaluate_exports(
        self,
        name: str,
        spec: dict,
        out: Path,
        state_dir: Path,
        final_only: bool,
        dev: str | None = None,
    ) -> None:
        plan_file = out / "plan.json"
        total = (
            json.loads(plan_file.read_text())["total_steps"]
            if plan_file.exists()
            else None
        )
        exports = sorted(
            p
            for p in out.glob("step-*")
            if p.is_dir() and (p / "decision_config.json").exists()
        )
        if exports and spec.get("parity", True):
            self.parity(name, exports[0], dev, state_dir)
        for export in exports:
            step = int(export.name.split("-")[1])
            final = total is not None and step >= total
            whats = spec["evals"]["final" if final else "intermediate"]
            for what in whats:
                marker = state_dir / f"eval-{export.name}-{what}"
                if (
                    marker.with_suffix(".done").exists()
                    or marker.with_suffix(".failed").exists()
                ):
                    continue
                self.eval_one(name, export, what, marker, spec)

    def eval_one(
        self, name: str, export: Path, what: str, marker: Path, spec: dict
    ) -> None:
        code = self.code_dir()
        result_dir = RESULTS / name / export.name
        result_dir.mkdir(parents=True, exist_ok=True)
        self.status["phase"] = f"eval {export.name} {what}"
        if not (code / "d25" / "vega" / "eval" / "ckpt_eval.py").exists():
            marker.with_suffix(".failed").write_text(
                f"{now()} ckpt_eval missing in {code}\n"
            )
            self.event(
                arm=name,
                kind="eval_failed",
                step=export.name,
                what=what,
                reason="ckpt_eval missing",
            )
            return
        began = time.time()
        timeout = float(spec["eval_timeout_h"].get(what, 6)) * 3600
        cmd = [
            sys.executable,
            "-m",
            "d25.vega.eval.ckpt_eval",
            "--ckpt",
            str(export),
            "--out",
            str(result_dir),
            "--gpus",
            str(self.gpus),
            "--what",
            what,
        ]
        rc = self.run(cmd, code, result_dir / f"ckpt_eval-{what}.log", timeout=timeout)
        wall = time.time() - began
        result = result_dir / "result.json"
        if rc != 0 or not result.exists():
            marker.with_suffix(".failed").write_text(
                f"{now()} exit {rc}; result.json {'present' if result.exists() else 'missing'}\n"
            )
            self.event(
                arm=name,
                kind="eval_failed",
                step=export.name,
                what=what,
                exit=rc,
                wall_s=round(wall),
            )
            return
        index = RESULTS / "index" / f"{name}-{export.name}.json"
        shutil.copyfile(result, index)
        marker.with_suffix(".done").write_text(now() + "\n")
        summary = {}
        try:
            data = json.loads(result.read_text())
            summary = {
                "public": (data.get("public") or {}).get("index"),
                "proxy": {
                    k: (data.get("proxy") or {}).get(k) for k in ("S_proxy", "O_proxy")
                },
                "gate": data.get("gate"),
            }
        except Exception:  # noqa: BLE001
            pass
        self.event(
            arm=name,
            kind="eval_done",
            step=export.name,
            what=what,
            result=str(index),
            wall_s=round(wall),
            **summary,
        )
        self.upload(index, f"{self.node}/{index.name}", code)
        self.upload(RESULTS / "events.jsonl", f"{self.node}/events.jsonl", code)

    # -- main loop -----------------------------------------------------------------------------
    def loop(self) -> None:
        self.log(f"runner up (gpus {self.gpus}, code {self.code_dir()})")
        while not STOPPING["flag"]:
            item = self.next_item()
            if item is None:
                self.status.update(item=None, phase="idle")
                self.maybe_upgrade()
                time.sleep(60)
                continue
            try:
                outcome, reason = self.run_item(item)
            except SystemExit:
                raise
            except Exception:  # noqa: BLE001
                detail = traceback.format_exc()
                self.log(f"{item.name}: runner error\n{detail}")
                state_file = self.queue / "state" / self.item_name(item) / "state.json"
                try:
                    state = (
                        json.loads(state_file.read_text())
                        if state_file.exists()
                        else {}
                    )
                except ValueError:
                    state = {}
                state["crashes"] = state.get("crashes", 0) + 1
                state_file.parent.mkdir(parents=True, exist_ok=True)
                state_file.write_text(json.dumps(state, indent=2))
                if state["crashes"] >= 3:
                    outcome, reason = (
                        "failed",
                        f"runner error x3: {detail.splitlines()[-1]}",
                    )
                else:
                    time.sleep(120)
                    continue
            name = self.item_name(item)
            self.event(arm=name, kind=f"arm_{outcome}", reason=reason)
            self.finish(item, outcome, reason)
            self.upload(
                RESULTS / "events.jsonl", f"{self.node}/events.jsonl", self.code_dir()
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--node", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    options = parser.parse_args()
    runner = Runner(options.node, options.gpus)

    def stop(signum, _frame):
        STOPPING["flag"] = True
        child = runner.child
        runner.log(
            f"signal {signum}: stopping (child {'running' if child else 'none'})"
        )
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        runner.loop()
    except SystemExit as exc:
        runner.log(f"exiting ({exc.code})")
        return int(exc.code or 0)
    return 143 if STOPPING["flag"] else 0


if __name__ == "__main__":
    sys.exit(main())
