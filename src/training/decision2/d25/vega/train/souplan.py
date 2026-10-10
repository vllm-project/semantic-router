"""Soup plans: FP32 weight averages of exports from any training node, built and evaluated in order.

Run as a runner ``tool`` item step on the node that should hold the soups (it uses that node's GPUs
for the evals):

    python -m d25.vega.train.souplan --plan /data/d25/vega/xfer/plans/<name>.json --node 04

Plan JSON::

    {"name": "w1-soups",
     "inputs": {"t2-final": {"local": "/data/d25/vega/ckpt/w1-t2-nc/step-005135"},
                "g2-final": {"node": "06", "path": "ckpt/w1-g2-nc/step-005135"}},
     "soups": [{"name": "soup-b-t2-g2-final", "members": ["t2-final", "g2-final"], "weights": null}],
     "proxy": true, "full_top": 2, "rank_by": "O_proxy", "wait_h": 6, "gpus": 8}

Remote members are copied with ``d25.vega.train.xfer`` into ``/data/d25/vega/imports/<node>/...``
(already done if the node's xfer server pulled them in the background); a soup whose members are
still missing when ``wait_h`` (from the plan start) runs out is skipped, the others continue. Every
soup gets a ``proxy`` eval; the best ``full_top`` by ``rank_by`` then get ``full``. Soups land in
``/data/d25/vega/ckpt/soups/<soup>/`` (export format), results in ``results/<soup>/soup/result.json``,
``results/index/<soup>-soup.json`` and ``results/events.jsonl``; the plan summary is
``results/soups/<plan>.json``. Everything finished is skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("D25_VEGA_ROOT", "/data/d25/vega"))
RESULTS = ROOT / "results"
SOUPS = ROOT / "ckpt" / "soups"
HF_RESULTS_REPO = "vllm-sr/d25-vega-results"
EVAL_TIMEOUT_H = {"proxy": 3, "full": 10}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{now()} [souplan] {message}", flush=True)


class Plan:
    def __init__(self, path: Path, node: str, gpus: int | None):
        self.spec = json.loads(path.read_text())
        self.name = self.spec["name"]
        self.node = node
        self.gpus = int(gpus or self.spec.get("gpus", 8))
        self.deadline = time.time() + float(self.spec.get("wait_h", 6)) * 3600
        self.summary_path = RESULTS / "soups" / f"{self.name}.json"
        self.summary = (
            json.loads(self.summary_path.read_text())
            if self.summary_path.exists()
            else {}
        )
        self.summary.update({"plan": self.name, "node": node, "spec": self.spec})
        self.summary.setdefault("soups", {})
        self.code = Path(
            os.environ.get("D25_CODE_DIR") or (ROOT / "src" / "current")
        ).resolve()

    def save(self) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary["updated"] = now()
        tmp = self.summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.summary, indent=2))
        os.replace(tmp, self.summary_path)

    def event(self, **fields) -> None:
        record = {"time": now(), "node": self.node, "plan": self.name, **fields}
        RESULTS.mkdir(parents=True, exist_ok=True)
        with (RESULTS / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        log(f"event {json.dumps(fields)[:300]}")


def obtain(spec: dict, deadline: float) -> Path | None:
    """Local export dir, or a verified copy of a remote one (pulled if needed); None if unavailable."""
    from d25.vega.train import xfer

    if "local" in spec:
        path = Path(spec["local"])
        while not (path / "decision_config.json").exists():
            if time.time() >= deadline:
                return None
            time.sleep(60)
        return path
    dest = (
        Path(spec["dest"])
        if spec.get("dest")
        else xfer.default_dest(spec["node"], spec["path"])
    )
    if xfer.is_verified(dest):
        return dest
    remaining_h = max(0.0, (deadline - time.time()) / 3600)
    try:
        xfer.pull(spec["node"], spec["path"], dest, wait_h=remaining_h)
    except xfer.NotReady as exc:
        log(f"{spec['node']}:{spec['path']} unavailable: {exc}")
        return None
    return dest if xfer.is_verified(dest) else None


def run_eval(plan: Plan, soup: str, export: Path, what: str) -> dict | None:
    """ckpt_eval for one soup (resumable on ws-measure's side); returns result.json or None."""
    out = RESULTS / soup / "soup"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists():
        current = json.loads(result_path.read_text())
        if current.get("what") == what or (
            what == "proxy" and current.get("what") == "full"
        ):
            return current
    cmd = [
        sys.executable,
        "-m",
        "d25.vega.eval.ckpt_eval",
        "--ckpt",
        str(export),
        "--out",
        str(out),
        "--gpus",
        str(plan.gpus),
        "--what",
        what,
        "--arm",
        soup,
        "--step",
        "soup",
    ]
    began = time.time()
    with (out / f"ckpt_eval-{what}.log").open("ab") as handle:
        try:
            rc = subprocess.run(
                cmd,
                cwd=str(plan.code),
                env=dict(os.environ),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=EVAL_TIMEOUT_H[what] * 3600,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    wall = round(time.time() - began)
    if rc != 0 or not result_path.exists():
        plan.event(
            arm=soup, kind="eval_failed", step="soup", what=what, exit=rc, wall_s=wall
        )
        return None
    result = json.loads(result_path.read_text())
    index = RESULTS / "index" / f"{soup}-soup.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result_path, index)
    plan.event(
        arm=soup,
        kind="eval_done",
        step="soup",
        what=what,
        wall_s=wall,
        result=str(index),
        public=(result.get("public") or {}).get("index"),
        proxy={k: (result.get("proxy") or {}).get(k) for k in ("S_proxy", "O_proxy")},
        gate=result.get("gate"),
    )
    upload(plan, index, f"{plan.node}/{index.name}")
    upload(plan, RESULTS / "events.jsonl", f"{plan.node}/events.jsonl")
    return result


def upload(plan: Plan, local: Path, remote: str) -> None:
    """Best effort mirror to the private HF results dataset; never raises (token may be rotated)."""
    script = (
        "from huggingface_hub import HfApi\napi=HfApi()\n"
        f"api.create_repo({HF_RESULTS_REPO!r}, repo_type='dataset', private=True, exist_ok=True)\n"
        f"api.upload_file(path_or_fileobj={str(local)!r}, path_in_repo={remote!r}, repo_id={HF_RESULTS_REPO!r}, "
        "repo_type='dataset', commit_message='d25 vega soup result')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ),
            timeout=300,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log(f"HF upload of {remote} failed (exit {result.returncode})")
    except Exception as exc:  # noqa: BLE001
        log(f"HF upload of {remote} failed: {type(exc).__name__}")


def build(plan: Plan, soup: dict) -> Path | None:
    from d25.vega.train.soup import build_soup

    out = SOUPS / soup["name"]
    if (out / "decision_config.json").exists():
        return out
    members = []
    for key in soup["members"]:
        path = obtain(plan.spec["inputs"][key], plan.deadline)
        if path is None:
            plan.event(
                arm=soup["name"],
                kind="soup_skipped",
                reason=f"member {key} unavailable",
            )
            return None
        members.append(path)
    log(f"building {soup['name']} from {[str(m) for m in members]}")
    summary = build_soup(members, soup.get("weights"), out, name=soup["name"])
    plan.event(
        arm=soup["name"],
        kind="soup_built",
        members=soup["members"],
        weights=summary["weights"],
        seconds=summary["seconds"],
    )
    return out


def score(result: dict | None, key: str) -> float | None:
    if not result:
        return None
    value = (result.get("proxy") or {}).get(key)
    if value is None:
        value = (result.get("gate") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--gpus", type=int)
    args = parser.parse_args()
    plan = Plan(Path(args.plan), args.node, args.gpus)
    SOUPS.mkdir(parents=True, exist_ok=True)
    plan.event(kind="soup_plan_started", soups=[s["name"] for s in plan.spec["soups"]])
    exports: dict[str, Path] = {}
    for soup in plan.spec["soups"]:
        entry = plan.summary["soups"].setdefault(
            soup["name"], {"members": soup["members"]}
        )
        try:
            out = build(plan, soup)
        except Exception as exc:  # noqa: BLE001
            log(traceback.format_exc()[-1500:])
            entry["error"] = f"build: {type(exc).__name__}: {exc}"
            plan.event(
                arm=soup["name"], kind="soup_failed", reason=entry["error"][:300]
            )
            plan.save()
            continue
        if out is None:
            entry["state"] = "skipped (member unavailable)"
            plan.save()
            continue
        exports[soup["name"]] = out
        entry.update(state="built", path=str(out))
        if plan.spec.get("proxy", True):
            result = run_eval(plan, soup["name"], out, "proxy")
            if result and result.get("proxy"):
                entry["proxy"] = {
                    k: result["proxy"].get(k)
                    for k in ("S_proxy", "S_proxy_clean", "O_proxy")
                }
        plan.save()
    rank_by = plan.spec.get("rank_by", "O_proxy")
    results = {}
    for name in exports:
        path = RESULTS / name / "soup" / "result.json"
        results[name] = json.loads(path.read_text()) if path.exists() else None
    ranked = sorted(
        (n for n in exports if score(results[n], rank_by) is not None),
        key=lambda n: -score(results[n], rank_by),
    )
    plan.summary["ranking"] = {"by": rank_by, "order": ranked}
    plan.save()
    for name in ranked[: int(plan.spec.get("full_top", 2))]:
        result = run_eval(plan, name, exports[name], "full")
        if result:
            plan.summary["soups"][name]["full"] = {
                "public": (result.get("public") or {}).get("index"),
                "gate": result.get("gate"),
            }
        plan.save()
    plan.event(kind="soup_plan_finished", ranking=ranked)
    return 0 if exports else 1


if __name__ == "__main__":
    sys.exit(main())
