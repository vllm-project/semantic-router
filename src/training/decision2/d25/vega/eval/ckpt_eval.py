"""Single evaluation entry point for training checkpoints (called by the training nodes' runner chains).

    python -m d25.vega.eval.ckpt_eval --ckpt <export dir> --out <dir> --gpus 8 --what proxy|full [--arm A --step S]

- ``proxy``: S-proxy + O-proxy (frozen pv1) with the shared code-readout engine (``run_rows``, one worker
  process per GPU), scored with the kit's metric code (``proxy.score``).
- ``full``: the 37 public-index benchmarks of suite 0.3 (``run_suite --index-only``, kit-scored) + the proxies,
  then the calibrated gate estimate (``proxy.gate``; ``gate`` is null until ``proxy/calibration_pv1.json`` exists).

Writes ``<out>/result.json``:
    {arm, step, ckpt, created, public: {index, areas{}, per_benchmark{}} | null,
     proxy: {S_proxy, S_proxy_clean, O_proxy, O_families, per_task{}}, gate: {...} | null, wall_s}
Resumable: finished parts are skipped (``run_rows``/``run_suite`` resume their shards). Runs entirely in the
calling pod from node disk (code, kit, suite, frozen proxy): no network. The checkpoint's own
``decision_config.json`` sets prompt, attention, limits and temperature; a directory without it is the
zero-shot mode of the engine (stock base).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KIT = "/data/d25/shared/decision-index-kit"
SUITE = "/data/d25/shared/index-suite-0.3"
PROXY = "/data/d25/vega/proxy/final/pv1"
CALIBRATION = Path(__file__).parent / "proxy" / "calibration_pv2.json"
if not CALIBRATION.exists():
    CALIBRATION = CALIBRATION.with_name("calibration_pv1.json")


def _env():
    env = dict(os.environ)
    paths = [p for p in env.get("PYTHONPATH", "").split(":") if p]
    for p in (KIT, str(Path(__file__).resolve().parents[3])):
        if p not in paths:
            paths.append(p)
    env["PYTHONPATH"] = ":".join(paths)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _devices(gpus: int) -> str:
    return "0" if gpus <= 1 else f"0-{gpus - 1}"


def _engine_args(a) -> list[str]:
    out = []
    for flag in (
        "prompt",
        "attention_mode",
        "max_length",
        "readout_dtype",
        "max_batch_tokens",
    ):
        v = getattr(a, flag)
        if v is not None:
            out += ["--" + flag.replace("_", "-"), str(v)]
    return out


def _run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n# {datetime.datetime.now().isoformat()} {' '.join(cmd)}\n")
        f.flush()
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=_env())
    if code:
        raise SystemExit(
            f"{cmd[2]} failed with exit code {code}; see {log} (re-run to resume)"
        )


def proxy_questions(proxy: Path, path: Path) -> Path:
    """One row per question of every pv1 request, ids ``<run_id>\\t<question key>``."""
    if path.exists():
        return path
    from d25.vega.eval.proxy.common import read_jsonl, write_jsonl

    rows = []
    for f in sorted((proxy / "s").glob("*.jsonl.gz")) + sorted(
        (proxy / "o").glob("*.jsonl.gz")
    ):
        for r in read_jsonl(f):
            for k, q in r["questions"].items():
                rows.append(
                    {
                        "id": f"{r['_evaluation']['run_id']}\t{k}",
                        "state": r["state"],
                        "question": q,
                    }
                )
    write_jsonl(path, rows)
    return path


def proxy_part(a, out: Path) -> dict:
    if KIT not in sys.path:
        sys.path.insert(0, KIT)
    from d25.vega.common import decision_format as df
    from d25.vega.eval.proxy.common import read_jsonl
    from d25.vega.eval.proxy.score import score_all

    proxy, pdir = Path(a.proxy), out / "proxy"
    scores_path = pdir / "scores.json"
    if scores_path.exists() and not a.rescore:
        return json.loads(scores_path.read_text())
    qfile = proxy_questions(proxy, pdir / "questions.jsonl.gz")
    rows_out = pdir / "rows"
    if not (rows_out / "summary.json").exists():
        _run(
            [
                sys.executable,
                "-m",
                "d25.vega.eval.run_rows",
                "run",
                "--ckpt",
                a.ckpt,
                "--rows",
                str(qfile),
                "--out",
                str(rows_out),
                "--devices",
                _devices(a.gpus),
            ]
            + _engine_args(a),
            pdir / "run_rows.log",
        )
    probs = {}
    for r in read_jsonl(rows_out / "probs.jsonl"):
        probs[r["id"]] = r
    results = {}
    for f in sorted((proxy / "s").glob("*.jsonl.gz")) + sorted(
        (proxy / "o").glob("*.jsonl.gz")
    ):
        for row in read_jsonl(f):
            rid = row["_evaluation"]["run_id"]
            answers, status = {}, "ok"
            for k, q in row["questions"].items():
                p = probs.get(f"{rid}\t{k}")
                if p is None or p["status"] != "ok":
                    status = "unsupported" if p is not None else "error"
                    break
                answers[k] = df.to_answer(q, p["probs"])
            results[rid] = {
                "run_id": rid,
                "catalog_id": row["_evaluation"]["catalog_id"],
                "status": status,
                "total_wall_ms": 0.0,
                **({"response": {"answers": answers}} if status == "ok" else {}),
            }
    sc = score_all(proxy, results)
    sc["ckpt"] = a.ckpt
    scores_path.write_text(json.dumps(sc, indent=1))
    return sc


def public_part(a, out: Path) -> dict:
    pub = out / "public"
    summary = pub / "summary.json"
    if not (summary.exists() and json.loads(summary.read_text()).get("complete")):
        _run(
            [
                sys.executable,
                "-m",
                "d25.vega.eval.run_suite",
                "run",
                "--ckpt",
                a.ckpt,
                "--kit",
                a.kit,
                "--suite-dir",
                a.suite,
                "--out",
                str(pub),
                "--devices",
                _devices(a.gpus),
                "--index-only",
                "--plan-root",
                str(out / "plans"),
            ]
            + _engine_args(a),
            out / "run_suite.log",
        )
    return json.loads(summary.read_text())


def summarize_public(s: dict) -> dict:
    return {
        "index": s.get("decision_index"),
        "raw_index": s.get("raw_index"),
        "complete": s.get("complete"),
        "areas": {
            k: round(100 * v["skill"], 2) for k, v in (s.get("areas") or {}).items()
        },
        "per_benchmark": {
            k: {
                "name": v.get("dataset"),
                "skill": round(100 * v["index_skill"], 2),
                "raw": v.get("index_raw"),
                "coverage": v.get("coverage"),
            }
            for k, v in (s.get("benchmarks") or {}).items()
            if v.get("in_index")
        },
    }


def summarize_proxy(sc: dict) -> dict:
    per = {f"S:{v['name']}": round(100 * v["skill"], 2) for v in sc["s"].values()}
    per.update({f"O:{t}": round(100 * v["skill"], 2) for t, v in sc["o"].items()})
    return {
        "S_proxy": round(sc["S_proxy"], 3),
        "S_proxy_clean": round(sc["S_proxy_clean"], 3),
        "O_proxy": round(sc["O_proxy"], 3),
        "O_families": {k: round(v, 2) for k, v in sc["O_families"].items()},
        "coverage": sc["coverage"],
        "per_task": per,
    }


def gate_part(
    public_index, proxy: dict, calibration: Path, public_bench: dict | None = None
) -> dict | None:
    if public_index is None or not calibration.exists():
        return None
    from d25.vega.eval.proxy.gate import gate

    g = gate(
        public_index,
        proxy["S_proxy"],
        proxy["O_proxy"],
        calibration,
        proxy.get("S_proxy_clean"),
        public_bench,
    )
    return {
        "S_hat": g["S_hat"],
        "O_hat": g["O_hat"],
        "Full_hat": g["Full_hat"],
        "margin": g["margin"],
        "lower_bound": g["Full_lower_90"],
        "board_rank": g["board"],
        "board_rank_at_lower_bound": g["board_at_lower_bound"],
        "decisions": g["decisions"],
        "models": g["models"],
        "sensitivity_clean": g.get("sensitivity_clean"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--what", choices=("proxy", "full"), default="proxy")
    ap.add_argument("--arm")
    ap.add_argument("--step")
    ap.add_argument(
        "--public",
        type=float,
        help="local public index, if measured elsewhere (enables the gate for --what proxy)",
    )
    ap.add_argument("--kit", default=KIT)
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--proxy", default=PROXY)
    ap.add_argument("--calibration", default=str(CALIBRATION))
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="re-score stored outputs (e.g. to backfill the gate)",
    )
    for flag in ("prompt", "attention-mode", "readout-dtype"):
        ap.add_argument("--" + flag)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--max-batch-tokens", type=int)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.ckpt)
    cfg = (
        json.loads((ck / "decision_config.json").read_text())
        if (ck / "decision_config.json").exists()
        else {}
    )
    prov = cfg.get("provenance") or {}
    result_path = out / "result.json"
    prev = json.loads(result_path.read_text()) if result_path.exists() else {}
    result = {
        "arm": a.arm or prov.get("run") or ck.parent.name,
        "step": a.step or prov.get("step") or ck.name,
        "ckpt": str(ck.resolve()),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "what": a.what,
        "public": prev.get("public"),
        "proxy": prev.get("proxy"),
        "gate": prev.get("gate"),
        "wall_s": prev.get("wall_s", 0.0),
    }

    def save():
        tmp = result_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, result_path)

    result["proxy"] = summarize_proxy(proxy_part(a, out))
    save()
    if a.what == "full":
        result["public"] = summarize_public(public_part(a, out))
        save()
    public_index = (
        (result["public"] or {}).get("index") if result["public"] else a.public
    )
    result["gate"] = gate_part(
        public_index,
        result["proxy"],
        Path(a.calibration),
        (result.get("public") or {}).get("per_benchmark"),
    )
    result["wall_s"] = round(result["wall_s"] + time.time() - t0, 1)
    save()
    print(
        json.dumps(
            {k: result[k] for k in ("arm", "step", "gate", "wall_s")}
            | {
                "S_proxy": result["proxy"]["S_proxy"],
                "O_proxy": result["proxy"]["O_proxy"],
                "public_index": public_index,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
