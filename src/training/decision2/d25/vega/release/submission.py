"""Decision Index 0.3 submission kit for a Decision 2.5 package (prepare only: nothing is made public here).

The public run is the kit's own runner with the package's engine (``decision25_engine:Decision25Engine``), one
request at a time, split over N processes (one per GPU) by deterministic shards:

    python -m d25.vega.release.submission shard --kit KIT --suite-dir SUITE --n 8 --out SHARDS
    # per GPU i (see submission_run.sh): PYTHONPATH=PKG:KIT python -m decision_index run \
    #     --engine decision25_engine:Decision25Engine --option model=PKG --option device=cuda:i \
    #     --rows SHARDS/shard-0i.jsonl.gz --out SHARDS/run-0i --compact          (resumable)
    python -m d25.vega.release.submission merge --kit KIT --suite-dir SUITE --shards SHARDS --name NAME --out RUN
    python -m d25.vega.release.submission stage --kit KIT --suite-dir SUITE --run RUN --out STAGE \
        --manifest PKG/MODEL_MANIFEST.json --repo vllm-sr/Decision-2.5-Vega-27B --revision SHA --hardware "..."
    python -m d25.vega.release.submission upload --stage STAGE --dataset vllm-sr/decision-2.5-decision-index --name NAME
    python -m d25.vega.release.submission pr-text --stage STAGE --dataset vllm-sr/decision-2.5-decision-index \
        --name NAME [--latency latency.json] --out PR.md

``merge`` checks that every scoreable run id of the edition has exactly one result and scores with
``decision_index score --edition 0.3``. ``stage`` writes the six files the board asks for (``results.jsonl.gz``,
``benchmark-summary.json``, ``index.json``, ``scores.json`` with ``complete: true``, ``environment.json``,
``status.json``) plus ``RUN_NOTES.md`` and ``SHA256SUMS``, and re-scores ``results.jsonl.gz`` to prove the
scores reproduce. ``upload`` creates the dataset PRIVATE (the lead flips it) and refuses a public one.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EDITION = "0.3"
KIT_COMMIT = "9eb2dbe2a358004c8782c66e40a83ac07b953fec"
STAGED = (
    "benchmark-summary.json",
    "index.json",
    "scores.json",
    "environment.json",
    "status.json",
)
VOLATILE = ("generated_utc",)


def use_kit(kit: Path) -> None:
    if str(kit) not in sys.path:
        sys.path.insert(0, str(kit))


def suite_rows(kit: Path, suite_dir: Path) -> list[dict]:
    use_kit(kit)
    from decision_index.suite.io import Suite

    return list(Suite(suite_dir, EDITION).rows(apply_exclusions=True))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [
            json.loads(line) for line in stream if line.endswith("\n") and line.strip()
        ]


def shard(kit: Path, suite_dir: Path, n: int, out: Path) -> dict[str, Any]:
    """Longest-processing-time split by payload length, so every shard gets about the same work."""
    rows = suite_rows(kit, suite_dir)
    cost = [
        len(
            json.dumps(
                {"state": r["state"], "questions": r["questions"]}, ensure_ascii=False
            )
        )
        for r in rows
    ]
    order = sorted(
        range(len(rows)), key=lambda i: (-cost[i], rows[i]["_evaluation"]["run_id"])
    )
    heap = [(0, s) for s in range(n)]
    assign = [0] * len(rows)
    for i in order:
        load, s = heapq.heappop(heap)
        assign[i] = s
        heapq.heappush(heap, (load + cost[i], s))
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter(assign)
    for s in range(n):
        with gzip.open(
            out / f"shard-{s:02d}.jsonl.gz", "wt", encoding="utf-8"
        ) as stream:
            for row, a in zip(rows, assign):
                if a == s:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ids = sorted(r["_evaluation"]["run_id"] for r in rows)
    plan = {
        "edition": EDITION,
        "rows": len(rows),
        "shards": n,
        "per_shard": [counts[s] for s in range(n)],
        "run_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }
    (out / "shards.json").write_text(json.dumps(plan, indent=1) + "\n")
    return plan


def kit_score(
    kit: Path, suite_dir: Path, results: Path, engine: str, out: Path
) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "decision_index",
            "score",
            "--edition",
            EDITION,
            "--suite-dir",
            str(suite_dir),
            "--results",
            str(results),
            "--engine",
            engine,
            "--out",
            str(out),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(kit)},
        stdout=subprocess.DEVNULL,
    )
    return json.loads((out / "scores.json").read_text())


def merge(
    kit: Path, suite_dir: Path, shards: Path, name: str, out: Path
) -> dict[str, Any]:
    rows = suite_rows(kit, suite_dir)
    wanted = [r["_evaluation"]["run_id"] for r in rows]
    found: dict[str, dict] = {}
    duplicates, environments = [], []
    for run in sorted(
        p for p in shards.iterdir() if p.is_dir() and p.name.startswith("run-")
    ):
        for result in read_jsonl(run / "results.jsonl"):
            rid = result["run_id"]
            if rid in found and found[rid]["status"] != "error":
                duplicates.append(rid)
            if rid not in found or result["status"] != "error":
                found[rid] = result
        environments.append(
            {"shard": run.name, **json.loads((run / "environment.json").read_text())}
        )
    missing = [rid for rid in wanted if rid not in found]
    errors = [rid for rid in wanted if rid in found and found[rid]["status"] == "error"]
    extra = sorted(set(found) - set(wanted))
    if missing or errors or duplicates or extra:
        raise SystemExit(
            json.dumps(
                {
                    "missing": len(missing),
                    "errors": len(errors),
                    "duplicates": len(duplicates),
                    "extra": len(extra),
                    "examples": (missing + errors + duplicates + extra)[:5],
                }
            )
        )
    sources = {json.dumps(e.get("model_source"), sort_keys=True) for e in environments}
    if len(sources) != 1:
        raise SystemExit("shards ran different model sources")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as stream:
        for rid in wanted:
            stream.write(json.dumps(found[rid], ensure_ascii=False) + "\n")
    environment = {
        **{k: v for k, v in environments[0].items() if k != "shard"},
        "runner_processes": len(environments),
        "runners": [
            {
                k: e.get(k)
                for k in (
                    "shard",
                    "gpu",
                    "device",
                    "loaded_seconds",
                    "torch",
                    "hip",
                    "cuda",
                )
            }
            for e in environments
        ],
    }
    (out / "environment.json").write_text(
        json.dumps(environment, indent=1, default=str) + "\n"
    )
    counts = Counter(found[rid]["status"] for rid in wanted)
    status = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": "complete",
        "engine": name,
        "completed": len(wanted),
        "counts": dict(counts),
        "runner_processes": len(environments),
    }
    (out / "status.json").write_text(json.dumps(status) + "\n")
    scores = kit_score(kit, suite_dir, out / "results.jsonl", name, out)
    if scores.get("complete") is not True:
        raise SystemExit("the kit does not report the run as complete")
    return {
        "run": str(out),
        "requests": len(wanted),
        "counts": dict(counts),
        "decision_index": scores["decision_index"],
        "complete": scores["complete"],
    }


def comparable(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in VOLATILE}


def stage(
    kit: Path,
    suite_dir: Path,
    run: Path,
    out: Path,
    manifest: dict,
    repo: str,
    revision: str,
    hardware: str,
    settings: str,
) -> dict[str, Any]:
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    for name in STAGED:
        shutil.copyfile(run / name, out / name)
    with open(run / "results.jsonl", "rb") as src, gzip.open(
        out / "results.jsonl.gz", "wb", compresslevel=6
    ) as dst:
        shutil.copyfileobj(src, dst)
    scores = json.loads((out / "scores.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        again = kit_score(
            kit, suite_dir, out / "results.jsonl.gz", scores["engine"], Path(tmp)
        )
        same = {
            name: comparable(json.loads((Path(tmp) / name).read_text()))
            == comparable(json.loads((out / name).read_text()))
            for name in ("scores.json", "index.json", "benchmark-summary.json")
        }
    if not all(same.values()) or again.get("complete") is not True:
        raise SystemExit(
            f"re-scoring results.jsonl.gz does not reproduce the staged files: {same}"
        )
    counts = scores["counts"]
    notes = [
        f"# {manifest['model_name']}: Decision Index {EDITION} public run",
        "",
        f"- Model: [{repo}](https://huggingface.co/{repo}) at revision `{revision}` (weights identity "
        f"`{manifest['identity']['model_sha256']}` in `MODEL_MANIFEST.json`).",
        f"- Kit: apolinario/decision-index `{KIT_COMMIT[:7]}` (edition {EDITION}), unmodified runner and scorer.",
        f"- Engine: `{package_engine(manifest)[0]}` from the model repository (`PYTHONPATH=<download dir>`), one "
        "request per call, the request's questions in request order, 8 per forward pass.",
        f"- Inference settings: {settings}",
        f"- Hardware: {hardware}",
        f"- Requests: {scores['completed']:,} scoreable, {', '.join(f'{v:,} {k}' for k, v in sorted(counts.items()))}; "
        "`unsupported` = a question prompt over the checkpoint's input limit (nothing truncated).",
        f"- Public index {scores['decision_index']} (raw {scores['raw_index']}); `scores.json` says complete: "
        f"{str(scores['complete']).lower()}.",
        "- Re-scoring `results.jsonl.gz` with `decision_index score --edition 0.3` reproduces `scores.json`, `index.json`"
        " and `benchmark-summary.json` (apart from `generated_utc`).",
        "",
    ]
    (out / "RUN_NOTES.md").write_text("\n".join(notes))
    sums = [f"{sha256_file(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file()]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return {
        "stage": str(out),
        "decision_index": scores["decision_index"],
        "complete": scores["complete"],
        "counts": counts,
        "rescored_identical": same,
    }


def package_engine(manifest: dict) -> tuple[str, str]:
    """The kit engine and the server script of a package (d3-family packages name their runtime d3_*)."""
    files = (manifest.get("runtime") or {}).get("files_sha256") or {}
    if "d3_engine.py" in files:
        return "d3_engine:D3Engine", "d3_server.py"
    return "decision25_engine:Decision25Engine", "decision25_server.py"


def upload(
    stage_dir: Path, dataset: str, name: str, allow_public: bool = False
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(dataset, repo_type="dataset")
    if info.private is not True and not allow_public:
        raise SystemExit(f"{dataset} is public; refusing (the lead flips visibility)")
    subprocess.run(
        [
            "hf",
            "upload",
            dataset,
            str(stage_dir),
            f"runs/{name}",
            "--repo-type",
            "dataset",
            "--commit-message",
            f"Decision Index 0.3 public run {name}",
        ],
        check=True,
    )
    info = api.repo_info(dataset, repo_type="dataset")
    return {
        "dataset": dataset,
        "path": f"runs/{name}",
        "revision": info.sha,
        "private": info.private,
    }


def pr_text(
    stage_dir: Path, dataset: str, name: str, latency: dict | None, revision: str | None
) -> str:
    scores = json.loads((stage_dir / "scores.json").read_text())
    notes = (stage_dir / "RUN_NOTES.md").read_text()
    model = notes.splitlines()[0].removeprefix("# ").split(":")[0]
    repo_line = next(line for line in notes.splitlines() if line.startswith("- Model:"))
    repo = repo_line.split("[", 1)[1].split("]", 1)[0]
    rev = revision or repo_line.split("revision `", 1)[1].split("`", 1)[0]
    engine = next(
        line for line in notes.splitlines() if line.startswith("- Engine:")
    ).split("`")[1]
    server = "d3_server.py" if engine.startswith("d3_") else "decision25_server.py"
    tree = f"https://huggingface.co/datasets/{dataset}/tree/main/runs/{name}"
    areas = {a["id"]: a["skill"] for a in scores["areas"]}
    lat = ""
    if latency:
        p95 = f", p95 {latency['p95_ms']:.0f} ms" if "p95_ms" in latency else ""
        lat = (
            f"Our latency measurement on one NVIDIA RTX PRO 6000 (single process, one request at a time, 750 timed + "
            f"10 warm-up requests stratified by benchmark and length): median {latency['median_ms']:.0f} ms, mean "
            f"{latency['mean_ms']:.0f} ms, p80 {latency['p80_ms']:.0f} ms{p95}. Please re-measure on your RTX PRO 6000."
        )
    line = (
        f"- **{model}**: [{repo}](https://huggingface.co/{repo}) @ `{rev[:8]}`; results [{dataset}/runs/{name}]"
        f"({tree}) (`scores.json`, complete); engine `{engine}` (in the model repo), kit "
        f"`{KIT_COMMIT[:7]}`; see RUN_NOTES.md for hardware and settings."
    )
    body = [
        f"submissions: add {model}, Decision Index 0.3 public run ({scores['decision_index']})",
        "",
        f"> Full public 0.3 run, complete: {scores['completed']:,} scoreable requests ("
        + ", ".join(f"{v:,} {k}" for k, v in sorted(scores["counts"].items()))
        + f"). Public index {scores['decision_index']} (raw {scores['raw_index']}). Results: {tree}.",
        "",
        "| Public 0.3 | Index | Knowledge | Language | Retrieval | Tools | Arts |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **{model}** | **{scores['decision_index']}** | "
        + " | ".join(
            f"{100 * areas[a]:.2f}"
            for a in ("knowledge", "language", "retrieval", "tools", "arts")
        )
        + " |",
        "",
        notes.split("\n", 2)[2].strip(),
        "",
        lat,
        "",
        "To run it (CUDA):",
        "```",
        f"hf download {repo} --revision {rev} --local-dir model",
        'pip install -r model/requirements.txt "decision-index @ git+https://github.com/apolinario/decision-index@'
        f'{KIT_COMMIT[:7]}"',
        f"PYTHONPATH=model python -m decision_index run --engine {engine} --option model=model "
        "--option device=cuda:0 --out runs/model --compact",
        "```",
        f"A `/v1/systemone` server is in the repo too: `python model/{server} --model model` (then the kit's "
        "`http` engine).",
        "",
        "Line for `submissions/README.md`:",
        "",
        line,
        "",
    ]
    return "\n".join(body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    m = sub.add_parser("merge")
    g = sub.add_parser("stage")
    for p in (s, m, g):
        p.add_argument("--kit", required=True, type=Path)
        p.add_argument("--suite-dir", required=True, type=Path)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--out", required=True, type=Path)
    m.add_argument("--shards", required=True, type=Path)
    m.add_argument("--name", required=True)
    m.add_argument("--out", required=True, type=Path)
    g.add_argument("--run", required=True, type=Path)
    g.add_argument("--out", required=True, type=Path)
    g.add_argument("--manifest", required=True, type=Path)
    g.add_argument("--repo", required=True)
    g.add_argument("--revision", required=True)
    g.add_argument("--hardware", required=True)
    g.add_argument(
        "--settings",
        default="BF16 backbone with SDPA attention, FP32 readout and softmax, temperature 1 "
        "(scores use argmax / p >= 0.5), noncausal full attention per decision_config.json, last-token "
        "readout over each question's answer codes, input limit from decision_config.json, thinking not "
        "applicable (no text is generated)",
    )
    u = sub.add_parser("upload")
    u.add_argument("--stage", required=True, type=Path)
    u.add_argument("--dataset", default="vllm-sr/decision-2.5-decision-index")
    u.add_argument("--name", required=True)
    u.add_argument("--allow-public", action="store_true")
    t = sub.add_parser("pr-text")
    t.add_argument("--stage", required=True, type=Path)
    t.add_argument("--dataset", default="vllm-sr/decision-2.5-decision-index")
    t.add_argument("--name", required=True)
    t.add_argument("--latency", type=Path)
    t.add_argument("--revision")
    t.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "shard":
        out = shard(a.kit, a.suite_dir, a.n, a.out)
    elif a.cmd == "merge":
        out = merge(a.kit, a.suite_dir, a.shards, a.name, a.out)
    elif a.cmd == "stage":
        out = stage(
            a.kit,
            a.suite_dir,
            a.run,
            a.out,
            json.loads(a.manifest.read_text()),
            a.repo,
            a.revision,
            a.hardware,
            a.settings,
        )
    elif a.cmd == "upload":
        out = upload(a.stage, a.dataset, a.name, a.allow_public)
    else:
        latency = json.loads(a.latency.read_text()) if a.latency else None
        a.out.write_text(pr_text(a.stage, a.dataset, a.name, latency, a.revision))
        out = {"pr_text": str(a.out)}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
