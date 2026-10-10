"""Text answers of two code revisions of one d3 package (same weights), in one process on one device.

Each package's own ``d3_runtime.py`` is imported under its own module name, loaded in turn (the first copy is freed
before the second loads, so a 96 GB GPU fits) and asked the same text requests through ``system_one``. One process
keeps the kernels and their Triton autotune state the same for both. Pass: every response is identical (answers,
probabilities, ``model`` and ``usage``).

    python -m d25.vega.release.text_parity_rev --old <v3.0.0 dir> --new <v3.0.1 dir> --rows parity-600.jsonl.gz \
        --out text-parity-rev.json [--device cuda:0]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from d25.vega.release.sample import read_rows


def load_runtime(package: Path, name: str):
    for cached in ("d3_format", "d3_runtime"):
        sys.modules.pop(cached, None)
    sys.path.insert(0, str(package))
    try:
        spec = importlib.util.spec_from_file_location(name, package / "d3_runtime.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(package))
    return module


def answers(
    package: Path, name: str, rows: list[dict], device: str
) -> tuple[list[dict], float]:
    import torch

    module = load_runtime(package, name)
    model = module.D3.from_pretrained(str(package), device=device)
    started = time.time()
    out = [model.system_one(state=r["state"], questions=r["questions"]) for r in rows]
    seconds = time.time() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, seconds


def probabilities(response: dict) -> dict[str, list[float]]:
    out = {}
    for key, answer in response["answers"].items():
        if "probabilities" in answer:
            out[key] = list(answer["probabilities"].values())
        elif "noul" in answer:
            out[key] = [1 - answer["noul"], answer["noul"]]
    return out


def decision(answer: dict):
    if "choice" in answer:
        return answer["choice"]
    if "noul" in answer:
        return answer["noul"] > 0.5
    return answer.get("score")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    rows = read_rows(args.rows)[: args.limit]
    old, old_seconds = answers(args.old.absolute(), "d3_runtime_old", rows, args.device)
    new, new_seconds = answers(args.new.absolute(), "d3_runtime_new", rows, args.device)
    identical = changed = 0
    max_dp = 0.0
    differing = []
    for row, a, b in zip(rows, old, new):
        if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
            identical += 1
            continue
        differing.append(row.get("run_id", row.get("id")))
        pa, pb = probabilities(a), probabilities(b)
        for key in set(pa) | set(pb):
            if key not in pa or key not in pb or len(pa[key]) != len(pb[key]):
                max_dp = float("inf")
                continue
            max_dp = max(max_dp, max(abs(x - y) for x, y in zip(pa[key], pb[key])))
        changed += sum(
            decision(a["answers"].get(k, {})) != decision(b["answers"].get(k, {}))
            for k in b["answers"]
        )
    report = {
        "old": str(args.old),
        "new": str(args.new),
        "runtime_sha256": {
            "old": sha(args.old / "d3_runtime.py"),
            "new": sha(args.new / "d3_runtime.py"),
        },
        "requests": len(rows),
        "questions": sum(len(r["questions"]) for r in rows),
        "identical_responses": identical,
        "differing_requests": differing[:20],
        "answer_changes": changed,
        "max_abs_dp": max_dp,
        "seconds": {"old": round(old_seconds, 1), "new": round(new_seconds, 1)},
        "model_field": sorted({r.get("model") for r in new}),
        "pass": identical == len(rows) and len(rows) > 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "requests",
                    "questions",
                    "identical_responses",
                    "answer_changes",
                    "max_abs_dp",
                    "model_field",
                    "pass",
                )
            }
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
