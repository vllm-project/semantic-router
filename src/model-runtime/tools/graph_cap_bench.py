"""Exact-profile throughput on Index-like traffic under one graph-size cap (performance records).

    python3 tools/graph_cap_bench.py --package DIR --stream NAME:PROMPTS.jsonl:COUNT ... --cap 4096
        --output OUT.json [--questions 1,1,1,1,2,2,3,4,8] [--window 1000] [--device rocm:0] [--base-path DIR]

Index-like traffic is many requests of a few questions each about states of very different lengths, so
request shapes (rows, padded length) rarely repeat exactly. Request i asks about prompt i's state the
questions of prompts i .. i+k-1, with k cycling through --questions. Every request runs once, in stream
order, on the exact profile, with graphs captured only up to --cap padded tokens (0: no graphs; the
default cap is fast.MAX_GRAPH_TOKENS). Reported: requests/s and questions/s overall and per window of
--window requests, latency percentiles, the graph cache (captures, replays, eager forwards, refusals once
full, cached graphs), and a digest of every answer. Graphs and eager forwards compute the same bits, so
the digest must not depend on the cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_bench import percentile
from gpu_parity import canonical, load, system_one
from vllm_srun.profiles.exact import ExactProfile


def stream(specs: list[str], sizes: list[int]) -> list[dict]:
    prompts = []
    for spec in specs:
        name, path, count = spec.split(":")
        with open(path, encoding="utf-8") as source:
            rows = [json.loads(line) for line in source][: int(count)]
        prompts.extend({"panel": name, **row} for row in rows)
    requests = []
    for index, prompt in enumerate(prompts):
        size = sizes[index % len(sizes)]
        questions = {}
        for offset, other in enumerate(prompts[index : index + size]):
            for question_id, question in other["questions"].items():
                questions[f"{offset}.{question_id}"] = question
        requests.append(
            {
                "id": f"{prompt['panel']}/{prompt['id']}",
                "state": prompt["state"],
                "questions": questions,
            }
        )
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", required=True)
    parser.add_argument("--stream", action="append", required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", default="1,1,1,1,2,2,3,4,8")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--device", default="rocm:0")
    parser.add_argument("--base-path")
    args = parser.parse_args()
    args.no_graphs = args.no_fused = False
    import torch

    requests = stream(args.stream, [int(size) for size in args.questions.split(",")])
    model = load(args)
    graphs = model.engine_model.graphs
    graphs.max_tokens = args.cap
    exact = ExactProfile()
    digest = hashlib.sha256()
    milliseconds, windows, questions = [], [], 0
    window_started, window_questions = time.perf_counter(), 0
    started = window_started
    for index, request in enumerate(requests, 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        response = system_one(model, exact, request["state"], request["questions"])
        torch.cuda.synchronize()
        milliseconds.append(1000 * (time.perf_counter() - tick))
        digest.update(canonical([request["id"], response["answers"]]).encode())
        questions += len(request["questions"])
        window_questions += len(request["questions"])
        if index % args.window == 0 or index == len(requests):
            now = time.perf_counter()
            windows.append(
                {
                    "requests": index,
                    "requests_per_s": (
                        index - (windows[-1]["requests"] if windows else 0)
                    )
                    / (now - window_started),
                    "questions_per_s": window_questions / (now - window_started),
                    "graphs": graphs.receipt(),
                }
            )
            print(json.dumps(windows[-1]), flush=True)
            window_started, window_questions = now, 0
    seconds = time.perf_counter() - started
    result = {
        "schema": "model-runtime-graph-cap-bench/1",
        "model": model.info.id,
        "device": model.engine_model.device_info.name,
        "cap": args.cap,
        "requests": len(requests),
        "questions": questions,
        "seconds": seconds,
        "requests_per_s": len(requests) / seconds,
        "questions_per_s": questions / seconds,
        "ms": {
            "p50": percentile(milliseconds, 0.5),
            "p95": percentile(milliseconds, 0.95),
            "p99": percentile(milliseconds, 0.99),
            "mean": statistics.mean(milliseconds),
        },
        "graphs": graphs.receipt(),
        "windows": windows,
        "answers_sha256": digest.hexdigest(),
    }
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(result, sink, indent=1)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("cap", "requests_per_s", "graphs", "answers_sha256")
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
