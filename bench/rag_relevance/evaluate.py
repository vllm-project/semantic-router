#!/usr/bin/env python3
"""Compare prepared RAG relevance bindings on identical labeled candidates."""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path

MAX_CANDIDATES = 32
MAX_RELEVANCE = 3


def ndcg(labels, order, k):
    def dcg(indices):
        return sum(
            (2 ** labels[i] - 1) / math.log2(rank + 2)
            for rank, i in enumerate(indices[:k])
        )

    ideal = dcg(sorted(range(len(labels)), key=lambda i: -labels[i]))
    return dcg(order) / ideal if ideal else None


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def load_cases(path):
    raw = path.read_bytes()
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not cases:
        raise ValueError("dataset is empty")
    for case in cases:
        candidates = case["candidates"]
        if not case["query"] or not 1 <= len(candidates) <= MAX_CANDIDATES:
            raise ValueError("each query requires 1 to 32 candidates")
        ids = [item["id"] for item in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique within a query")
        for candidate in candidates:
            grade = candidate["relevance"]
            if (
                not candidate["text"]
                or type(grade) is not int
                or not 0 <= grade <= MAX_RELEVANCE
            ):
                raise ValueError(
                    "each candidate requires text and a relevance grade from 0 to 3"
                )
    return cases, hashlib.sha256(raw).hexdigest()


def evaluate(args):
    cases, digest = load_cases(args.dataset)
    authorization = os.environ.get("ROUTER_API_TOKEN")
    headers = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = "Bearer " + authorization
    runs = {name: [] for name in ("retrieval", "cross_encoder", "decision")}
    for case in cases:
        candidates = case["candidates"]
        labels = [item["relevance"] for item in candidates]
        baseline = list(range(len(candidates)))
        runs["retrieval"].append(
            {"query_id": case["id"], "ndcg": ndcg(labels, baseline, args.top_k)}
        )
        for name, recipe in (
            ("cross_encoder", args.cross_encoder_recipe),
            ("decision", args.decision_recipe),
        ):
            payload = {
                "recipe": recipe,
                "binding": "rag.reranker",
                "pairs": [
                    {"query": case["query"], "document": item["text"]}
                    for item in candidates
                ],
            }
            request = urllib.request.Request(
                args.url.rstrip("/") + "/api/v1/diagnostics/models/rerank",
                data=json.dumps(payload).encode(),
                headers=headers,
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=40) as response:
                result = json.load(response)
            elapsed = time.perf_counter() - started
            output = result["result"]
            expected = "decision_relevance" if name == "decision" else "relevance_logit"
            scores = output["scores"]
            if (
                output["score_type"] != expected
                or len(scores) != len(candidates)
                or not all(math.isfinite(score) for score in scores)
            ):
                raise ValueError("unexpected score contract")
            if name == "decision" and (
                output.get("coverage") != ["complete"] * len(candidates)
                or not all(0 <= score <= MAX_RELEVANCE for score in scores)
            ):
                raise ValueError("incomplete or invalid decision judgment")
            order = sorted(baseline, key=lambda i: -scores[i])
            runs[name].append(
                {
                    "query_id": case["id"],
                    "scores": scores,
                    "ranking": [candidates[i]["id"] for i in order],
                    "ndcg": ndcg(labels, order, args.top_k),
                    "seconds": elapsed,
                    "binding": result["binding"],
                    "input_tokens": output.get("input_tokens"),
                }
            )
    summaries = {}
    for name, rows in runs.items():
        quality = [row["ndcg"] for row in rows if row["ndcg"] is not None]
        summary = {
            "queries": len(rows),
            "queries_with_relevant_candidates": len(quality),
            "mean_ndcg": statistics.mean(quality) if quality else None,
        }
        if name != "retrieval":
            latencies = [row["seconds"] for row in rows]
            summary.update(
                p50_seconds=percentile(latencies, 0.5),
                p95_seconds=percentile(latencies, 0.95),
            )
        summaries[name] = summary
    return {
        "dataset_sha256": digest,
        "top_k": args.top_k,
        "measurement": "one diagnostic call per query and binding; includes transport and any runtime cache hits; excludes retrieval and generation",
        "summary": summaries,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Router management API base URL")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cross-encoder-recipe", required=True)
    parser.add_argument("--decision-recipe", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    result = evaluate(args)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
