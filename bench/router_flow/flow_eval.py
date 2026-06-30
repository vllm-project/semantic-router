#!/usr/bin/env python3
"""Run a small Router Flow comparison.

The harness is intentionally lightweight: it calls an OpenAI-compatible chat
endpoint, optionally grades answers with an OpenAI-compatible judge, and writes
JSONL plus a summary JSON. It does not contain benchmark data from SWE-bench,
TerminalBench, LiveCodeBench, or other restricted sources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path(__file__).with_name("prompts.jsonl")
HEURISTIC_PASS_THRESHOLD = 0.45
MIN_RUBRIC_TERM_LENGTH = 5


@dataclass(frozen=True)
class Arm:
    name: str
    model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default="http://localhost:8899/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Evaluation arm as name=model, for example flow=vllm-sr/flow",
    )
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument("--judge-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--judge-model", default="")
    parser.add_argument(
        "--judge-request-extra-json",
        default="",
        help="Optional JSON object merged into judge chat completion requests.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("bench/router_flow/results")
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--request-extra-json",
        default="",
        help="Optional JSON object merged into every chat completion request.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def parse_arm(raw: str) -> Arm:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--arm must be formatted as name=model")
    name, model = raw.split("=", 1)
    name = name.strip()
    model = model.strip()
    if not name or not model:
        raise argparse.ArgumentTypeError("--arm must include non-empty name and model")
    return Arm(name=name, model=model)


def load_dataset(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def chat_completion(
    *,
    base_url: str,
    api_key_env: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    request_extra: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, str], float]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(request_extra)
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(api_key_env, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            elapsed_ms = (time.perf_counter() - started) * 1000
            data = json.loads(response.read().decode())
            return data, dict(response.headers.items()), elapsed_ms
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request to {url} failed: {exc}") from exc


def answer_text(completion: dict[str, Any]) -> str:
    choices = completion.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return json.dumps({"tool_calls": tool_calls}, sort_keys=True)
    return ""


def usage(completion: dict[str, Any]) -> dict[str, int]:
    raw = completion.get("usage") or {}
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
    }


def looper_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower().startswith("x-vsr-looper")
    }


def heuristic_score(answer: str, rubric: str) -> dict[str, Any]:
    if not answer.strip():
        return {"score": 0.0, "pass": False, "rationale": "empty answer"}
    answer_terms = set(re.findall(r"[a-zA-Z0-9_]+", answer.lower()))
    rubric_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9_]+", rubric.lower())
        if len(term) >= MIN_RUBRIC_TERM_LENGTH
    }
    overlap = len(answer_terms & rubric_terms)
    denom = max(6, min(16, len(rubric_terms)))
    score = min(1.0, overlap / denom)
    return {
        "score": round(score, 3),
        "pass": score >= HEURISTIC_PASS_THRESHOLD,
        "rationale": "heuristic lexical overlap; use --judge-model for publishable grading",
    }


def judge_score(
    *,
    judge_base_url: str,
    judge_api_key_env: str,
    judge_model: str,
    judge_request_extra: dict[str, Any],
    item: dict[str, Any],
    answer: str,
    timeout: float,
) -> dict[str, Any]:
    if not judge_base_url or not judge_model:
        return heuristic_score(answer, str(item.get("rubric") or ""))
    prompt = (
        "Grade the answer against the rubric. Return only JSON with keys "
        "score (0 to 1), pass (boolean), and rationale (short string).\n\n"
        f"Task:\n{item['prompt']}\n\nRubric:\n{item.get('rubric', '')}\n\nAnswer:\n{answer}"
    )
    completion, _, _ = chat_completion(
        base_url=judge_base_url,
        api_key_env=judge_api_key_env,
        model=judge_model,
        messages=[
            {"role": "system", "content": "You are a strict benchmark grader."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
        request_extra=judge_request_extra,
        timeout=timeout,
    )
    text = answer_text(completion)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {
            "score": 0.0,
            "pass": False,
            "rationale": f"judge returned non-JSON: {text[:160]}",
        }
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "score": 0.0,
            "pass": False,
            "rationale": f"judge JSON parse error: {exc}",
        }
    score = float(parsed.get("score") or 0.0)
    return {
        "score": max(0.0, min(1.0, score)),
        "pass": bool(parsed.get("pass")),
        "rationale": str(parsed.get("rationale") or "")[:500],
    }


def run_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arms = [parse_arm(raw) for raw in args.arm]
    dataset = load_dataset(args.dataset, args.limit)
    request_extra = parse_request_extra(args.request_extra_json)
    judge_request_extra = parse_request_extra(args.judge_request_extra_json)
    samples: list[dict[str, Any]] = []

    for item in dataset:
        for arm in arms:
            messages = [
                {
                    "role": "system",
                    "content": "Answer directly. Show concise reasoning when useful.",
                },
                {"role": "user", "content": item["prompt"]},
            ]
            started = time.time()
            error = ""
            completion: dict[str, Any] = {}
            headers: dict[str, str] = {}
            latency_ms = 0.0
            try:
                completion, headers, latency_ms = chat_completion(
                    base_url=args.base_url,
                    api_key_env=args.api_key_env,
                    model=arm.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    request_extra=request_extra,
                    timeout=args.timeout,
                )
            except Exception as exc:
                error = str(exc)
            answer = answer_text(completion)
            grade = judge_score(
                judge_base_url=args.judge_base_url,
                judge_api_key_env=args.judge_api_key_env,
                judge_model=args.judge_model,
                judge_request_extra=judge_request_extra,
                item=item,
                answer=answer,
                timeout=args.timeout,
            )
            samples.append(
                {
                    "id": item["id"],
                    "category": item.get("category", ""),
                    "arm": arm.name,
                    "model": arm.model,
                    "answer": answer,
                    "grade": grade,
                    "latency_ms": round(latency_ms, 1),
                    "usage": usage(completion),
                    "looper_headers": looper_headers(headers),
                    "error": error,
                    "created": int(started),
                }
            )
    return samples, summarize(samples)


def parse_request_extra(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--request-extra-json must decode to a JSON object")
    return parsed


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_arm.setdefault(sample["arm"], []).append(sample)

    summary: dict[str, Any] = {"arms": {}, "categories": {}}
    for arm, rows in sorted(by_arm.items()):
        scores = [float(row["grade"]["score"]) for row in rows]
        latencies = [float(row["latency_ms"]) for row in rows if row["latency_ms"]]
        total_tokens = [int(row["usage"]["total_tokens"]) for row in rows]
        summary["arms"][arm] = {
            "n": len(rows),
            "pass_rate": round(
                sum(1 for row in rows if row["grade"]["pass"]) / max(1, len(rows)), 3
            ),
            "mean_score": round(statistics.fmean(scores), 3) if scores else 0.0,
            "mean_latency_ms": (
                round(statistics.fmean(latencies), 1) if latencies else 0.0
            ),
            "mean_total_tokens": (
                round(statistics.fmean(total_tokens), 1) if total_tokens else 0.0
            ),
            "errors": sum(1 for row in rows if row["error"]),
        }

    for category in sorted({row["category"] for row in samples}):
        summary["categories"][category] = {}
        for arm, rows in sorted(by_arm.items()):
            cat_rows = [row for row in rows if row["category"] == category]
            if not cat_rows:
                continue
            scores = [float(row["grade"]["score"]) for row in cat_rows]
            summary["categories"][category][arm] = {
                "n": len(cat_rows),
                "mean_score": round(statistics.fmean(scores), 3),
                "pass_rate": round(
                    sum(1 for row in cat_rows if row["grade"]["pass"])
                    / max(1, len(cat_rows)),
                    3,
                ),
            }
    return summary


def write_outputs(
    output_dir: Path, samples: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.jsonl").open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    samples, summary = run_eval(args)
    write_outputs(args.output_dir, samples, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
