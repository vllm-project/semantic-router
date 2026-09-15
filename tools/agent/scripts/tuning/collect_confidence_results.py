#!/usr/bin/env python3
"""Collect paired small/large model results for confidence calibration.

This is a deliberately conservative data-collection helper.  It accepts a
prepared question file, calls an OpenAI-compatible endpoint with bounded
concurrency, and writes the paired result files consumed by
``confidence_calibration.py``.  It never changes router configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from tuning.confidence_calibration import build_artifact, write_artifact  # noqa: E402

ANSWER_PATTERN = re.compile(
    r"(?:answer|答案)\s*[:\uFF1A]?\s*\[?\s*([A-J])\s*\]?",
    re.IGNORECASE,
)
SPLITS = ("train", "calibration", "held_out")
ERROR_SNIPPET_HALF = 120
ERROR_SNIPPET_LIMIT = ERROR_SNIPPET_HALF * 2


def _read_questions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("dataset must be a JSON array or JSONL file of objects")

    seen: set[str] = set()
    for item in value:
        question_id = str(item.get("question_id") or "").strip()
        if not question_id:
            raise ValueError("every question requires a non-empty question_id")
        if question_id in seen:
            raise ValueError(f"duplicate question_id: {question_id}")
        seen.add(question_id)
        if not str(item.get("category") or "").strip():
            raise ValueError(f"question {question_id} has no category")
        if not str(item.get("prompt") or "").strip():
            raise ValueError(f"question {question_id} has no prompt")
        if not str(item.get("correct_answer") or "").strip():
            raise ValueError(f"question {question_id} has no correct_answer")
        if item.get("split") not in SPLITS:
            raise ValueError(
                f"question {question_id} split must be one of {', '.join(SPLITS)}"
            )
    if not value:
        raise ValueError("dataset must contain at least one question")
    counts = {split: sum(item["split"] == split for item in value) for split in SPLITS}
    missing = [split for split, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"dataset is missing non-empty splits: {', '.join(missing)}")
    return value


def _request_json(
    endpoint: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request failed: {error.reason}") from error

    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError("model endpoint returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("model endpoint returned a non-object JSON response")
    if "error" in value:
        raise RuntimeError(f"model endpoint returned an error: {value['error']}")
    return value


def _call_model(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    item: Mapping[str, Any],
    include_logprobs: bool,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer this multiple-choice question. Output only one uppercase "
                    "letter from A to J. Do not explain your reasoning."
                ),
            },
            {"role": "user", "content": item["prompt"]},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    if include_logprobs:
        payload.update({"logprobs": True, "top_logprobs": 0})

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = _request_json(endpoint, api_key, payload, timeout)
            return _result_from_response(response, item, model, include_logprobs)
        except (RuntimeError, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise RuntimeError(
        f"{model} failed for {item['question_id']} after {retries + 1} attempts: "
        f"{last_error}"
    ) from last_error


def _result_from_response(
    response: Mapping[str, Any],
    item: Mapping[str, Any],
    model: str,
    include_logprobs: bool,
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("model response has no choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("model response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response has empty content")

    predicted = _extract_answer(content)
    expected = str(item["correct_answer"]).strip().upper()
    result: dict[str, Any] = {
        "question_id": str(item["question_id"]),
        "category": str(item["category"]),
        "predicted": predicted,
        "correct_answer": expected,
        "correct": predicted == expected,
        "model": model,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
    }
    if include_logprobs:
        average, token_count = _extract_avg_logprob(choice, message)
        result["avg_logprob"] = average
        result["logprob_token_count"] = token_count
    return result


def _extract_answer(content: str) -> str:
    matches = ANSWER_PATTERN.findall(content)
    if matches:
        return matches[-1].upper()
    fallback = re.findall(r"\b([A-J])\b", content.upper())
    if fallback:
        return fallback[-1]
    normalized = " ".join(content.split())
    snippet = normalized[:ERROR_SNIPPET_HALF]
    if len(normalized) > ERROR_SNIPPET_LIMIT:
        snippet += " ... " + normalized[-ERROR_SNIPPET_HALF:]
    raise ValueError(
        "model response does not contain an answer letter A-J; " f"content={snippet!r}"
    )


def _extract_avg_logprob(
    choice: Mapping[str, Any], message: Mapping[str, Any]
) -> tuple[float, int]:
    containers = [message.get("logprobs"), choice.get("logprobs")]
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        tokens = container.get("content")
        if not isinstance(tokens, list) or not tokens:
            continue
        values: list[float] = []
        for token in tokens:
            if not isinstance(token, Mapping) or not isinstance(
                token.get("logprob"), (int, float)
            ):
                raise ValueError("model response contains unusable token logprob")
            values.append(float(token["logprob"]))
        if values:
            return sum(values) / len(values), len(values)
    raise ValueError("model response does not contain content token logprobs")


def _collect_model(
    items: list[dict[str, Any]],
    *,
    stage: str,
    endpoint: str,
    api_key: str,
    model: str,
    include_logprobs: bool,
    max_tokens: int,
    timeout: float,
    retries: int,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    ordered_items = sorted(items, key=lambda item: str(item["question_id"]))
    results: dict[str, dict[str, Any]] = {}
    total = len(ordered_items)
    print(f"{stage}: starting {total} requests", flush=True)

    def collect_one(item: dict[str, Any], position: int) -> dict[str, Any]:
        print(
            f"{stage}: {position}/{total} requesting {item['question_id']}",
            flush=True,
        )
        result = _call_model(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            item=item,
            include_logprobs=include_logprobs,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
        )
        print(f"{stage}: {position}/{total} completed", flush=True)
        return result

    executor = ThreadPoolExecutor(max_workers=max_concurrency)
    futures: dict[Future[dict[str, Any]], str] = {}
    next_position = 1

    def submit_next() -> None:
        nonlocal next_position
        item = ordered_items[next_position - 1]
        future = executor.submit(collect_one, item, next_position)
        futures[future] = str(item["question_id"])
        next_position += 1

    try:
        for _ in range(min(max_concurrency, total)):
            submit_next()
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                question_id = futures.pop(future)
                results[question_id] = future.result()
                if next_position <= total:
                    submit_next()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return [results[str(item["question_id"])] for item in ordered_items]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect paired results and build a confidence artifact"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    parser.add_argument("--small-model", default="qwen3-8b")
    parser.add_argument("--large-model", default="qwen3-32b")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--dataset-name", default="mmlu-pro")
    parser.add_argument("--dataset-version", default="test")
    parser.add_argument("--small-version", default=None)
    parser.add_argument("--large-version", default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override both model token limits (legacy convenience option)",
    )
    parser.add_argument("--small-max-tokens", type=int, default=64)
    parser.add_argument("--large-max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="collect only the first N questions in each split (for smoke tests)",
    )
    parser.add_argument("--max-escalation-rate", type=float, default=0.85)
    parser.add_argument("--min-net-uplift", type=int, default=0)
    parser.add_argument("--max-regression-rate", type=float, default=None)
    parser.add_argument("--current-threshold", type=float, default=0.72)
    return parser


def _validate_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[int, int]:

    if args.timeout <= 0 or args.retries < 0:
        parser.error("timeout must be positive; retries cannot be negative")
    if args.max_concurrency < 1:
        parser.error("max-concurrency must be positive")
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("limit-per-split must be positive")
    small_max_tokens = (
        args.max_tokens if args.max_tokens is not None else args.small_max_tokens
    )
    large_max_tokens = (
        args.max_tokens if args.max_tokens is not None else args.large_max_tokens
    )
    if small_max_tokens < 1 or large_max_tokens < 1:
        parser.error("model token limits must be positive")
    return small_max_tokens, large_max_tokens


def _require_api_key(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"environment variable {args.api_key_env} is not set")
    return api_key


def _select_questions(path: Path, limit_per_split: int | None) -> list[dict[str, Any]]:
    questions = _read_questions(path)
    if limit_per_split is None:
        return questions

    questions_by_split = {
        split: sorted(
            (item for item in questions if item["split"] == split),
            key=lambda item: str(item["question_id"]),
        )[:limit_per_split]
        for split in SPLITS
    }
    selected = [item for split in SPLITS for item in questions_by_split[split]]
    print(
        f"Smoke-test limit: {limit_per_split} questions per split",
        flush=True,
    )
    return selected


def _collect_models(
    args: argparse.Namespace,
    api_key: str,
    questions: list[dict[str, Any]],
    small_max_tokens: int,
    large_max_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    small_results = _collect_model(
        questions,
        stage=f"small/{args.small_model}",
        endpoint=args.endpoint,
        api_key=api_key,
        model=args.small_model,
        include_logprobs=True,
        max_tokens=small_max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        max_concurrency=args.max_concurrency,
    )
    large_results = _collect_model(
        questions,
        stage=f"large/{args.large_model}",
        endpoint=args.endpoint,
        api_key=api_key,
        model=args.large_model,
        include_logprobs=False,
        max_tokens=large_max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        max_concurrency=args.max_concurrency,
    )
    return small_results, large_results


def _write_split_results(
    output_dir: Path,
    questions: list[dict[str, Any]],
    small_results: list[dict[str, Any]],
    large_results: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_split = {
        split: [item for item in questions if item["split"] == split]
        for split in SPLITS
    }
    small_by_id = {item["question_id"]: item for item in small_results}
    large_by_id = {item["question_id"]: item for item in large_results}
    for split in SPLITS:
        ids = {item["question_id"] for item in by_split[split]}
        _write_json(
            output_dir / f"small-{split}.json",
            [small_by_id[question_id] for question_id in sorted(ids)],
        )
        _write_json(
            output_dir / f"large-{split}.json",
            [large_by_id[question_id] for question_id in sorted(ids)],
        )


def _build_manifest(
    args: argparse.Namespace,
    small_max_tokens: int,
    large_max_tokens: int,
) -> dict[str, Any]:
    return {
        "schema_version": "confidence-calibration/v1",
        "name": f"{args.dataset_name}-confidence-calibration",
        "method": "avg_logprob",
        "score_domain": {"min": 0.0, "max": 1.0},
        "normalization": {
            "type": "linear_clamped",
            "min_logprob": -3.0,
            "max_logprob": 0.0,
        },
        "dataset": {
            "name": args.dataset_name,
            "version": args.dataset_version,
            "digest": _sha256_file(args.dataset),
        },
        "population": "Deterministically prepared MMLU-Pro test questions",
        "outcome": "Whether the parsed model answer matches the answer key",
        "expected_impact": "Improve escalation quality within the declared budget",
        "models": {
            "small": {
                "id": args.small_model,
                "version": args.small_version or args.small_model,
            },
            "large": {
                "id": args.large_model,
                "version": args.large_version or args.large_model,
            },
        },
        "splits": {
            split: {
                "small_results": f"small-{split}.json",
                "large_results": f"large-{split}.json",
            }
            for split in SPLITS
        },
        "objective": {
            "primary_metric": "accuracy",
            "max_escalation_rate": args.max_escalation_rate,
            "min_net_uplift": args.min_net_uplift,
            **(
                {"max_regression_rate": args.max_regression_rate}
                if args.max_regression_rate is not None
                else {}
            ),
        },
        "fallback": {"on_no_safe_threshold": "retain_current"},
        "policy": {
            "current_threshold": args.current_threshold,
            "rollback_identity": f"threshold-{args.current_threshold:g}",
        },
        "collection": {
            "endpoint": args.endpoint,
            "api_key_env": args.api_key_env,
            "temperature": 0,
            "enable_thinking": False,
            "small_logprobs": {"enabled": True, "top_logprobs": 0},
            "retries": args.retries,
            "max_concurrency": args.max_concurrency,
            "limit_per_split": args.limit_per_split,
            "max_tokens": {
                "small": small_max_tokens,
                "large": large_max_tokens,
            },
        },
        "approval_state": "pending_review",
    }


def _write_artifact(
    output_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    artifact = build_artifact(manifest_path)
    artifact_path = output_dir / "confidence-calibration-artifact.json"
    write_artifact(artifact, artifact_path)
    return artifact, manifest_path, artifact_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    small_max_tokens, large_max_tokens = _validate_args(args, parser)
    api_key = _require_api_key(args, parser)

    try:
        questions = _select_questions(args.dataset, args.limit_per_split)
        small_results, large_results = _collect_models(
            args, api_key, questions, small_max_tokens, large_max_tokens
        )
        _write_split_results(args.output_dir, questions, small_results, large_results)
        manifest = _build_manifest(args, small_max_tokens, large_max_tokens)
        artifact, manifest_path, artifact_path = _write_artifact(
            args.output_dir, manifest
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"confidence result collection failed: {error}", file=sys.stderr)
        return 2

    print(f"Collected {len(questions)} questions")
    print(f"Manifest: {manifest_path}")
    print(f"Artifact: {artifact_path}")
    print(f"Artifact ID: {artifact['artifact_id']}")
    print(f"Status: {artifact['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
