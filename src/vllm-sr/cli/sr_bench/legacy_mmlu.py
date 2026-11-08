from __future__ import annotations

import re
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from .adapters import AdapterContext, BenchmarkAdapter
from .sources import (
    COUNTS,
    HF_SOURCES,
    _acquire,
    _local_source,
    _stratified_order,
    _write_dataset,
)

BENCHMARK: str = "legacy-mmlu-pro"
PROTOCOL: str = "bench-mmlu-pro-plain-v1"
SOURCE_BENCHMARK: str = "mmlu-pro"

# Preserve the legacy precedence, including its permissive prose fallbacks.
# https://github.com/vllm-project/semantic-router/blob/a00f4b219357fbc77410748f3e4dbdc0a8e59e28/bench/reasoning/router_reason_bench_multi_dataset.py#L369-L431
ANSWER_PATTERNS: tuple[str, ...] = (
    r"(?:answer\s*:?\s+)([A-Z])(?:\s|[.!?)]|$)",
    r"(?:final\s*answer\s*:?\s+)([A-Z])(?:\s|[.!?)]|$)",
    r"(?:therefore|thus|so).*?(?:answer\s+is\s+|is\s+)([A-Z])(?:\s|[.!?)]|$)",
    r"(?:correct\s+answer\s+is\s+)([A-Z])",
    r"(?:option\s+)([A-Z])",
    r"(?:choice\s+)([A-Z])",
    r"([A-Z])\)",
    r"([A-Z])\s*[.!]?\s*$",
)


def _extract_answer(final: str) -> str | None:
    structured = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", final, re.IGNORECASE)
    if structured:
        answer = re.sub(r"[.!?]+$", "", structured.group(1).strip())
        if answer:
            return answer
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, final, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    lines = final.strip().split("\n")[-3:]
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if len(line) == 1 and line.upper() in ascii_uppercase:
            return line.upper()
    for raw_line in reversed(lines):
        match = re.search(
            r"(?:is|answer|option|choice)\s+([A-Z])(?:\s|[.!?]|$)",
            raw_line.strip(),
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
    match = re.search(r"\b([A-Z])(?:\s|[.!?)]|$)", final)
    return match.group(1).upper() if match else None


def grade_answer(case: dict[str, Any], final: str) -> dict[str, Any]:
    answer = _extract_answer(final)
    correct = answer is not None and answer == case["answer"]
    return {
        "answer": answer,
        "correct": correct,
        "score": float(correct),
        "details": {
            "grader_version": PROTOCOL,
            "answer_status": "parsed" if answer else "unparsed",
        },
    }


def _execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    response = context.call(case["messages"])
    return grade_answer(case=case, final=response["final"])


def _preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    metadata = case.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Legacy MMLU requires choice metadata")
    count = metadata.get("choice_count")
    answer = case.get("answer")
    if (
        type(count) is not int
        or not 1 <= count <= len(ascii_uppercase)
        or not isinstance(answer, str)
        or len(answer) != 1
        or answer not in ascii_uppercase[:count]
    ):
        raise ValueError("Legacy MMLU requires a reference letter within its choices")


def normalize_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Legacy MMLU source records must be objects")
        identity = row.get("question_id")
        question, category, options = (
            row.get("question"),
            row.get("category"),
            row.get("options"),
        )
        if (
            isinstance(identity, bool)
            or not isinstance(identity, (str, int))
            or not str(identity).strip()
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(category, str)
            or not category.strip()
            or not isinstance(options, list)
            or not 1 <= len(options) <= len(ascii_uppercase)
            or not all(isinstance(option, str) for option in options)
        ):
            raise ValueError(
                "Legacy MMLU requires question_id, question, category and text choices"
            )
        identity = str(identity)
        if identity in ids:
            raise ValueError("Duplicate legacy MMLU source question IDs")
        ids.add(identity)
        choices = "\n".join(
            f"{ascii_uppercase[index]}) {option}"
            for index, option in enumerate(options)
            if option.lower() != "n/a"
        ).rstrip()
        case: dict[str, Any] = {
            "id": f"{BENCHMARK}/{identity}",
            "benchmark": BENCHMARK,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nOptions:\n{choices}\n\n"
                        "Instructions:\n"
                        "- Read the question carefully\n"
                        "- Consider each option thoroughly\n"
                        "- Choose the single best answer\n"
                        "- Respond with ONLY the format: Answer: [letter]\n"
                        "- Do not include any other text after your answer\n\n"
                        "Your response:"
                    ),
                }
            ],
            "answer": row.get("answer"),
            "metadata": {
                "source_question_id": identity,
                "stratum": category,
                "choice_count": len(options),
            },
        }
        _preflight(case=case, manifest={}, cache={})
        cases.append(case)
    if not cases:
        raise ValueError("Legacy MMLU source contains no tasks")
    return cases


def _prepare(
    *,
    benchmark: str,
    profile: str,
    store: str | Path,
    source_path: str | Path | None,
    revision: str | None,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    profiles: dict[str, int] = dict(
        zip(("smoke", "quick", "standard"), COUNTS[SOURCE_BENCHMARK], strict=True)
    )
    if benchmark != BENCHMARK or profile not in profiles:
        raise ValueError("Unknown legacy MMLU benchmark or profile")
    desired = profiles[profile]
    if limit is not None:
        if type(limit) is not int or not 0 < limit <= desired:
            raise ValueError(
                "limit must be a positive integer within the profile budget"
            )
        desired = limit
    root = Path(store).expanduser().resolve()
    rows, source = (
        _local_source(benchmark=SOURCE_BENCHMARK, path=source_path, revision=revision)
        if source_path
        else _acquire(benchmark=SOURCE_BENCHMARK, root=root)
    )
    ordered = _stratified_order(cases=normalize_records(rows), seed=seed)
    offset = profiles["quick"] if profile == "standard" else 0
    selected = ordered[offset : offset + desired]
    if len(selected) != desired:
        raise ValueError(
            f"Source has insufficient tasks for disjoint {profile}: "
            f"need {offset + desired}, found {len(ordered)}"
        )
    for case in selected:
        case["metadata"].update(
            split="holdout" if profile == "standard" else "dev",
            source_revision=source["revision"],
        )
    return _write_dataset(
        root=root,
        cases=selected,
        profile=profile,
        seed=seed,
        sources={BENCHMARK: {**source, "normalizer": PROTOCOL}},
        custom=limit is not None,
    )


ADAPTER: BenchmarkAdapter = BenchmarkAdapter(
    id=BENCHMARK,
    title="MMLU-Pro (legacy plain diagnostic)",
    kind="multiple-choice",
    source_url="https://huggingface.co/datasets/" + HF_SOURCES[SOURCE_BENCHMARK][0],
    version=PROTOCOL,
    execute=_execute,
    preflight=_preflight,
    prepare=_prepare,
    requires_answer=True,
)
