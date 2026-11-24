"""C1: convert MCQ sources into model_eval's tasks format.

Verified facts this module encodes (plan doc §3.2 item (a), verified
2026-11-01 against src/training/model_eval/multi_benchmark_eval.py and
mcq_judge.py):

- multi_benchmark_eval.py reads ``task_id`` and ``prompt_user`` (required)
  plus ``prompt_system``, ``dataset``, ``category``, ``language`` and
  ``difficulty`` (optional, carried through to records verbatim). It never
  reads an options field: MCQ options must be rendered into ``prompt_user``
  text.
- mcq_judge.py reads the ground truth from the task row's ``answer`` field
  and compares it for exact equality against the letter extracted from the
  completion. The extractor only produces uppercase A-J, so a lowercase or
  non-letter ``answer`` would fail every judgment; rows are validated to a
  single uppercase A-J letter at conversion time (ContractError -> exit 2).
- The judge's answer protocol ("Answer: X" or a final standalone letter)
  is enforced by the prompt instruction, mirroring sr-bench's own MCQ
  rendering so both sources produce prompts the same judge can grade.

Supported sources (auto-detected per row):
- sr-bench normalized dataset rows (src/vllm-sr/cli/sr_bench/
  source_records.py ``_case``): id / benchmark / messages / answer /
  metadata.stratum. MMLU-Pro questions taken from sr-bench's pinned
  revision arrive in this shape; ``category`` comes from
  ``metadata.stratum``.
- raw MCQ rows (office 题集 read directly): id / source / category /
  question / options / answer / answer_type. Options are rendered into
  ``prompt_user`` with the same layout sr-bench uses.

Every emitted task row also carries the recipe-lab cases vocabulary
(case_id = task_id, prompt = prompt_user, truth = answer) so the existing
``split`` / ``hash`` / contract machinery consumes tasks.jsonl unchanged
(grouping defaults to one group per question id).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .contract import ContractError

# mcq_judge.py's extractor recognizes uppercase A-J only.
ANSWER_LETTER_RE = re.compile(r"^[A-J]$")

# mcq_judge grades letters A-J, so a row can carry at most ten options.
MAX_MCQ_OPTIONS = 10


def normalize_answer_letter(raw: Any, where: str) -> str:
    """Validate and normalize one ground-truth letter to uppercase A-J."""
    if not isinstance(raw, str):
        raise ContractError(f"{where}: answer must be a letter string, got {raw!r}")
    letter = raw.strip().upper()
    if not ANSWER_LETTER_RE.match(letter):
        raise ContractError(f"{where}: answer must be a single letter A-J, got {raw!r}")
    return letter


def render_mcq_prompt(question: str, options: list[str]) -> str:
    """Render question + options like sr-bench's _mcq (source_records.py)."""
    rendered = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
    return f"{question}\n\n{rendered}\n\nReturn only the letter of the correct answer."


def last_user_message(messages: Any) -> str:
    """Extract the last user message content from an sr-bench row."""
    if not isinstance(messages, list) or not messages:
        raise ContractError("sr-bench row has no messages")
    for message in reversed(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        ):
            return message["content"]
    raise ContractError("sr-bench row has no non-empty user message")


def _task_row_from_srbench(row: dict[str, Any]) -> dict[str, Any]:
    task_id = row.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ContractError("sr-bench row is missing a non-empty id")
    where = f"task {task_id}"
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ContractError(f"{where}: metadata must be an object")
    answer = normalize_answer_letter(row.get("answer"), where)
    prompt_user = last_user_message(row.get("messages"))
    task: dict[str, Any] = {
        "task_id": task_id,
        "prompt_user": prompt_user,
        "answer": answer,
    }
    dataset = row.get("benchmark")
    if isinstance(dataset, str) and dataset:
        task["dataset"] = dataset
    category = metadata.get("stratum")
    if isinstance(category, str) and category:
        task["category"] = category
    return task


def _task_row_from_raw(
    row: dict[str, Any], default_dataset: str | None
) -> dict[str, Any]:
    task_id = row.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ContractError("raw MCQ row is missing a non-empty id")
    where = f"task {task_id}"
    question = row.get("question")
    options = row.get("options")
    if not isinstance(question, str) or not question.strip():
        raise ContractError(f"{where}: question must be a non-empty string")
    if (
        not isinstance(options, list)
        or not options
        or not all(isinstance(option, str) and option for option in options)
    ):
        raise ContractError(f"{where}: options must be a non-empty string list")
    if len(options) > MAX_MCQ_OPTIONS:
        raise ContractError(
            f"{where}: {len(options)} options exceed the judge's A-J range"
        )
    answer = normalize_answer_letter(row.get("answer"), where)
    prompt_user = render_mcq_prompt(question, options)
    task: dict[str, Any] = {
        "task_id": task_id,
        "prompt_user": prompt_user,
        "answer": answer,
    }
    dataset = row.get("source") or default_dataset
    if isinstance(dataset, str) and dataset:
        task["dataset"] = dataset
    category = row.get("category")
    if isinstance(category, str) and category:
        task["category"] = category
    language = row.get("language")
    if isinstance(language, str) and language:
        task["language"] = language
    return task


def rows_to_model_eval_tasks(
    rows: list[dict[str, Any]], *, default_dataset: str | None = None
) -> list[dict[str, Any]]:
    """Convert source rows into model_eval task rows (with cases mirrors).

    Detection per row: ``messages`` + ``benchmark`` => sr-bench normalized
    shape; ``question`` + ``options`` => raw MCQ shape; anything else is a
    ContractError, never a silently mangled row.
    """
    tasks: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ContractError(f"source line {index} is not a JSON object")
        if "messages" in row or "benchmark" in row:
            task = _task_row_from_srbench(row)
        elif "question" in row or "options" in row:
            task = _task_row_from_raw(row, default_dataset)
        else:
            raise ContractError(
                f"source line {index} is neither an sr-bench normalized row "
                "(id/benchmark/messages/answer) nor a raw MCQ row "
                "(id/question/options/answer)"
            )
        # Cases-vocabulary mirrors keep split/hash/contract working on the
        # same file (load_cases requires case_id + prompt).
        task["case_id"] = task["task_id"]
        task["prompt"] = task["prompt_user"]
        task["truth"] = task["answer"]
        tasks.append(task)
    return tasks


def load_source_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load the source JSONL (one JSON object per line)."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path}:{line_no} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{line_no} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ContractError(f"{path} contains no source rows")
    return rows


def write_tasks(tasks: list[dict[str, Any]], path: str | Path) -> Path:
    """Write the tasks JSONL deterministically (sorted keys)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    return target
