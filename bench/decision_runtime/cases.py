"""Synthetic Decision requests shared byte-for-byte by both measured services."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

# This benchmark runs from a source checkout and validates against the runtime's
# actual public contract. It does not keep a second response schema in bench/.
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "vllm-sr"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from decision_runtime.contracts import SystemOneRequest  # noqa: E402

MODELS = (
    "llm-semantic-router/Decision-1.0-Kai-0.6B",
    "llm-semantic-router/Decision-1.0-Lex-0.6B",
    "llm-semantic-router/Decision-1.0-Eos-0.8B",
    "llm-semantic-router/Decision-1.0-Sol-2B",
    "llm-semantic-router/Decision-1.0-Nox-4B",
    "llm-semantic-router/Decision-1.0-Lux-9B",
)
DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")


@dataclass(frozen=True)
class Case:
    id: str
    request: SystemOneRequest
    body: bytes
    sha256: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("nonfinite JSON number")


def load_cases(path: Path, model: str) -> tuple[Case, ...]:
    """Load a fixed case cohort and validate each request before any traffic."""

    if model not in MODELS:
        raise ValueError(f"unsupported Decision model: {model}")
    cases: list[Case] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(
                raw_line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"cases line {line_number} is invalid JSON") from error
        if not isinstance(item, dict) or set(item) != {"id", "state", "questions"}:
            raise ValueError(f"cases line {line_number} must have id, state, questions")
        case_id = item["id"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or len(case_id) > 80
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in case_id
            )
        ):
            raise ValueError(f"cases line {line_number} has an invalid id")
        if case_id in seen:
            raise ValueError(f"cases line {line_number} has a duplicate id")
        seen.add(case_id)
        payload = {
            "model": model,
            "state": item["state"],
            "questions": item["questions"],
        }
        try:
            request = SystemOneRequest.model_validate(payload)
        except ValidationError as error:
            raise ValueError(
                f"cases line {line_number} violates SystemOne request contract"
            ) from error
        body = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        cases.append(
            Case(
                id=case_id,
                request=request,
                body=body,
                sha256=hashlib.sha256(body).hexdigest(),
            )
        )
    if not cases:
        raise ValueError("cases file contains no requests")
    return tuple(cases)


def cohort_sha256(cases: tuple[Case, ...]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.id.encode("ascii"))
        digest.update(b"\x00")
        digest.update(case.body)
        digest.update(b"\x00")
    return digest.hexdigest()
