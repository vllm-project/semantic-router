"""Versioned deterministic graders for visible final answers only."""

from __future__ import annotations

import json
import re

MCQ_GRADER_VERSION = "sr-bench-mcq-final-v2"
FINAL_GRADER_VERSION = "sr-bench-final-v2"
ARC_MAX_COLOR = 9

_STRICT_ANSWER = re.compile(r"(?:ANSWER\s*:\s*)?\(?([A-J])\)?[.]?", re.IGNORECASE)
_LABEL = r"(?:\*\*|__|`)?\(?([A-J])\)?(?:\*\*|__|`)?"
_DECLARATION = re.compile(
    r"\b(?:the\s+(?:correct\s+|final\s+)?answer\s+is|"
    r"(?:final\s+|correct\s+)?answer\s*:)\s*" + _LABEL + r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_ALTERNATIVE = re.compile(
    r"\s*(?:or|and|/|,)\s*" + _LABEL + r"(?![A-Za-z0-9_])", re.IGNORECASE
)
_BOXED = re.compile(r"\\boxed\{\s*([A-J])\s*\}", re.IGNORECASE)


def _unwrapped(line):
    text = line.strip()
    for wrapper in ("**", "__", "`"):
        if text.startswith(wrapper) and text.endswith(wrapper):
            return text[len(wrapper) : -len(wrapper)].strip()
    return text


def multiple_choice_grade(case, final):
    """Accept explicit, unambiguous answers without guessing from prose.

    This is sr-bench's conservative adapter, not the upstream last-letter
    heuristic. Markdown is presentation; conflicting declarations are ambiguous.
    Format compliance remains independent from capability correctness.
    """
    text = final.strip()
    strict = _STRICT_ANSWER.fullmatch(text)
    candidates = set()
    # Transport owns channel separation. An unresolved thinking boundary must
    # never become an answer source for direct/offline grader callers either.
    if "<think>" not in text and "</think>" not in text:
        lines = [line for line in text.splitlines() if line.strip()]
        standalone = [
            match.group(1).upper()
            for line in lines
            if (match := _STRICT_ANSWER.fullmatch(_unwrapped(line)))
        ]
        # A leading answer line may be followed by explanation. Other isolated
        # letters are not an answer unless part of an explicit declaration.
        if lines and _STRICT_ANSWER.fullmatch(_unwrapped(lines[0])):
            candidates.update(standalone)
        for match in _DECLARATION.finditer(text):
            candidates.add(match.group(1).upper())
            tail = text[match.end() :]
            while alternative := _ALTERNATIVE.match(tail):
                candidates.add(alternative.group(1).upper())
                tail = tail[alternative.end() :]
        candidates.update(m.group(1).upper() for m in _BOXED.finditer(text))
    answer = next(iter(candidates)) if len(candidates) == 1 else None
    correct = answer is not None and answer == str(case["answer"]).upper()
    return {
        "answer": answer,
        "correct": correct,
        "score": float(correct),
        "details": {
            "grader_version": MCQ_GRADER_VERSION,
            "strict_format": strict is not None,
            "answer_status": (
                "parsed" if answer else "ambiguous" if candidates else "unparsed"
            ),
        },
    }


def basic_grade(case, final):
    expected = case["answer"]
    if case["benchmark"] in {"mmlu-pro", "gpqa-diamond"}:
        return multiple_choice_grade(case, final)
    if case["benchmark"] == "arc-agi-2":
        try:
            answer = json.loads(final)

            def valid_grid(grid):
                return (
                    isinstance(grid, list)
                    and bool(grid)
                    and bool(grid[0])
                    and all(
                        isinstance(row, list)
                        and len(row) == len(grid[0])
                        and all(type(x) is int and 0 <= x <= ARC_MAX_COLOR for x in row)
                        for row in grid
                    )
                )

            valid = (
                (
                    isinstance(answer, list)
                    and bool(answer)
                    and all(valid_grid(grid) for grid in answer)
                )
                if case.get("metadata", {}).get("output_format") == "grids"
                else valid_grid(answer)
            )
        except (ValueError, TypeError):
            answer = None
            valid = False
        correct = valid and answer == expected
        return {
            "answer": answer,
            "correct": correct,
            "score": float(correct),
            "details": {"valid_grid": valid},
        }
    raise ValueError("No basic grader for benchmark")
