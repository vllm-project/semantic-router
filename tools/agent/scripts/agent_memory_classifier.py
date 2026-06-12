#!/usr/bin/env python3
"""Classify whether a PR has the requested author-provided review brief."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MEMORY_THRESHOLD = 500
LABEL_PRESENT = "agent-memory-present"
LABEL_MISSING = "agent-memory-missing"
LABEL_NOT_REQUIRED = "agent-memory-not-required"
LABELS = (LABEL_PRESENT, LABEL_MISSING, LABEL_NOT_REQUIRED)

BRIEF_PATH_PATTERN = re.compile(
    r"^docs/agent/reviews/(?P<year>\d{4})/"
    r"(?P=year)-(?P<month>0[1-9]|1[0-2])-"
    r"(?P<day>0[1-9]|[12]\d|3[01])-"
    r"[a-z0-9]+(?:-[a-z0-9]+)*\.md$"
)
REVIEW_BRIEF_LINE_PATTERN = re.compile(
    r"(?im)^\s*Review brief:\s*(?:\[[^\]]+\]\()?`?" r"(?P<path>[^)`\s]+\.md)"
)
ANY_REVIEW_BRIEF_PATH_PATTERN = re.compile(
    r"docs/agent/reviews/[0-9]{4}/[A-Za-z0-9._/-]+\.md"
)


@dataclass(frozen=True)
class Classification:
    memory_required: bool
    memory_present: bool
    memory_invalid: bool
    memory_path: str | None
    invalid_reason: str | None
    gate_passed: bool
    gate_reason: str | None
    labels_to_add: list[str]
    labels_to_remove: list[str]
    comment_body: str


def normalize_path(path: str) -> str:
    normalized = path.strip().strip("`").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def valid_brief_path(path: str) -> bool:
    return BRIEF_PATH_PATTERN.fullmatch(normalize_path(path)) is not None


def extract_review_brief_path(body: str) -> str | None:
    match = REVIEW_BRIEF_LINE_PATTERN.search(body or "")
    if match:
        return normalize_path(match.group("path"))
    fallback = ANY_REVIEW_BRIEF_PATH_PATTERN.search(body or "")
    if fallback:
        return normalize_path(fallback.group(0))
    return None


def classify_pr(
    *,
    body: str,
    additions: int,
    deletions: int,
    changed_files: list[str],
    repo_root: Path,
    threshold: int = MEMORY_THRESHOLD,
) -> Classification:
    total_delta = additions + deletions
    memory_required = total_delta >= threshold
    changed = {normalize_path(path) for path in changed_files}
    memory_path = extract_review_brief_path(body)
    invalid_reason = None
    if memory_path is not None:
        invalid_reason = validate_memory_reference(memory_path, changed, repo_root)
    memory_missing = memory_path is None
    memory_invalid = memory_path is not None and invalid_reason is not None
    memory_present = memory_path is not None and not memory_invalid
    gate_reason = review_brief_gate_failure_reason(
        memory_required=memory_required,
        memory_present=memory_present,
        memory_invalid=memory_invalid,
        memory_missing=memory_missing,
        invalid_reason=invalid_reason,
    )
    gate_passed = gate_reason is None

    if memory_invalid or (memory_required and not memory_present):
        labels_to_add = [LABEL_MISSING]
    elif memory_present:
        labels_to_add = [LABEL_PRESENT]
    elif not memory_required:
        labels_to_add = [LABEL_NOT_REQUIRED]
    else:
        labels_to_add = [LABEL_MISSING]

    labels_to_remove = [label for label in LABELS if label not in labels_to_add]
    return Classification(
        memory_required=memory_required,
        memory_present=memory_present,
        memory_invalid=memory_invalid,
        memory_path=memory_path,
        invalid_reason=invalid_reason,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
        labels_to_add=labels_to_add,
        labels_to_remove=labels_to_remove,
        comment_body=build_comment(
            memory_required=memory_required,
            memory_present=memory_present,
            memory_invalid=memory_invalid,
            memory_path=memory_path,
            invalid_reason=invalid_reason,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
            total_delta=total_delta,
            threshold=threshold,
        ),
    )


def review_brief_gate_failure_reason(
    *,
    memory_required: bool,
    memory_present: bool,
    memory_invalid: bool,
    memory_missing: bool,
    invalid_reason: str | None,
) -> str | None:
    if memory_invalid:
        return invalid_reason or "Review brief reference is invalid"
    if memory_required and memory_missing:
        return "Review brief is required for this PR size but no valid brief was found"
    if memory_required and not memory_present:
        return "Review brief is required for this PR size but no valid brief was found"
    return None


def validate_memory_reference(
    memory_path: str,
    changed_files: set[str],
    repo_root: Path,
) -> str | None:
    if not valid_brief_path(memory_path):
        return (
            "Review brief path must match "
            "docs/agent/reviews/YYYY/YYYY-MM-DD-<short-kebab-title>.md"
        )
    if memory_path in changed_files:
        return None
    if (repo_root / memory_path).is_file():
        return None
    return "Review brief must exist on the base branch or be part of this PR"


def build_comment(
    *,
    memory_required: bool,
    memory_present: bool,
    memory_invalid: bool,
    memory_path: str | None,
    invalid_reason: str | None,
    gate_passed: bool,
    gate_reason: str | None,
    total_delta: int,
    threshold: int,
) -> str:
    lines = [
        "## Agent review brief status",
        "",
        f"- Size signal: `{total_delta}` changed lines; threshold is `{threshold}`.",
    ]
    if memory_invalid:
        lines.append(f"- Memory context: invalid. {invalid_reason}.")
    elif memory_present:
        lines.append(f"- Memory context: present at `{memory_path}`.")
    elif not memory_required:
        lines.append("- Memory context: not required for this PR size.")
    else:
        lines.append("- Memory context: missing.")
    if gate_passed:
        lines.append("- Hard gate: passed.")
    else:
        lines.append(f"- Hard gate: failed. {gate_reason}.")
    lines.extend(
        [
            "",
            "This check is a hard merge gate for required, valid review-brief references.",
        ]
    )
    return "\n".join(lines)


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def classify_from_metadata(
    metadata: dict[str, Any], repo_root: Path, threshold: int
) -> Classification:
    return classify_pr(
        body=str(metadata.get("body") or ""),
        additions=int(metadata.get("additions") or 0),
        deletions=int(metadata.get("deletions") or 0),
        changed_files=[str(path) for path in metadata.get("changed_files", [])],
        repo_root=repo_root,
        threshold=threshold,
    )


def write_github_outputs(output: Classification, github_output: Path) -> None:
    payload = asdict(output)
    lines = []
    for key, value in payload.items():
        if isinstance(value, (list, dict)):
            rendered = json.dumps(value)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = ""
        else:
            rendered = str(value)
        if "\n" in rendered:
            lines.append(f"{key}<<EOF\n{rendered}\nEOF")
        else:
            lines.append(f"{key}={rendered}")
    with github_output.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--threshold", type=int, default=MEMORY_THRESHOLD)
    args = parser.parse_args()

    result = classify_from_metadata(
        load_metadata(args.metadata),
        repo_root=args.repo_root,
        threshold=args.threshold,
    )
    rendered = json.dumps(asdict(result), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    if "GITHUB_OUTPUT" in os.environ:
        github_output = Path(os.environ["GITHUB_OUTPUT"])
        write_github_outputs(result, github_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
