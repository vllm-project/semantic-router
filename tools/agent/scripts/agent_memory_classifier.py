#!/usr/bin/env python3
"""Classify whether a PR has the requested author-provided review brief."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable
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
REVIEW_BRIEF_LINE_PATTERN = re.compile(r"(?im)^\s*Review brief:\s*(?P<value>.*?)\s*$")


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


@dataclass(frozen=True)
class ChangedFileRef:
    filename: str
    status: str | None = None


def normalize_path(path: str) -> str:
    normalized = path.strip().strip("`").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def valid_brief_path(path: str) -> bool:
    return BRIEF_PATH_PATTERN.fullmatch(normalize_path(path)) is not None


def extract_path_from_review_brief_value(value: str) -> str | None:
    """Extract a valid normalized brief path from the Review brief line."""
    candidate = re.sub(r"<!--.*?-->", "", value).strip()
    if not candidate:
        return None

    candidate = normalize_path(candidate)
    if not candidate:
        return None
    if not valid_brief_path(candidate):
        return None
    return candidate


def extract_review_brief_path(body: str) -> str | None:
    for match in REVIEW_BRIEF_LINE_PATTERN.finditer(body or ""):
        return extract_path_from_review_brief_value(match.group("value"))
    return None


def normalize_status(status: Any) -> str | None:
    if status is None:
        return None
    normalized = str(status).strip().lower()
    return normalized or None


def coerce_changed_file_ref(entry: Any) -> ChangedFileRef | None:
    """Normalize one changed-file entry from workflow metadata.

    The entry may already be a ChangedFileRef, a legacy changed_files path string,
    or a GitHub listFiles dict with filename/status. Invalid shapes are ignored.
    """
    if isinstance(entry, ChangedFileRef):
        return entry
    if isinstance(entry, str):
        return ChangedFileRef(filename=normalize_path(entry))
    if not isinstance(entry, dict):
        return None

    filename = entry.get("filename")
    if not filename:
        return None
    return ChangedFileRef(
        filename=normalize_path(str(filename)),
        status=normalize_status(entry.get("status")),
    )


def normalize_changed_file_refs(
    *,
    changed_files: Iterable[str] | None,
    changed_file_entries: Iterable[Any] | None,
) -> list[ChangedFileRef]:
    """Normalize new status-aware entries or legacy path strings."""
    refs: list[ChangedFileRef] = []
    if changed_file_entries is not None:
        for entry in changed_file_entries:
            ref = coerce_changed_file_ref(entry)
            if ref is not None:
                refs.append(ref)
        return refs

    for path in changed_files or []:
        refs.append(ChangedFileRef(filename=normalize_path(str(path))))
    return refs


def changed_file_refs(metadata: dict[str, Any]) -> list[ChangedFileRef]:
    entries = metadata.get("changed_file_entries")
    if isinstance(entries, list):
        return normalize_changed_file_refs(
            changed_files=None,
            changed_file_entries=entries,
        )
    return normalize_changed_file_refs(
        changed_files=[str(path) for path in metadata.get("changed_files", [])],
        changed_file_entries=None,
    )


def brief_text_has_body(text: str) -> bool:
    if not text.strip():
        return False

    body_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == "# review brief":
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped.startswith("#"):
            continue
        body_lines.append(stripped)

    return bool(body_lines)


def base_review_brief_has_body(memory_path: str, repo_root: Path) -> bool:
    path = repo_root / memory_path
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return brief_text_has_body(text)


def review_brief_is_usable(
    memory_path: str,
    changed_files: list[ChangedFileRef],
    repo_root: Path,
    *,
    review_brief_text_by_path: dict[str, Any] | None = None,
) -> bool:
    if not valid_brief_path(memory_path):
        return False

    changed_file = next(
        (entry for entry in changed_files if entry.filename == memory_path),
        None,
    )
    if changed_file is not None:
        if changed_file.status == "removed":
            return False
        if (
            review_brief_text_by_path is not None
            and memory_path in review_brief_text_by_path
        ):
            brief_text = review_brief_text_by_path[memory_path]
            return brief_text_has_body(
                str(brief_text) if brief_text is not None else ""
            )
        return True

    return base_review_brief_has_body(memory_path, repo_root)


def classify_pr(
    *,
    body: str,
    additions: int,
    deletions: int,
    changed_files: list[str] | None = None,
    changed_file_entries: list[Any] | None = None,
    review_brief_text_by_path: dict[str, Any] | None = None,
    repo_root: Path,
    threshold: int = MEMORY_THRESHOLD,
) -> Classification:
    total_delta = additions + deletions
    memory_required = total_delta >= threshold
    changed = normalize_changed_file_refs(
        changed_files=changed_files,
        changed_file_entries=changed_file_entries,
    )
    candidate_path = extract_review_brief_path(body)
    memory_path = (
        candidate_path
        if candidate_path
        and review_brief_is_usable(
            candidate_path,
            changed,
            repo_root,
            review_brief_text_by_path=review_brief_text_by_path,
        )
        else None
    )
    memory_invalid = False
    invalid_reason = None
    memory_present = memory_path is not None
    gate_reason = review_brief_gate_failure_reason(
        memory_required=memory_required,
        memory_present=memory_present,
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
) -> str | None:
    if memory_required and not memory_present:
        return "Review brief is required for this PR size but no valid brief was found"
    return None


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
            "This check is a hard merge gate for required review-brief context.",
        ]
    )
    return "\n".join(lines)


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def classify_from_metadata(
    metadata: dict[str, Any], repo_root: Path, threshold: int
) -> Classification:
    """Adapt workflow metadata into the classifier's normalized inputs.

    Uses body, additions/deletions, legacy changed_files, status-aware
    changed_file_entries, and optional PR-head brief text.
    """
    review_brief_text_by_path = metadata.get("review_brief_text_by_path")
    if not isinstance(review_brief_text_by_path, dict):
        review_brief_text_by_path = {}
    return classify_pr(
        body=str(metadata.get("body") or ""),
        additions=int(metadata.get("additions") or 0),
        deletions=int(metadata.get("deletions") or 0),
        changed_files=[str(path) for path in metadata.get("changed_files", [])],
        changed_file_entries=changed_file_refs(metadata),
        review_brief_text_by_path=review_brief_text_by_path,
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
