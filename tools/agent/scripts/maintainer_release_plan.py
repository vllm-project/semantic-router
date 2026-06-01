"""Release-plan parsing helpers for maintainer ops."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MIN_RELEASE_TASK_MATCH_TOKENS = 3
MIN_SEARCH_TOKEN_LENGTH = 4
ISSUE_LINK_RE = re.compile(
    r"\[#(?P<number>\d+)\]\((?P<url>https://github\.com/[^)]+/issues/\d+)\)"
    r"\s*(?P<title>.*)"
)


def open_release_tasks(plan_path: Path) -> list[str]:
    tasks: list[str] = []
    current: list[str] | None = None
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ]"):
            if current:
                tasks.append(" ".join(current))
            current = [stripped[6:].strip()]
            continue
        if stripped.startswith("- ["):
            if current:
                tasks.append(" ".join(current))
            current = None
            continue
        if current is not None:
            if line.startswith((" ", "\t")) and stripped:
                current.append(stripped)
                continue
            if stripped:
                tasks.append(" ".join(current))
                current = None
    if current:
        tasks.append(" ".join(current))
    return [re.sub(r"`([^`]+)`", r"\1", task) for task in tasks]


def release_plan_issue_context(
    snapshot_issues: list[dict[str, Any]], release_plan: Path
) -> list[dict[str, Any]]:
    issues_by_number = {
        int(issue["number"]): issue
        for issue in snapshot_issues
        if str(issue.get("number", "")).isdigit()
    }
    for issue in release_plan_issue_anchors(release_plan):
        issues_by_number.setdefault(issue["number"], issue)
    return list(issues_by_number.values())


def release_plan_issue_anchors(release_plan: Path) -> list[dict[str, Any]]:
    anchors = []
    seen = set()
    lines = release_plan.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        match = ISSUE_LINK_RE.search(line)
        if not match:
            continue
        number = int(match.group("number"))
        if number in seen:
            continue
        seen.add(number)
        title_parts = [match.group("title").strip()]
        for continuation in lines[index + 1 :]:
            if not continuation.startswith((" ", "\t")) or not continuation.strip():
                break
            title_parts.append(continuation.strip())
        title = " ".join(part for part in title_parts if part).strip(" .")
        anchors.append(
            {
                "number": number,
                "title": title or f"issue #{number}",
                "url": match.group("url"),
                "labels": [],
                "milestone": {},
                "_plan_anchor": True,
            }
        )
    return anchors


def match_release_tasks_to_issues(
    tasks: list[str], issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    matches = []
    for task in tasks:
        task_tokens = searchable_tokens(task)
        task_td_ids = set(re.findall(r"TD\d{3}", task, flags=re.IGNORECASE))
        issue_matches = []
        for issue in issues:
            title = issue.get("title", "")
            title_tokens = searchable_tokens(title)
            title_td_ids = set(re.findall(r"TD\d{3}", title, flags=re.IGNORECASE))
            if task_td_ids and task_td_ids.intersection(title_td_ids):
                issue_matches.append(issue)
                continue
            min_match_tokens = (
                2 if issue.get("_plan_anchor") else MIN_RELEASE_TASK_MATCH_TOKENS
            )
            if len(task_tokens.intersection(title_tokens)) >= min_match_tokens:
                issue_matches.append(issue)
        matches.append({"task": task, "matches": issue_matches})
    return matches


def searchable_tokens(value: str) -> set[str]:
    stop_words = {
        "and",
        "the",
        "for",
        "with",
        "into",
        "this",
        "that",
        "using",
        "covering",
        "close",
        "create",
        "update",
        "release",
    }
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", value)
        if len(token) >= MIN_SEARCH_TOKEN_LENGTH and token.lower() not in stop_words
    }
