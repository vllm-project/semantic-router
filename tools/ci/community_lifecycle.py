#!/usr/bin/env python3
"""Validate and synchronize the GitHub issue and pull-request lifecycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from community_lifecycle_github import (
    GitHubClient,
    accept_issue_event,
    sync_issue_event,
    sync_issue_kind_event,
    sync_issue_queue,
    sync_pull_request_event,
    sync_pull_request_queue,
    validate_pull_request_event,
    validate_title_event,
)
from community_lifecycle_policy import (
    EPIC,
    ISSUE_DELIVERY_STATE_LABELS,
    MAINTAINER_OWNER,
    OWNER_LABELS,
    PR_STATE_LABELS,
    WORKGROUP_LABELS,
    WORKGROUP_OPTIONS,
    evaluate_issue_acceptance,
    evaluate_pull_request,
    extract_related_issue_numbers,
    is_epic_title,
    plan_issue,
    plan_issue_kind,
    proposed_workgroup,
    title_format_error,
)

__all__ = [
    "EPIC",
    "ISSUE_DELIVERY_STATE_LABELS",
    "MAINTAINER_OWNER",
    "OWNER_LABELS",
    "PR_STATE_LABELS",
    "WORKGROUP_LABELS",
    "WORKGROUP_OPTIONS",
    "evaluate_issue_acceptance",
    "evaluate_pull_request",
    "extract_related_issue_numbers",
    "is_epic_title",
    "plan_issue",
    "plan_issue_kind",
    "proposed_workgroup",
    "title_format_error",
]


def load_event(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "sync-issue",
            "sync-issue-kind",
            "sync-issues",
            "accept-issue",
            "validate-pr",
            "sync-pr",
            "sync-prs",
            "validate-title",
        ),
    )
    parser.add_argument("--event", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    event = load_event(args.event)
    if args.command == "validate-title":
        validate_title_event(event)
        return 0
    client = GitHubClient()
    if args.command == "sync-issue":
        sync_issue_event(client, event)
    elif args.command == "sync-issue-kind":
        sync_issue_kind_event(client, event)
    elif args.command == "sync-issues":
        sync_issue_queue(client, event)
    elif args.command == "accept-issue":
        accept_issue_event(client, event)
    elif args.command == "validate-pr":
        validate_pull_request_event(client, event)
    elif args.command == "sync-pr":
        sync_pull_request_event(client, event)
    else:
        sync_pull_request_queue(client, event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
