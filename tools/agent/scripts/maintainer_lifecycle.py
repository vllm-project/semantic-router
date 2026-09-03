"""Build deterministic label proposals for the Maintainer board."""

from __future__ import annotations

from typing import Any


def label_names(item: dict[str, Any]) -> set[str]:
    return {
        label["name"] if isinstance(label, dict) else str(label)
        for label in item.get("labels", [])
    }


def missing_label_actions(
    items: list[dict[str, Any]], *, action: str, label: str
) -> list[dict[str, Any]]:
    return [
        {
            "action": action,
            "target": f"#{item['number']}",
            "labels": [label],
        }
        for item in items
        if label not in label_names(item)
    ]


def proposed_actions(
    snapshot: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    lifecycle = policy["labels"]["lifecycle"]
    pr_state = policy["labels"]["pr_state"]
    intake_issues = [
        issue
        for issue in snapshot["raw"]["issues"]
        if not label_names(issue).intersection(
            {lifecycle["accepted"], lifecycle["needs_acceptance"]}
        )
    ]
    actions = missing_label_actions(
        intake_issues,
        action="label_issue",
        label=lifecycle["needs_acceptance"],
    )
    actions.extend(
        missing_label_actions(
            snapshot["groups"]["issues"]["stale"],
            action="label_issue",
            label=lifecycle["stale"],
        )
    )
    pr_groups = (
        ("needs-rebase", "needs_rebase"),
        ("review-now", "needs_review"),
        ("needs-author", "needs_author"),
        ("unblock", "blocked"),
        ("merge-candidate", "merge_ready"),
        ("close-candidate", "close_candidate"),
    )
    for group, state in pr_groups:
        actions.extend(
            missing_label_actions(
                snapshot["groups"]["pull_requests"][group],
                action="label_pr",
                label=pr_state[state],
            )
        )
    return actions
