"""GitHub API adapters for community lifecycle policy enforcement."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any
from urllib.parse import quote

from community_lifecycle_policy import (
    ACCEPTED,
    API_PAGE_SIZE,
    MAINTAINER_PERMISSIONS,
    NEEDS_ACCEPTANCE,
    PR_STATE_LABELS,
    RELEASE_BLOCKER,
    WORKGROUP_LABELS,
    evaluate_issue_acceptance,
    evaluate_pull_request,
    extract_related_issue_numbers,
    label_names,
    plan_issue,
    plan_issue_kind,
    title_format_error,
)


class GitHubClient:
    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        ignore_not_found: bool = False,
    ) -> Any:
        command = ["gh", "api", endpoint, "--method", method]
        if payload is not None:
            command.extend(["--input", "-"])
        result = subprocess.run(
            command,
            input=json.dumps(payload) if payload is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            if ignore_not_found and "HTTP 404" in result.stderr:
                return None
            sys.stderr.write(result.stderr)
            raise SystemExit(result.returncode)
        return json.loads(result.stdout) if result.stdout.strip() else None


def repository_name(event: dict[str, Any]) -> str:
    value = os.environ.get("GITHUB_REPOSITORY")
    if value:
        return value
    return event["repository"]["full_name"]


def validate_title_event(event: dict[str, Any]) -> None:
    """Validate the issue or pull-request title carried by one webhook event."""

    item = event.get("pull_request") or event.get("issue") or {}
    error = title_format_error(item.get("title"))
    if error:
        print(f"::error::{error}")
        raise SystemExit(1)


def actor_can_manage(client: GitHubClient, repo: str, actor: str | None) -> bool:
    if not actor:
        return False
    if actor == "github-actions[bot]":
        return True
    result = client.request(
        f"repos/{repo}/collaborators/{quote(actor, safe='')}/permission",
        ignore_not_found=True,
    )
    return (result or {}).get("permission") in MAINTAINER_PERMISSIONS


def add_labels(client: GitHubClient, repo: str, number: int, labels: set[str]) -> None:
    if labels:
        client.request(
            f"repos/{repo}/issues/{number}/labels",
            method="POST",
            payload={"labels": sorted(labels)},
        )


def remove_labels(
    client: GitHubClient, repo: str, number: int, labels: set[str]
) -> None:
    for label in sorted(labels):
        client.request(
            f"repos/{repo}/issues/{number}/labels/{quote(label, safe='')}",
            method="DELETE",
            ignore_not_found=True,
        )


def comment_once(
    client: GitHubClient, repo: str, number: int, code: str, message: str
) -> None:
    marker = f"<!-- community-lifecycle:{code} -->"
    comments = (
        client.request(f"repos/{repo}/issues/{number}/comments?per_page=100") or []
    )
    if any(marker in (comment.get("body") or "") for comment in comments):
        return
    client.request(
        f"repos/{repo}/issues/{number}/comments",
        method="POST",
        payload={"body": f"{marker}\n{message}"},
    )


def sync_issue_event(client: GitHubClient, event: dict[str, Any]) -> None:
    repo = repository_name(event)
    number = int(event["issue"]["number"])
    issue = client.request(f"repos/{repo}/issues/{number}")
    event_label = (event.get("label") or {}).get("name")
    sender = (event.get("sender") or {}).get("login")
    plan = plan_issue(
        issue,
        event_action=event.get("action", ""),
        event_label=event_label,
        actor_can_manage=actor_can_manage(client, repo, sender),
    )

    remove_labels(client, repo, number, plan.remove_labels)
    add_labels(client, repo, number, plan.add_labels)
    if plan.clear_milestone:
        client.request(
            f"repos/{repo}/issues/{number}",
            method="PATCH",
            payload={"milestone": None},
        )
    if plan.remove_assignees:
        client.request(
            f"repos/{repo}/issues/{number}/assignees",
            method="DELETE",
            payload={"assignees": sorted(plan.remove_assignees)},
        )
    for code, message in plan.comments:
        comment_once(client, repo, number, code, message)


def sync_issue_kind_event(client: GitHubClient, event: dict[str, Any]) -> None:
    """Synchronize only structural kind metadata for a title-only edit."""

    repo = repository_name(event)
    number = int(event["issue"]["number"])
    issue = client.request(f"repos/{repo}/issues/{number}")
    plan = plan_issue_kind(issue)
    remove_labels(client, repo, number, plan.remove_labels)
    add_labels(client, repo, number, plan.add_labels)


def accept_issue_event(client: GitHubClient, event: dict[str, Any]) -> None:
    """Apply an explicit acceptance transition requested by ``/accept``."""

    command = ((event.get("comment") or {}).get("body") or "").strip()
    if command != "/accept":
        print("::error::The acceptance command must be exactly `/accept`.")
        raise SystemExit(1)

    repo = repository_name(event)
    number = int(event["issue"]["number"])
    actor = (event.get("sender") or {}).get("login")
    issue = client.request(f"repos/{repo}/issues/{number}")
    evaluation = evaluate_issue_acceptance(
        issue,
        actor_can_manage=actor_can_manage(client, repo, actor),
    )
    if not evaluation.valid:
        print(f"::error::{evaluation.error}")
        raise SystemExit(1)

    add_labels(client, repo, number, {ACCEPTED})
    remove_labels(client, repo, number, {NEEDS_ACCEPTANCE})


def linked_issues_for_pull_request(
    client: GitHubClient, repo: str, pull_request: dict[str, Any]
) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    number = int(pull_request["number"])
    query = """
      query ($owner: String!, $repo: String!, $number: Int!) {
        repository(owner: $owner, name: $repo) {
          pullRequest(number: $number) {
            closingIssuesReferences(first: 20) { nodes { number } }
          }
        }
      }
    """
    result = client.request(
        "graphql",
        method="POST",
        payload={
            "query": query,
            "variables": {"owner": owner, "repo": name, "number": number},
        },
    )
    repository = ((result or {}).get("data") or {}).get("repository") or {}
    pull_request_data = repository.get("pullRequest") or {}
    closing_references = pull_request_data.get("closingIssuesReferences") or {}
    nodes = closing_references.get("nodes") or []
    issue_numbers = {int(node["number"]) for node in nodes}
    issue_numbers.update(
        extract_related_issue_numbers(
            pull_request.get("title"),
            pull_request.get("body"),
            repository=repo,
        )
    )
    issues = []
    for issue_number in sorted(issue_numbers):
        issue = client.request(
            f"repos/{repo}/issues/{issue_number}",
            ignore_not_found=True,
        )
        if issue and not issue.get("pull_request"):
            issues.append(issue)
    return issues


def pull_request_signals(
    client: GitHubClient, repo: str, number: int
) -> dict[str, str | None]:
    owner, name = repo.split("/", 1)
    query = """
      query ($owner: String!, $repo: String!, $number: Int!) {
        repository(owner: $owner, name: $repo) {
          pullRequest(number: $number) {
            reviewDecision
            mergeStateStatus
            commits(last: 1) {
              nodes { commit { statusCheckRollup { state } } }
            }
          }
        }
      }
    """
    result = client.request(
        "graphql",
        method="POST",
        payload={
            "query": query,
            "variables": {"owner": owner, "repo": name, "number": number},
        },
    )
    repository = ((result or {}).get("data") or {}).get("repository") or {}
    pull_request = repository.get("pullRequest") or {}
    commits = (pull_request.get("commits") or {}).get("nodes") or []
    commit = (commits[-1] if commits else {}).get("commit") or {}
    rollup = commit.get("statusCheckRollup") or {}
    return {
        "review_decision": pull_request.get("reviewDecision"),
        "merge_state": pull_request.get("mergeStateStatus"),
        "checks_state": rollup.get("state"),
    }


def validate_pull_request_event(client: GitHubClient, event: dict[str, Any]) -> None:
    pull_request = event["pull_request"]
    if (
        pull_request.get("draft")
        or (pull_request.get("user") or {}).get("type") == "Bot"
    ):
        return
    repo = repository_name(event)
    linked = linked_issues_for_pull_request(client, repo, pull_request)
    evaluation = evaluate_pull_request(pull_request, linked)
    if not evaluation.valid:
        for error in evaluation.errors:
            print(f"::error::{error}")
        raise SystemExit(1)


def sync_pull_request(
    client: GitHubClient,
    repo: str,
    number: int,
    *,
    validation_succeeded: bool = True,
) -> None:
    pull_request = client.request(f"repos/{repo}/pulls/{number}", ignore_not_found=True)
    if not pull_request:
        return
    linked = linked_issues_for_pull_request(client, repo, pull_request)
    signals = pull_request_signals(client, repo, number)
    evaluation = evaluate_pull_request(
        pull_request,
        linked,
        validation_succeeded=validation_succeeded,
        **signals,
    )

    current_labels = label_names(pull_request)
    desired_state = (
        {evaluation.state_label} if pull_request.get("state") == "open" else set()
    )
    remove = current_labels.intersection(PR_STATE_LABELS) - desired_state
    add = desired_state - current_labels

    current_workgroups = current_labels.intersection(WORKGROUP_LABELS)
    desired_workgroups = {evaluation.owner_label} if evaluation.owner_label else set()
    remove.update(current_workgroups - desired_workgroups)
    add.update(desired_workgroups - current_workgroups)
    if evaluation.release_blocker:
        add.add(RELEASE_BLOCKER)
    else:
        remove.add(RELEASE_BLOCKER)

    remove_labels(client, repo, number, remove)
    add_labels(client, repo, number, add)
    current_milestone = pull_request.get("milestone") or {}
    current_milestone_number = current_milestone.get("number")
    if current_milestone_number != evaluation.milestone_number:
        client.request(
            f"repos/{repo}/issues/{number}",
            method="PATCH",
            payload={"milestone": evaluation.milestone_number},
        )


def sync_pull_request_event(client: GitHubClient, event: dict[str, Any]) -> None:
    run = event.get("workflow_run") or {}
    check_suite = event.get("check_suite") or {}
    pull_requests = run.get("pull_requests") or check_suite.get("pull_requests") or []
    if not pull_requests:
        return
    sync_pull_request(
        client,
        repository_name(event),
        int(pull_requests[0]["number"]),
        validation_succeeded=(run.get("conclusion") == "success" if run else True),
    )


def sync_open_pull_requests(client: GitHubClient, repo: str) -> None:
    page = 1
    while True:
        pull_requests = (
            client.request(
                f"repos/{repo}/pulls?state=open&per_page={API_PAGE_SIZE}&page={page}"
            )
            or []
        )
        for pull_request in pull_requests:
            sync_pull_request(client, repo, int(pull_request["number"]))
        if len(pull_requests) < API_PAGE_SIZE:
            return
        page += 1


def sync_labeled_closed_pull_requests(client: GitHubClient, repo: str) -> None:
    """Strip in-flight state labels from pull requests that already closed.

    A closed pull request appears in these listings only while it still
    carries a ``pr/*`` state label, so the walk shrinks to one near-empty
    request per state label once the backlog is clean. The numbers are
    collected before any write because removing a label while paging
    through its own listing would shift the pages under the walk.
    """

    numbers: set[int] = set()
    for label in sorted(PR_STATE_LABELS):
        page = 1
        while True:
            issues = (
                client.request(
                    f"repos/{repo}/issues?state=closed"
                    f"&labels={quote(label, safe='')}"
                    f"&per_page={API_PAGE_SIZE}&page={page}"
                )
                or []
            )
            numbers.update(
                int(issue["number"]) for issue in issues if "pull_request" in issue
            )
            if len(issues) < API_PAGE_SIZE:
                break
            page += 1
    for number in sorted(numbers):
        sync_pull_request(client, repo, number)


def sync_pull_request_queue(client: GitHubClient, event: dict[str, Any]) -> None:
    repo = repository_name(event)
    sync_open_pull_requests(client, repo)
    sync_labeled_closed_pull_requests(client, repo)
