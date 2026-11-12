"""GitHub API adapters for community lifecycle policy enforcement."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlparse

from community_lifecycle_policy import (
    ACCEPTED,
    API_PAGE_SIZE,
    ISSUE_DELIVERY_STATE_LABELS,
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

# gh prints "message (HTTP 502)" for a JSON error and a bare "HTTP 502" otherwise.
SERVER_ERROR = re.compile(
    r"(?:\(HTTP |^gh: HTTP )(5[0-9]{2})\)?\s*\Z", flags=re.MULTILINE
)
RETRY_DELAYS_SECONDS: tuple[int, ...] = (5, 15, 45)
RETRY_BUDGET_SECONDS = 120

# Requests the acceptance sweep leaves to every other workflow sharing the
# repository's Actions token, whose hourly budget is about a thousand.
API_RESERVE_REQUESTS = 300


def is_read(method: str, endpoint: str, payload: dict[str, Any] | None) -> bool:
    """A GET, or a GraphQL query, which travels as a POST but only reads."""

    if method == "GET":
        return True
    query = (payload or {}).get("query") or ""
    return endpoint == "graphql" and query.lstrip().startswith("query")


class GitHubClient:
    """Call ``gh api`` and remember the token's remaining REST budget."""

    retry_budget_seconds = RETRY_BUDGET_SECONDS

    def __init__(self) -> None:
        self.rate_limit_remaining: int | None = None

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        ignore_not_found: bool = False,
    ) -> Any:
        command = ["gh", "api", endpoint, "--method", method, "--include"]
        if payload is not None:
            command.extend(["--input", "-"])
        read = is_read(method, endpoint, payload)
        delays = iter(RETRY_DELAYS_SECONDS)
        while True:
            result = subprocess.run(
                command,
                input=json.dumps(payload) if payload is not None else None,
                text=True,
                capture_output=True,
                check=False,
            )
            headers, body = split_http_response(result.stdout)
            remaining = headers.get("x-ratelimit-remaining", "")
            if remaining.isdigit():
                self.rate_limit_remaining = int(remaining)
            if result.returncode == 0:
                return json.loads(body) if body.strip() else None
            if ignore_not_found and "HTTP 404" in result.stderr:
                return None
            server_error = SERVER_ERROR.search(result.stderr[-256:])
            delay = next(delays, None)
            # A write that failed with a 5xx may already have landed, so only reads wait.
            if (
                not read
                or server_error is None
                or delay is None
                or delay > self.retry_budget_seconds
            ):
                sys.stderr.write(result.stderr)
                raise SystemExit(result.returncode)
            print(
                f"::warning::GitHub returned HTTP {server_error[1]} for a read;"
                f" retrying in {delay} seconds."
            )
            self.retry_budget_seconds -= delay
            time.sleep(delay)


def split_http_response(output: str) -> tuple[dict[str, str], str]:
    """Split ``gh api --include`` output into lower-cased headers and the body."""

    if not output.startswith("HTTP/"):
        return {}, output
    ends = [
        (index, len(separator))
        for separator in ("\r\n\r\n", "\n\n")
        if (index := output.find(separator)) >= 0
    ]
    index, length = min(ends) if ends else (len(output), 0)
    headers: dict[str, str] = {}
    for line in output[:index].splitlines()[1:]:
        name, separator, value = line.partition(":")
        if separator:
            headers[name.strip().lower()] = value.strip()
    return headers, output[index + length :]


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


def sync_issue(
    client: GitHubClient,
    repo: str,
    number: int,
    *,
    event_action: str = "",
    event_label: str | None = None,
    actor_can_manage: bool = False,
) -> None:
    """Fetch one issue and apply its current label plan."""

    issue = client.request(f"repos/{repo}/issues/{number}", ignore_not_found=True)
    if not issue:
        return
    plan = plan_issue(
        issue,
        event_action=event_action,
        event_label=event_label,
        actor_can_manage=actor_can_manage,
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


def sync_issue_event(client: GitHubClient, event: dict[str, Any]) -> None:
    repo = repository_name(event)
    number = int(event["issue"]["number"])
    event_label = (event.get("label") or {}).get("name")
    sender = (event.get("sender") or {}).get("login")
    sync_issue(
        client,
        repo,
        number,
        event_action=event.get("action", ""),
        event_label=event_label,
        actor_can_manage=actor_can_manage(client, repo, sender),
    )


def sync_labeled_closed_issues(client: GitHubClient, repo: str) -> None:
    """Sweep closed issues that still carry a stale delivery-state label."""

    numbers: set[int] = set()
    for label in sorted(ISSUE_DELIVERY_STATE_LABELS):
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
                int(issue["number"]) for issue in issues if "pull_request" not in issue
            )
            if len(issues) < API_PAGE_SIZE:
                break
            page += 1
    for number in sorted(numbers):
        sync_issue(client, repo, number)


def sync_issue_queue(client: GitHubClient, event: dict[str, Any]) -> None:
    sync_labeled_closed_issues(client, repository_name(event))


def sync_issue_kind_event(client: GitHubClient, event: dict[str, Any]) -> None:
    """Synchronize only structural kind metadata for a title-only edit."""

    repo = repository_name(event)
    number = int(event["issue"]["number"])
    issue = client.request(f"repos/{repo}/issues/{number}")
    plan = plan_issue_kind(issue)
    remove_labels(client, repo, number, plan.remove_labels)
    add_labels(client, repo, number, plan.add_labels)


def issue_acceptance_actors(client: GitHubClient, repo: str, number: int) -> set[str]:
    """Return the authors of every exact ``/accept`` comment on one issue."""

    actors: set[str] = set()
    page = 1
    while True:
        comments = (
            client.request(
                f"repos/{repo}/issues/{number}/comments"
                f"?per_page={API_PAGE_SIZE}&page={page}"
            )
            or []
        )
        for comment in comments:
            actor = (comment.get("user") or {}).get("login")
            if comment.get("body") == "/accept" and actor:
                actors.add(actor)
        if len(comments) < API_PAGE_SIZE:
            return actors
        page += 1


def linked_open_pull_requests(
    client: GitHubClient, repo: str, number: int
) -> list[int]:
    """List this repository's open pull requests that reference one issue."""

    numbers: set[int] = set()
    page = 1
    while True:
        events = (
            client.request(
                f"repos/{repo}/issues/{number}/timeline"
                f"?per_page={API_PAGE_SIZE}&page={page}"
            )
            or []
        )
        for event in events:
            source = (event.get("source") or {}).get("issue") or {}
            if (
                event.get("event") == "cross-referenced"
                and source.get("pull_request")
                and source.get("state") == "open"
                and urlparse(source.get("repository_url") or "").path
                == f"/repos/{repo}"
            ):
                numbers.add(int(source["number"]))
        if len(events) < API_PAGE_SIZE:
            return sorted(numbers)
        page += 1


def acceptance_actor_can_manage(
    client: GitHubClient,
    repo: str,
    actors: set[str],
    permission_cache: dict[str, bool],
) -> bool:
    """Return whether any command actor is authorized, failing closed on API errors."""

    permission_failed = False
    for actor in sorted(actors):
        try:
            if actor not in permission_cache:
                permission_cache[actor] = actor_can_manage(client, repo, actor)
            if permission_cache[actor]:
                return True
        except (Exception, SystemExit):
            permission_failed = True
    if permission_failed:
        raise RuntimeError(
            "Could not verify every `/accept` actor's repository permission."
        )
    return False


def reconcile_acceptance_candidate(
    client: GitHubClient,
    repo: str,
    number: int,
    actors: set[str],
    permission_cache: dict[str, bool],
) -> str:
    """Fetch, validate, and reconcile one deduplicated acceptance candidate."""

    issue = client.request(f"repos/{repo}/issues/{number}", ignore_not_found=True)
    if not issue:
        print(
            f"::warning title=Issue #{number} skipped::"
            "The issue is no longer available."
        )
        return "skipped"
    if issue.get("pull_request"):
        print(
            f"::notice title=Issue #{number} skipped::"
            "Exact `/accept` comments on pull requests do not accept issues."
        )
        return "skipped"

    labels = label_names(issue)
    if ACCEPTED in labels and NEEDS_ACCEPTANCE not in labels:
        print(
            f"::notice title=Issue #{number} already synchronized::"
            f"{ACCEPTED} is present and {NEEDS_ACCEPTANCE} is absent."
        )
        return "synchronized"

    evaluation = evaluate_issue_acceptance(
        issue,
        actor_can_manage=acceptance_actor_can_manage(
            client,
            repo,
            actors,
            permission_cache,
        ),
    )
    if not evaluation.valid:
        print(f"::warning title=Issue #{number} not accepted::{evaluation.error}")
        return "skipped"

    add_labels(client, repo, number, {ACCEPTED})
    remove_labels(client, repo, number, {NEEDS_ACCEPTANCE})
    print(
        f"::notice title=Issue #{number} accepted::"
        f"Added {ACCEPTED} and removed {NEEDS_ACCEPTANCE}."
    )
    return "accepted"


def reconciliation_failure_message(error: BaseException) -> str:
    if isinstance(error, SystemExit):
        return "A GitHub API request failed; see the preceding `gh api` error."
    return str(error) or "A GitHub API request failed."


def accept_issue_event(client: GitHubClient, event: dict[str, Any]) -> None:
    """Reconcile the commented issue, then the open pull requests that link it."""

    comment = event.get("comment") or {}
    if (comment.get("body") or "") != "/accept":
        print("::error::The acceptance command must be exactly `/accept`.")
        raise SystemExit(1)

    repo = repository_name(event)
    number = int(event["issue"]["number"])
    # Read the decisions back instead of trusting the payload, so an `/accept`
    # edited or deleted before this job runs is withdrawn.
    actors = issue_acceptance_actors(client, repo, number)
    if not actors:
        print(
            f"::notice title=Issue #{number} skipped::"
            "No exact `/accept` comment remains on the issue."
        )
        return
    result = reconcile_acceptance_candidate(client, repo, number, actors, {})
    if result == "skipped":
        return

    failures = 0
    for pull_request in linked_open_pull_requests(client, repo, number):
        try:
            sync_pull_request(client, repo, pull_request)
        except (Exception, SystemExit) as error:
            failures += 1
            print(
                f"::error title=Pull request #{pull_request} sync failed::"
                f"{reconciliation_failure_message(error)}"
            )
    if failures:
        raise SystemExit(1)


def acceptance_sweep_candidates(
    client: GitHubClient, repo: str, since: str | None
) -> list[int]:
    """List open ``needs-acceptance`` issues with comments, newest activity first.

    The numbers are collected before any write because accepting an issue
    removes it from the listing being paged.
    """

    query = (
        f"repos/{repo}/issues?state=open&labels={quote(NEEDS_ACCEPTANCE, safe='')}"
        f"&sort=updated&direction=desc&per_page={API_PAGE_SIZE}"
    )
    if since:
        query += f"&since={quote(since, safe='')}"
    numbers: list[int] = []
    page = 1
    while True:
        issues = client.request(f"{query}&page={page}") or []
        numbers.extend(
            int(issue["number"])
            for issue in issues
            if "pull_request" not in issue and issue.get("comments")
        )
        if len(issues) < API_PAGE_SIZE:
            return numbers
        page += 1


def sweep_issue_acceptance(
    client: GitHubClient,
    repo: str,
    *,
    since: str | None,
    reserve: int,
) -> None:
    """Backfill missed ``/accept`` decisions without spending the API budget.

    Each candidate costs a few requests, so the sweep stops before one once
    the token's remaining budget reaches ``reserve``. Accepted issues leave
    ``needs-acceptance``, so the next run resumes with the rest.
    """

    candidates = acceptance_sweep_candidates(client, repo, since)
    permission_cache: dict[str, bool] = {}
    results = {"accepted": 0, "synchronized": 0, "skipped": 0, "uncommanded": 0}
    failures = 0
    window = f"updated since {since}" if since else "open"
    print(f"::notice::Found {len(candidates)} {window} issue(s) awaiting acceptance.")
    for index, number in enumerate(candidates):
        remaining = client.rate_limit_remaining
        if remaining is not None and remaining <= reserve:
            print(
                f"::notice::Stopped with {remaining} API requests left "
                f"(reserve {reserve}); {len(candidates) - index} issue(s) "
                "remain for the next run."
            )
            break
        try:
            actors = issue_acceptance_actors(client, repo, number)
            if not actors:
                results["uncommanded"] += 1
                continue
            result = reconcile_acceptance_candidate(
                client,
                repo,
                number,
                actors,
                permission_cache,
            )
            results[result] += 1
        except (Exception, SystemExit) as error:
            failures += 1
            print(
                f"::error title=Issue #{number} reconciliation failed::"
                f"{reconciliation_failure_message(error)}"
            )

    print(
        "::notice::Acceptance sweep complete: "
        f"{results['accepted']} accepted, "
        f"{results['synchronized']} already synchronized, "
        f"{results['skipped']} skipped, "
        f"{results['uncommanded']} without `/accept`, {failures} failed."
    )
    if failures:
        raise SystemExit(1)


def accept_issue_queue(client: GitHubClient, event: dict[str, Any]) -> None:
    """Run the scheduled or manual acceptance backfill."""

    lookback_hours = int(os.environ.get("ACCEPTANCE_SWEEP_LOOKBACK_HOURS") or "0")
    since = None
    if lookback_hours > 0:
        start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        since = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    sweep_issue_acceptance(
        client,
        repository_name(event),
        since=since,
        reserve=int(
            os.environ.get("COMMUNITY_API_RESERVE") or str(API_RESERVE_REQUESTS)
        ),
    )


PULL_REQUEST_QUERY = """
  query ($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        reviewDecision
        mergeStateStatus
        closingIssuesReferences(first: 20) { nodes { number } }
        commits(last: 1) {
          nodes { commit { statusCheckRollup { state } } }
        }
      }
    }
  }
"""


def pull_request_graph(client: GitHubClient, repo: str, number: int) -> dict[str, Any]:
    """Read the closing references and review signals of one pull request."""

    owner, name = repo.split("/", 1)
    result = client.request(
        "graphql",
        method="POST",
        payload={
            "query": PULL_REQUEST_QUERY,
            "variables": {"owner": owner, "repo": name, "number": number},
        },
    )
    repository = ((result or {}).get("data") or {}).get("repository") or {}
    return repository.get("pullRequest") or {}


def linked_issues_for_pull_request(
    client: GitHubClient,
    repo: str,
    pull_request: dict[str, Any],
    graph: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if graph is None:
        graph = pull_request_graph(client, repo, int(pull_request["number"]))
    closing_references = graph.get("closingIssuesReferences") or {}
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


def pull_request_signals(graph: dict[str, Any]) -> dict[str, str | None]:
    commits = (graph.get("commits") or {}).get("nodes") or []
    commit = (commits[-1] if commits else {}).get("commit") or {}
    rollup = commit.get("statusCheckRollup") or {}
    return {
        "review_decision": graph.get("reviewDecision"),
        "merge_state": graph.get("mergeStateStatus"),
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
    graph = pull_request_graph(client, repo, number)
    linked = linked_issues_for_pull_request(client, repo, pull_request, graph)
    signals = pull_request_signals(graph)
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

    remove_labels(client, repo, number, remove & current_labels)
    add_labels(client, repo, number, add - current_labels)
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
    # Review-only Community runs skip validation and still carry fresh review
    # signals. Reconciliation independently validates the current linked issues.
    review_signal = (
        run.get("event") == "pull_request_review" and run.get("conclusion") == "skipped"
    )
    sync_pull_request(
        client,
        repository_name(event),
        int(pull_requests[0]["number"]),
        validation_succeeded=(
            not run or run.get("conclusion") == "success" or review_signal
        ),
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
