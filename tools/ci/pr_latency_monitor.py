#!/usr/bin/env python3
"""Read-only PR/Actions monitor with deduplicated Slack incident alerts.

Run periodically on an independent host. GitHub CLI supplies authentication;
the Slack webhook lives in a file outside the checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

MAX_PAGES = 5
PAGE_SIZE = 100
QUEUE_MARKER = "-*- Mergify Payload -*-"
QUEUE_END = "-*- Mergify Payload End -*-"
REPO = "vllm-project/semantic-router"
RUNBOOK_URL = (
    "https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md"
    "#troubleshoot-a-pr-workflow-with-no-jobs"
)
REQUIRED_APPROVALS = 2
ZERO_JOB_MINUTES = 10
MISSING_GATE_MINUTES = 10
RECENT_DEQUEUE_MINUTES = 10
QUEUE_ADMISSION_MINUTES = 10
QUEUE_NO_CHECK_MINUTES = 20
QUEUE_AGE_MINUTES = 90
MAX_ALERT_EXAMPLES = 10
HTTP_OK = 200
SECONDS_PER_MINUTE = 60


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def minutes_since(value: str, now: datetime) -> float:
    return (now - timestamp(value)).total_seconds() / SECONDS_PER_MINUTE


class GitHub:
    def get(self, endpoint: str) -> dict | list:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"GitHub API failed for {endpoint}: {result.stderr.strip()}"
            )
        return json.loads(result.stdout)

    def pages(self, endpoint: str, key: str | None = None):
        separator = "&" if "?" in endpoint else "?"
        for page in range(1, MAX_PAGES + 1):
            result = self.get(f"{endpoint}{separator}per_page={PAGE_SIZE}&page={page}")
            rows = result[key] if key else result
            if not isinstance(rows, list):
                raise RuntimeError(f"Invalid GitHub collection: {endpoint}")
            yield from rows
            if len(rows) < PAGE_SIZE:
                return
        raise RuntimeError(f"GitHub collection exceeds {MAX_PAGES} pages: {endpoint}")


def queue_payload(comments: list[dict]) -> dict | None:
    for comment in reversed(comments):
        body = comment.get("body") or ""
        start = body.find(QUEUE_MARKER)
        end = body.find(QUEUE_END)
        if (
            start < 0
            or end < start
            or comment.get("user", {}).get("login") != "mergify[bot]"
        ):
            continue
        try:
            return json.loads(body[start + len(QUEUE_MARKER) : end].strip())
        except json.JSONDecodeError:
            continue
    return None


def current_approvals(reviews: list[dict]) -> list[str]:
    latest: dict[str, str] = {}
    for review in reviews:
        user = (review.get("user") or {}).get("login")
        if user and review.get("state") in {
            "APPROVED",
            "CHANGES_REQUESTED",
            "DISMISSED",
        }:
            latest[user] = review["state"]
    return [user for user, state in latest.items() if state == "APPROVED"]


def gate(checks: list[dict]) -> dict | None:
    matches = [check for check in checks if check.get("name") == "PR Gate"]
    return max(matches, key=lambda check: check.get("id", 0)) if matches else None


def waiting_for_committer(check: dict | None) -> bool:
    if not check or check.get("status") != "in_progress":
        return False
    output = check.get("output") or {}
    return "waiting for committer approval" in (
        f"{output.get('title', '')} {output.get('summary', '')}".lower()
    )


def alerts_for_queued_runs(client: GitHub, repo: str, now: datetime) -> dict[str, str]:
    alerts = {}
    endpoint = f"repos/{repo}/actions/workflows/pr.yml/runs?status=queued"
    latest: dict[tuple[str, str], dict] = {}
    for run in client.pages(endpoint, "workflow_runs"):
        source = (run.get("head_repository") or {}).get("full_name", "")
        branch = run.get("head_branch", "")
        key = (source, branch)
        if key not in latest or run["created_at"] > latest[key]["created_at"]:
            latest[key] = run
    for run in latest.values():
        if (
            run.get("status") != "queued"
            or run.get("event") != "pull_request"
            or minutes_since(run["created_at"], now) < ZERO_JOB_MINUTES
        ):
            continue
        query = urlencode({"event": "pull_request", "head_sha": run["head_sha"]})
        same_head = [
            other
            for other in client.pages(
                f"repos/{repo}/actions/workflows/pr.yml/runs?{query}",
                "workflow_runs",
            )
            if other.get("head_branch") == run.get("head_branch")
            and (other.get("head_repository") or {}).get("full_name")
            == (run.get("head_repository") or {}).get("full_name")
        ]
        if (
            same_head
            and max(same_head, key=lambda item: (item["created_at"], item["id"]))["id"]
            != run["id"]
        ):
            continue
        jobs = client.get(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=1")
        if jobs.get("total_count") == 0:
            owner = (run.get("actor") or {}).get("login", "unknown")
            alerts[f"zero-jobs:{run['id']}"] = (
                f"PR workflow has zero jobs after {minutes_since(run['created_at'], now):.0f} min: "
                f"{run['html_url']} (head {run['head_sha'][:12]}, actor {owner}). "
                "Check GitHub Actions admission and status; the merge queue cannot run this revision yet."
            )
    return alerts


def alerts_for_approved_prs(client: GitHub, repo: str, now: datetime) -> dict[str, str]:
    alerts = {}
    query = urlencode({"q": f"repo:{repo} is:pr is:open review:approved"})
    for item in client.pages(f"search/issues?{query}", "items"):
        number = item["number"]
        pr = client.get(f"repos/{repo}/pulls/{number}")
        if pr.get("draft") or pr.get("state") != "open":
            continue
        reviews = list(client.pages(f"repos/{repo}/pulls/{number}/reviews"))
        approvals = current_approvals(reviews)
        if len(approvals) < REQUIRED_APPROVALS:
            continue
        head = pr["head"]["sha"]
        checks = client.get(
            f"repos/{repo}/commits/{head}/check-runs?check_name=PR%20Gate&per_page=100"
        )["check_runs"]
        pr_gate = gate(checks)
        approved_at = max(
            review["submitted_at"]
            for review in reviews
            if review.get("state") == "APPROVED"
            and (review.get("user") or {}).get("login") in approvals
            and review.get("submitted_at")
        )
        url = pr["html_url"]
        owner = (pr.get("user") or {}).get("login", "unknown")
        if not pr_gate:
            if minutes_since(approved_at, now) >= MISSING_GATE_MINUTES:
                alerts[f"gate-missing:{number}:{head}"] = (
                    f"Approved PR #{number} has no PR Gate after "
                    f"{minutes_since(approved_at, now):.0f} min: {url} (author {owner}). "
                    "Inspect the current-head Pull Request workflow before Mergify."
                )
            continue
        if pr_gate.get("status") != "completed" or waiting_for_committer(pr_gate):
            continue
        if pr_gate.get("conclusion") != "success":
            continue
        ready_at = max(timestamp(approved_at), timestamp(pr_gate["completed_at"]))
        comments = list(client.pages(f"repos/{repo}/issues/{number}/comments"))
        queue = queue_payload(comments)
        if queue and queue.get("state") == "dequeued":
            left_at = queue.get("left_at")
            reason = queue.get("dequeue_reason") or {}
            code = reason.get("code", "")
            if (
                left_at
                and minutes_since(left_at, now) <= RECENT_DEQUEUE_MINUTES
                and code
                in {
                    "INCOMPATIBILITY_WITH_BRANCH_PROTECTIONS",
                    "PULL_REQUEST_CANNOT_BE_UPDATED",
                }
            ):
                alerts[f"queue-dequeued:{number}:{head}:{left_at}"] = (
                    f"PR #{number} was dequeued because {reason.get('label', code)}: "
                    f"{url} (author {owner}). Inspect Mergify's reason and the required PR Gate."
                )
            continue
        queued_at = queue.get("queued_at") if queue else None
        if not queued_at:
            if (
                now - ready_at
            ).total_seconds() >= QUEUE_ADMISSION_MINUTES * SECONDS_PER_MINUTE:
                alerts[f"queue-admission:{number}:{head}"] = (
                    f"Approved PR #{number} has a green PR Gate but has not entered "
                    f"the merge queue after {(now - ready_at).total_seconds() / SECONDS_PER_MINUTE:.0f} min: {url} (author {owner}). "
                    "Inspect Mergify's effective queue conditions and review threads."
                )
            continue
        if queue.get("state") not in {"queued", "testing"}:
            continue
        queue_age = minutes_since(queued_at, now)
        if queue_age >= QUEUE_AGE_MINUTES:
            alerts[f"queue-age:{number}:{head}"] = (
                f"PR #{number} has remained in the merge queue for {queue_age:.0f} min: "
                f"{url} (author {owner}). Check queue status, speculative checks, and dequeue history."
            )
        elif queue_age >= QUEUE_NO_CHECK_MINUTES and not queue.get(
            "speculative_check_pr"
        ):
            alerts[f"queue-no-check:{number}:{head}"] = (
                f"PR #{number} entered the merge queue {queue_age:.0f} min ago "
                f"without a speculative check PR: {url} (author {owner}). Inspect queue activity."
            )
    return alerts


def reconcile(previous: dict[str, str], current: dict[str, str]) -> list[str]:
    new = [message for key, message in current.items() if key not in previous]
    resolved = [message for key, message in previous.items() if key not in current]
    messages = []
    for heading, items in (
        (":warning: PR monitor: new incidents", new),
        (":white_check_mark: PR monitor: recovered incidents", resolved),
    ):
        if items:
            examples = "\n".join(f"• {item}" for item in items[:MAX_ALERT_EXAMPLES])
            remainder = (
                f"\n… and {len(items) - MAX_ALERT_EXAMPLES} more."
                if len(items) > MAX_ALERT_EXAMPLES
                else ""
            )
            messages.append(
                f"{heading} ({len(items)})\n{examples}{remainder}\nRunbook: {RUNBOOK_URL}"
            )
    return messages


def post(webhook: str, message: str) -> None:
    request = Request(
        webhook,
        data=json.dumps({"text": message}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        if response.status != HTTP_OK:
            raise RuntimeError(f"Slack webhook returned HTTP {response.status}")


def save_state(path: Path, alerts: dict[str, str], now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(
            {"checked_at": now.isoformat(), "alerts": alerts}, temporary, indent=2
        )
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--webhook-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Record current incidents without sending",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("--repo must be owner/name")
    if not (args.dry_run or args.bootstrap) and not args.webhook_file:
        parser.error(
            "--webhook-file is required unless --dry-run or --bootstrap is set"
        )

    now = datetime.now(timezone.utc)
    client = GitHub()
    current = alerts_for_queued_runs(client, args.repo, now)
    current.update(alerts_for_approved_prs(client, args.repo, now))
    previous_state = (
        json.loads(args.state_file.read_text()) if args.state_file.exists() else {}
    )
    previous = previous_state.get("alerts", {})
    messages = reconcile(previous, current)
    if args.dry_run:
        for message in messages:
            print(message)
        print(f"{len(current)} active alert(s); state unchanged (dry run)")
        return 0
    if args.bootstrap:
        save_state(args.state_file, current, now)
        print(f"{len(current)} existing alert(s) recorded without notification")
        return 0

    if args.webhook_file.stat().st_mode & 0o077:
        raise RuntimeError(
            "Webhook file must be readable only by its owner (chmod 600)"
        )
    webhook = args.webhook_file.read_text().strip()
    parsed = urlparse(webhook)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "hooks.slack.com"
        or not parsed.path.startswith("/services/")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Webhook file does not contain a Slack incoming webhook URL")
    for message in messages:
        post(webhook, message)
    save_state(args.state_file, current, now)
    print(f"{len(current)} active alert(s); {len(messages)} notification(s) sent")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"PR monitor failed: {error}", file=sys.stderr)
        sys.exit(1)
