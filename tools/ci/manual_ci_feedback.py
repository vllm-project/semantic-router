"""Best-effort feedback for a verified CI command comment, never authorization."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

PAGE_SIZE = 100
PAGE_LIMIT = 20
BOT_LOGIN = "github-actions[bot]"
OPERATIONS = frozenset({"comment-lookup", "reaction", "comment", "reaction-cleanup"})
FAILURES = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    subprocess.TimeoutExpired,
)
MESSAGES = {
    "permission": (
        "This request cannot authorize CI. Ask a committer with write, maintain or "
        "admin access to post `/ci run`."
    ),
    "syntax": (
        "This command was not accepted. Use `/ci run` for affected checks, "
        "`/ci run full` for full CPU qualification, or `/ci run <scope>` with "
        "`affected`, `models`, `storage`, `product`, `e2e`, `operator`, or "
        "`performance`. Multiple scopes may be separated by spaces."
    ),
    "edited": "The command changed after creation. Post a new, unedited `/ci run` comment.",
    "expired": (
        "The PR revision or authorization changed. Wait for the current PR's Plan "
        "job, then ask a committer to post a new `/ci run` comment."
    ),
    "revision": (
        "CI requests need an open PR targeting `main` with an available merge "
        "revision. Update the branch onto current `main` or resolve conflicts, "
        "wait for its Plan job, then request CI again."
    ),
    "plan": "The current PR's CI plan is not ready. Wait for its Plan job, then post a new `/ci run` comment.",
    "busy": "Automatic CI is still running. After it finishes, rerun this command handler or post a new `/ci run` comment.",
    "trigger-failed": (
        "I could not confirm that the CI rerun was started. Check the PR CI run "
        "before retrying this command handler or posting a new `/ci run` comment."
    ),
    "controller": (
        "The command handler could not transfer its authorization evidence. "
        "Check the handler logs, then rerun it or post a new `/ci run` comment."
    ),
    "failed": "The command handler could not process this request. Check its logs, then retry or ask a committer for help.",
}


def positive_id(value) -> bool:
    return type(value) is int and value > 0


def command_event(event: dict) -> bool:
    """Only the exact trusted creation event may identify a feedback target."""
    try:
        issue, comment = event.get("issue", {}), event.get("comment", {})
        actor = event.get("sender", {}).get("login")
        return bool(
            os.environ.get("GITHUB_EVENT_NAME") == "issue_comment"
            and event.get("action") == "created"
            and event.get("repository", {}).get("full_name")
            == os.environ["GITHUB_REPOSITORY"]
            and issue.get("pull_request")
            and positive_id(issue.get("number"))
            and positive_id(comment.get("id"))
            and isinstance(actor, str)
            and actor
            and actor == comment.get("user", {}).get("login")
            and isinstance(comment.get("body"), str)
            and comment["body"].startswith("/ci run")
        )
    except (AttributeError, KeyError, TypeError):
        return False


def warning(operation: str, error: Exception) -> None:
    # API stderr and exception text can contain untrusted data or credentials.
    label = operation if operation in OPERATIONS else "feedback"
    status = getattr(error, "http_status", None)
    diagnostic = (
        f"HTTP {status}"
        if type(status) is int and re.fullmatch(r"[45][0-9]{2}", str(status))
        else "HTTP status unavailable"
    )
    print(
        f"::warning::Could not update CI command feedback ({label}; {diagnostic}); "
        "authorization and execution are unchanged."
    )


def feedback_for_event(client, event: dict, *, record: dict | None = None):
    if not command_event(event):
        return None
    repo = os.environ["GITHUB_REPOSITORY"]
    number, comment_id = event["issue"]["number"], event["comment"]["id"]
    actor = event["sender"]["login"]
    if record is not None and (
        not isinstance(record, dict)
        or record.get("repository") != repo
        or record.get("pr_number") != number
        or record.get("comment_id") != comment_id
        or record.get("actor") != actor
    ):
        return None
    try:
        comment = client.request(f"repos/{repo}/issues/comments/{comment_id}")
        if (
            comment.get("id") != comment_id
            or comment.get("issue_url")
            != f"https://api.github.com/repos/{repo}/issues/{number}"
            or comment.get("user", {}).get("login") != actor
        ):
            return None
    except FAILURES as error:
        warning("comment-lookup", error)
        return None
    return CommandFeedback(client, repo, number, comment_id)


def bot_owned(item: dict) -> bool:
    return (
        item.get("user", {}).get("login") == BOT_LOGIN
        and item.get("user", {}).get("type") == "Bot"
    )


@dataclass
class CommandFeedback:
    client: object
    repo: str
    number: int
    comment_id: int

    def items(self, endpoint: str):
        for page in range(1, PAGE_LIMIT + 1):
            result = self.client.request(f"{endpoint}?per_page={PAGE_SIZE}&page={page}")
            if not isinstance(result, list):
                raise ValueError("Invalid feedback collection")
            yield from result
            if len(result) < PAGE_SIZE:
                return
        raise ValueError("Feedback collection exceeds limit")

    def reaction(self, content: str) -> None:
        try:
            # GitHub returns the existing reaction for the same user/content.
            self.client.write(
                f"repos/{self.repo}/issues/comments/{self.comment_id}/reactions",
                {"content": content},
            )
        except FAILURES as error:
            warning("reaction", error)

    def comment(self, message: str, *, create: bool) -> None:
        marker = f"<!-- semantic-router-ci-command:{self.comment_id} -->"
        request_url = f"https://github.com/{self.repo}/pull/{self.number}#issuecomment-{self.comment_id}"
        body = f"{marker}\nFor [this CI request]({request_url}): {message}\n"
        controller = os.environ.get("GITHUB_RUN_ID", "")
        if controller.isdecimal():
            body += f"\n[Command handler logs](https://github.com/{self.repo}/actions/runs/{controller}).\n"
        try:
            owned = [
                item
                for item in self.items(
                    f"repos/{self.repo}/issues/{self.number}/comments"
                )
                if bot_owned(item)
                and item.get("body", "").startswith(marker + "\n")
                and positive_id(item.get("id"))
            ]
            if owned:
                latest = max(owned, key=lambda item: item["id"])
                if latest.get("body") != body:
                    self.client.write(
                        f"repos/{self.repo}/issues/comments/{latest['id']}",
                        {"body": body},
                        method="PATCH",
                    )
            elif create:
                self.client.write(
                    f"repos/{self.repo}/issues/{self.number}/comments", {"body": body}
                )
        except FAILURES as error:
            warning("comment", error)

    def received(self) -> None:
        self.reaction("eyes")

    def rejected(self, reason: str, *, run_id=None) -> None:
        self.reaction("confused")
        message = MESSAGES.get(reason, MESSAGES["failed"])
        if positive_id(run_id):
            message += (
                f" [PR CI run](https://github.com/{self.repo}/actions/runs/{run_id})."
            )
        self.comment(message, create=True)

    def started(self, run_id: int) -> None:
        self.reaction("rocket")
        try:
            endpoint = f"repos/{self.repo}/issues/comments/{self.comment_id}/reactions"
            for reaction in self.items(endpoint):
                if (
                    bot_owned(reaction)
                    and reaction.get("content") == "confused"
                    and positive_id(reaction.get("id"))
                ):
                    self.client.request(f"{endpoint}/{reaction['id']}", method="DELETE")
        except FAILURES as error:
            warning("reaction-cleanup", error)
        self.comment(
            "Latest result: "
            f"[CI rerun requested](https://github.com/{self.repo}/actions/runs/{run_id}). "
            "GitHub has accepted the request; jobs may still be queued. This does not mean tests passed.",
            create=False,
        )
