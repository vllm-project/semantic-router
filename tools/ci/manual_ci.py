#!/usr/bin/env python3
"""Authorize bounded CI requests for an immutable pull-request merge revision."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode

from manual_ci_feedback import command_event, feedback_for_event

CONTROLLER = ".github/workflows/manual-ci.yml"
AUTHORIZE_JOB = "Authorize requested CI"
# Public immutable GitHub identity: https://api.github.com/users/mergify%5Bbot%5D
MERGIFY_BOT_ID = 37929162
SCOPES = frozenset(
    {"affected", "models", "storage", "product", "e2e", "operator", "performance"}
)
SHA = re.compile(r"[0-9a-f]{40}")
PINNED_REVISION_COUNT = 3
MERGE_PARENT_COUNT = 2
MERGE_EVIDENCE_UNAVAILABLE = "planned_merge_unavailable"
NO_AUTHORIZATION = {"authorized": False, "scopes": [], "full": False}
PAGE_SIZE = 100
AUTHORIZATION_LIMIT = 65536
PLAN_LIMIT = 16 * 1024 * 1024
ERROR_STATUS_TAIL_LIMIT = 256
# A rate limit, and for reads a 5xx or a timeout, is waited out within one
# budget per client, so a job stays inside its five-minute timeout.
RETRY_DELAYS = (5, 15, 45)
RETRY_BUDGET_SECONDS = 120
GH_TIMEOUT_SECONDS = 60
HTTP_SERVER_ERROR = 500
HTTP_NOT_FOUND = 404


def http_error_status(stderr: str | bytes) -> int | None:
    """Retain only gh's bounded HTTP error suffix, never response text.

    gh ends with "(HTTP 503)" after a JSON message, or prints a bare
    "gh: HTTP 503" line when the body is not JSON or carries no message.
    """
    tail = stderr[-ERROR_STATUS_TAIL_LIMIT:]
    if isinstance(tail, bytes):
        tail = tail.decode("ascii", errors="replace")
    match = re.search(
        r"(?:\(HTTP ([45][0-9]{2})\)|^gh: HTTP ([45][0-9]{2}))\s*\Z", tail, re.M
    )
    return int(match[1] or match[2]) if match else None


def rate_limited(status: int | None, stderr: str | bytes) -> bool:
    """Classify a primary or secondary rate limit from gh's text, never echo it."""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return status in {403, 429} and "rate limit" in stderr.lower()


class ManualCIError(RuntimeError):
    """A credential-free failure suitable for a workflow summary."""

    def __init__(
        self, message: str, *, reason: str = "failed", http_status: int | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.http_status = http_status


def api_failure(result, message: str) -> ManualCIError:
    """Name the HTTP status and a rate limit, never gh's response text."""
    status = http_error_status(result.stderr)
    if rate_limited(status, result.stderr):
        message += f" (HTTP {status}, rate limited); rerun after the limit resets."
    elif status in {None, 401, 403, 404}:
        suffix = f" (HTTP {status})" if status else ""
        message += f"{suffix}; check workflow permissions."
    else:
        message += f" (HTTP {status})."
    return ManualCIError(message, http_status=status)


class GitHubClient:
    retry_budget = RETRY_BUDGET_SECONDS

    def gh(self, args: list[str], *, read: bool, **options):
        """Run gh and wait out a transient failure while the budget lasts."""
        for delay in (*RETRY_DELAYS, None):
            try:
                result = subprocess.run(
                    args,
                    capture_output=True,
                    check=False,
                    timeout=GH_TIMEOUT_SECONDS,
                    **options,
                )
            except subprocess.TimeoutExpired:
                self.retry_budget -= GH_TIMEOUT_SECONDS
                if not read or delay is None or delay > self.retry_budget:
                    raise
            else:
                status = http_error_status(result.stderr)
                transient = result.returncode != 0 and (
                    rate_limited(status, result.stderr)
                    or (read and status is not None and status >= HTTP_SERVER_ERROR)
                )
                if not transient or delay is None or delay > self.retry_budget:
                    return result
            self.retry_budget -= delay
            time.sleep(delay)

    def raw(self, endpoint: str, *, method="GET") -> bytes:
        result = self.gh(
            ["gh", "api", endpoint, "--method", method], read=method == "GET"
        )
        if result.returncode:
            raise api_failure(result, "GitHub API request failed")
        return result.stdout

    def request(self, endpoint: str, *, method="GET"):
        value = self.raw(endpoint, method=method)
        return json.loads(value) if value.strip() else None

    def write(self, endpoint: str, payload: dict, *, method="POST"):
        result = self.gh(
            ["gh", "api", endpoint, "--method", method, "--input", "-"],
            read=False,
            input=json.dumps(payload),
            text=True,
        )
        if result.returncode:
            raise api_failure(result, "GitHub feedback API request failed")
        return json.loads(result.stdout) if result.stdout.strip() else None


def parse_command(body: str) -> dict | None:
    """Accept only literal catalog scopes, optionally followed by all three SHAs."""
    if not isinstance(body, str) or not re.fullmatch(r"/ci run(?: [a-z0-9-]+)*", body):
        return None
    tokens = body.split(" ")[2:]
    pinned = None
    if len(tokens) > PINNED_REVISION_COUNT and all(
        SHA.fullmatch(value) for value in tokens[-PINNED_REVISION_COUNT:]
    ):
        pinned, tokens = (
            tokens[-PINNED_REVISION_COUNT:],
            tokens[:-PINNED_REVISION_COUNT],
        )
        if tokens not in (["full"], ["affected"]):
            return None
    full = tokens == ["full"]
    scopes = [] if full else tokens or ["affected"]
    if len(scopes) != len(set(scopes)) or set(scopes) - SCOPES:
        return None
    return {"scopes": scopes, "full": full, "pinned": pinned}


def timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("missing timezone")
        return result
    except (AttributeError, TypeError, ValueError):
        raise ManualCIError("Invalid authorization timestamp.") from None


def main_contains(client, repo: str, base: str) -> bool:
    """A planned base remains trusted when main advances without rewriting it."""
    if not isinstance(base, str) or not SHA.fullmatch(base):
        return False
    try:
        base_ref = client.request(f"repos/{repo}/git/ref/heads/main")
        if (
            not isinstance(base_ref, dict)
            or base_ref.get("ref") != "refs/heads/main"
            or base_ref.get("object", {}).get("type") != "commit"
        ):
            return False
        current_main = base_ref.get("object", {}).get("sha")
        if not isinstance(current_main, str) or not SHA.fullmatch(current_main):
            return False
        if current_main == base:
            return True
        comparison = client.request(f"repos/{repo}/compare/{base}...{current_main}")
        return bool(
            isinstance(comparison, dict)
            and comparison.get("merge_base_commit", {}).get("sha") == base
        )
    except (ManualCIError, KeyError, TypeError, ValueError):
        return False


def source_identity(client, repo: str, number: int) -> dict:
    pull = client.request(f"repos/{repo}/pulls/{number}")
    if (
        pull.get("state") != "open"
        or pull.get("base", {}).get("ref") != "main"
        or pull.get("base", {}).get("repo", {}).get("full_name") != repo
    ):
        raise ManualCIError(
            "CI requests require an open pull request targeting main.",
            reason="revision",
        )
    # The PR API's base.sha can lag behind the test merge commit. Read the
    # actual target ref, then require that the merge commit has a valid head
    # and a base that is either the current main tip or an ancestor of main.
    base_ref = client.request(f"repos/{repo}/git/ref/heads/main")
    if (
        base_ref.get("ref") != "refs/heads/main"
        or base_ref.get("object", {}).get("type") != "commit"
    ):
        raise ManualCIError(
            "The current main revision is unavailable.", reason="revision"
        )
    current_main = base_ref.get("object", {}).get("sha")
    result = {
        "head_sha": pull.get("head", {}).get("sha"),
        "source_sha": pull.get("merge_commit_sha"),
    }
    if (
        not isinstance(current_main, str)
        or not SHA.fullmatch(current_main)
        or any(
            not isinstance(value, str) or not SHA.fullmatch(value)
            for value in result.values()
        )
    ):
        raise ManualCIError(
            "The current PR merge revision is unavailable; resolve conflicts first.",
            reason="revision",
        )
    commit = client.request(f"repos/{repo}/git/commits/{result['source_sha']}")
    parents = commit.get("parents")
    if (
        not isinstance(parents, list)
        or len(parents) != MERGE_PARENT_COUNT
        or any(not isinstance(parent, dict) for parent in parents)
        or parents[1].get("sha") != result["head_sha"]
        or not isinstance(parents[0].get("sha"), str)
        or not SHA.fullmatch(parents[0]["sha"])
    ):
        raise ManualCIError(
            "The PR merge revision changed or is still being prepared; wait for its Plan job.",
            reason="revision",
        )
    merge_base = parents[0]["sha"]
    if not main_contains(client, repo, merge_base):
        raise ManualCIError(
            "The PR merge revision changed or is still being prepared; wait for its Plan job.",
            reason="revision",
        )
    result = {
        "head_sha": result["head_sha"],
        "base_sha": merge_base,
        "source_sha": result["source_sha"],
    }
    return result


def merge_revision_matches(
    client,
    repo: str,
    source: str,
    base: str,
    head: str,
    *,
    equivalent_source: str | None = None,
) -> bool:
    """Verify planned parents and, when required, the replacement merge's tree."""
    revisions = (source, base, head)
    if equivalent_source is not None:
        revisions += (equivalent_source,)
    if any(
        not isinstance(value, str) or not SHA.fullmatch(value) for value in revisions
    ):
        return False
    try:
        commit = client.request(f"repos/{repo}/git/commits/{source}")
    except ManualCIError as error:
        if error.http_status != HTTP_NOT_FOUND:
            raise
        raise ManualCIError(
            "The original planned merge commit is unavailable; its qualification source cannot be verified.",
            reason=MERGE_EVIDENCE_UNAVAILABLE,
            http_status=HTTP_NOT_FOUND,
        ) from None
    parents = commit.get("parents") if isinstance(commit, dict) else None
    if not (
        isinstance(parents, list)
        and len(parents) == MERGE_PARENT_COUNT
        and all(isinstance(parent, dict) for parent in parents)
        and [parent.get("sha") for parent in parents] == [base, head]
    ):
        return False
    if equivalent_source is None:
        return True
    replacement = client.request(f"repos/{repo}/git/commits/{equivalent_source}")
    trees = (
        commit.get("tree"),
        replacement.get("tree") if isinstance(replacement, dict) else None,
    )
    tree_shas = [tree.get("sha") if isinstance(tree, dict) else None for tree in trees]
    return (
        all(isinstance(value, str) and SHA.fullmatch(value) for value in tree_shas)
        and tree_shas[0] == tree_shas[1]
    )


def merge_source_matches(
    client, repo: str, source: str, base: str, identity: dict
) -> bool:
    """Accept a verified equivalent merge or the existing ancestor-base policy."""
    if any(
        not isinstance(value, str) or not SHA.fullmatch(value)
        for value in (source, base)
    ):
        return False
    if source == identity["source_sha"] and base == identity["base_sha"]:
        return True
    same_base = base == identity["base_sha"]
    if not merge_revision_matches(
        client,
        repo,
        source,
        base,
        identity["head_sha"],
        equivalent_source=identity["source_sha"] if same_base else None,
    ):
        return False
    return same_base or main_contains(client, repo, base)


def trusted_queue_pull(pull: dict, repo: str) -> bool:
    """Queue admission already required the original PR's qualification/reviews."""
    author, head, base = (
        pull.get("user", {}),
        pull.get("head", {}),
        pull.get("base", {}),
    )
    return bool(
        author.get("id") == MERGIFY_BOT_ID
        and author.get("login") == "mergify[bot]"
        and author.get("type") == "Bot"
        and (head.get("repo") or {}).get("full_name") == repo
        and (base.get("repo") or {}).get("full_name") == repo
        and base.get("ref") == "main"
        and isinstance(head.get("ref"), str)
        and head["ref"].startswith("mergify/merge-queue/")
    )


def can_write(client, repo: str, actor: str) -> bool:
    if not actor or not isinstance(actor, str):
        return False
    result = client.request(
        f"repos/{repo}/collaborators/{quote(actor, safe='')}/permission"
    )
    return result.get("permission") in {"write", "maintain", "admin"}


def unchanged_comment(client, repo: str, number: int, comment_id: int) -> dict:
    comment = client.request(f"repos/{repo}/issues/comments/{comment_id}")
    if (
        comment.get("id") != comment_id
        or comment.get("issue_url")
        != f"https://api.github.com/repos/{repo}/issues/{number}"
        or comment.get("created_at") != comment.get("updated_at")
    ):
        raise ManualCIError(
            "Use a new, unedited /ci run comment on this pull request.", reason="edited"
        )
    timestamp(comment.get("created_at"))
    return comment


def read_artifact(client, repo: str, artifact: dict, filename: str, limit: int) -> dict:
    if artifact.get("expired") or not 0 < artifact.get("size_in_bytes", 0) <= limit:
        raise ManualCIError(
            "CI authorization or plan artifact is expired or too large."
        )
    raw = client.raw(f"repos/{repo}/actions/artifacts/{int(artifact['id'])}/zip")
    if len(raw) > limit:
        raise ManualCIError("CI artifact exceeds its size limit.")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != filename
                or entries[0].file_size > limit
            ):
                raise ManualCIError("CI artifact has an unexpected file inventory.")
            result = json.loads(archive.read(entries[0]))
            if not isinstance(result, dict):
                raise ValueError("expected object")
            return result
    except (zipfile.BadZipFile, ValueError, KeyError, OSError):
        raise ManualCIError("CI artifact is not a valid bounded JSON object.") from None


def plan_for_run(client, repo: str, run: dict, identity: dict) -> dict | None:
    artifacts = client.request(
        f"repos/{repo}/actions/runs/{run['id']}/artifacts?name=ci-plan&per_page=10"
    ).get("artifacts", [])
    for artifact in artifacts:
        if artifact.get("name") != "ci-plan":
            continue
        plan = read_artifact(client, repo, artifact, "plan.json", PLAN_LIMIT)
        unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
        digest = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if plan.get("plan_sha256") != digest:
            continue
        if run.get("head_sha") == identity["head_sha"] and merge_source_matches(
            client, repo, plan.get("source_sha"), plan.get("base_sha"), identity
        ):
            return plan
    return None


def authorize(client, event: dict) -> dict | None:
    repo = os.environ["GITHUB_REPOSITORY"]
    issue, event_comment = event.get("issue", {}), event.get("comment", {})
    actor = event.get("sender", {}).get("login")
    command = parse_command(event_comment.get("body"))
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "issue_comment"
        or event.get("action") != "created"
        or not issue.get("pull_request")
        or event.get("repository", {}).get("full_name") != repo
        or actor != event_comment.get("user", {}).get("login")
        or command is None
    ):
        return None
    if not can_write(client, repo, actor):
        raise ManualCIError(
            "Only repository write, maintain or admin collaborators may request CI.",
            reason="permission",
        )
    number, comment_id = int(issue["number"]), int(event_comment["id"])
    comment = unchanged_comment(client, repo, number, comment_id)
    if (
        comment.get("body") != event_comment.get("body")
        or comment.get("user", {}).get("login") != actor
    ):
        raise ManualCIError(
            "The CI request changed after its creation event.", reason="edited"
        )
    identity = source_identity(client, repo, number)
    head = client.request(f"repos/{repo}/pulls/{number}")["head"]
    if command["pinned"] and command["pinned"] != list(identity.values()):
        # Verification against the planned run occurs once selected below.
        pass
    query = urlencode(
        {"event": "pull_request", "head_sha": identity["head_sha"], "per_page": 20}
    )
    runs = client.request(f"repos/{repo}/actions/workflows/pr.yml/runs?{query}").get(
        "workflow_runs", []
    )
    for run in sorted(
        runs,
        key=lambda value: (timestamp(value["created_at"]), value["id"]),
        reverse=True,
    ):
        if (
            run.get("event") != "pull_request"
            or run.get("head_sha") != identity["head_sha"]
            or run.get("head_repository", {}).get("full_name")
            != (head.get("repo") or {}).get("full_name")
            or run.get("head_branch") != head.get("ref")
            or (
                run.get("pull_requests")
                and not any(
                    pull.get("number") == number for pull in run["pull_requests"]
                )
            )
        ):
            continue
        # Never fall back to an older run while the newest PR plan is pending.
        # Its rerun would cancel the current run through the PR concurrency group.
        if timestamp(comment["created_at"]) < timestamp(run["created_at"]):
            break
        plan = plan_for_run(client, repo, run, identity)
        if plan is None:
            break
        if command["pinned"] and command["pinned"] not in (
            list(identity.values()),
            [identity["head_sha"], plan["base_sha"], plan["source_sha"]],
        ):
            raise ManualCIError(
                "The requested head, base or merge SHA is no longer current.",
                reason="expired",
            )
        return {
            "schema_version": 1,
            "repository": repo,
            "pr_number": number,
            "head_sha": identity["head_sha"],
            "base_sha": plan["base_sha"],
            "source_sha": plan["source_sha"],
            "pr_run_id": run["id"],
            "pr_run_created_at": run["created_at"],
            "plan_sha256": plan["plan_sha256"],
            "scopes": command["scopes"],
            "full": command["full"],
            "actor": actor,
            "comment_id": comment_id,
            "comment_created_at": comment["created_at"],
            "controller_run_id": int(os.environ["GITHUB_RUN_ID"]),
            "controller_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
            "controller_sha": os.environ["GITHUB_SHA"],
        }
    raise ManualCIError(
        "No current PR CI plan is available; wait for its Plan job, then post a new request.",
        reason="plan",
    )


def trusted_controller(client, repo: str, record: dict, artifact: dict) -> bool:
    run_id = record.get("controller_run_id")
    if (
        not isinstance(run_id, int)
        or artifact.get("workflow_run", {}).get("id") != run_id
    ):
        return False
    run = client.request(f"repos/{repo}/actions/runs/{run_id}")
    workflow = client.request(f"repos/{repo}/actions/workflows/manual-ci.yml")
    controller_sha = record.get("controller_sha", "")
    if (
        run.get("workflow_id") != workflow.get("id")
        or run.get("path") != CONTROLLER
        or run.get("event") != "issue_comment"
        or run.get("head_branch") != "main"
        or run.get("head_sha") != controller_sha
        or not SHA.fullmatch(controller_sha)
        or run.get("head_repository", {}).get("full_name") != repo
        or run.get("actor", {}).get("login") != record.get("actor")
    ):
        return False
    comparison = client.request(f"repos/{repo}/compare/{controller_sha}...main")
    if comparison.get("merge_base_commit", {}).get("sha") != controller_sha:
        return False
    attempt = record.get("controller_run_attempt")
    if not isinstance(attempt, int) or attempt < 1:
        return False
    jobs = client.request(
        f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page={PAGE_SIZE}"
    ).get("jobs", [])
    return any(
        job.get("name") == AUTHORIZE_JOB
        and job.get("status") == "completed"
        and job.get("conclusion") == "success"
        for job in jobs
    )


def matching_authorization(
    client, repo: str, number: int, run: dict, identity: dict, record: dict
) -> bool:
    if (
        record.get("schema_version") != 1
        or record.get("repository") != repo
        or record.get("pr_number") != number
        or record.get("pr_run_id") != run["id"]
        or record.get("pr_run_created_at") != run["created_at"]
        or record.get("head_sha") != identity["head_sha"]
        or run.get("head_sha") != identity["head_sha"]
    ):
        return False
    comment = unchanged_comment(client, repo, number, int(record["comment_id"]))
    command = parse_command(comment.get("body"))
    return bool(
        command
        and command["scopes"] == record.get("scopes")
        and command["full"] == record.get("full")
        and (
            not command["pinned"]
            or (
                command["pinned"]
                == [
                    record.get("head_sha"),
                    record.get("base_sha"),
                    record.get("source_sha"),
                ]
            )
        )
        and comment.get("user", {}).get("login") == record.get("actor")
        and comment["created_at"] == record.get("comment_created_at")
        and timestamp(comment["created_at"]) >= timestamp(run["created_at"])
        and can_write(client, repo, record["actor"])
    )


def bootstrap_authorizations(
    client, repo: str, number: int, run: dict, identity: dict
) -> list[dict]:
    """Explicit SHA tuples let a new controller qualify its own introduction PR."""
    query = urlencode({"since": run["created_at"], "per_page": PAGE_SIZE})
    comments = client.request(f"repos/{repo}/issues/{number}/comments?{query}")
    approved = []
    for comment in comments:
        command = parse_command(comment.get("body"))
        if (
            not command
            or command["pinned"] != list(identity.values())
            or comment.get("created_at") != comment.get("updated_at")
            or timestamp(comment["created_at"]) < timestamp(run["created_at"])
        ):
            continue
        if can_write(client, repo, comment.get("user", {}).get("login")):
            approved.append(command)
    return approved


def resolve(client, event: dict, *, source: str, base: str) -> dict:
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return dict(NO_AUTHORIZATION)
    repo, number = os.environ["GITHUB_REPOSITORY"], int(event["number"])
    identity = source_identity(client, repo, number)
    head_sha = event.get("pull_request", {}).get("head", {}).get("sha")
    if (
        identity["head_sha"] != head_sha
        or not isinstance(source, str)
        or not SHA.fullmatch(source)
        or not isinstance(base, str)
        or not SHA.fullmatch(base)
    ):
        return dict(NO_AUTHORIZATION)
    if not merge_source_matches(client, repo, source, base, identity):
        return dict(NO_AUTHORIZATION)
    run_id = int(os.environ["GITHUB_RUN_ID"])
    run = client.request(f"repos/{repo}/actions/runs/{run_id}")
    if (
        run.get("event") != "pull_request"
        or run.get("head_sha") != identity["head_sha"]
    ):
        return dict(NO_AUTHORIZATION)
    if trusted_queue_pull(client.request(f"repos/{repo}/pulls/{number}"), repo):
        return {"authorized": True, "scopes": ["affected"], "full": False}
    approved = bootstrap_authorizations(client, repo, number, run, identity)
    name = f"ci-manual-authorization-{run_id}"
    artifacts = client.request(
        f"repos/{repo}/actions/artifacts?name={name}&per_page={PAGE_SIZE}"
    ).get("artifacts", [])
    for artifact in artifacts:
        if artifact.get("name") != name or artifact.get("expired"):
            continue
        try:
            record = read_artifact(
                client, repo, artifact, "authorization.json", AUTHORIZATION_LIMIT
            )
            if (
                matching_authorization(client, repo, number, run, identity, record)
                and record.get("source_sha") == source
                and record.get("base_sha") == base
                and trusted_controller(client, repo, record, artifact)
            ):
                approved.append(record)
        except (ManualCIError, KeyError, TypeError, ValueError):
            continue  # Malformed or obsolete authorization never permits execution.
    return {
        "authorized": bool(approved),
        "scopes": sorted({scope for record in approved for scope in record["scopes"]}),
        "full": any(record["full"] for record in approved),
    }


def rerun(client, record: dict, *, wait_seconds: int = 900) -> None:
    repo, number, run_id = (
        record["repository"],
        record["pr_number"],
        record["pr_run_id"],
    )
    deadline = time.monotonic() + wait_seconds
    while True:
        identity = source_identity(client, repo, number)
        if not main_contains(client, repo, record["base_sha"]):
            raise ManualCIError(
                "The PR revision or authorization changed; post a new /ci run request.",
                reason="expired",
            )
        run = client.request(f"repos/{repo}/actions/runs/{run_id}")
        if not matching_authorization(
            client, repo, number, run, identity, record
        ) or not merge_source_matches(
            client, repo, record.get("source_sha"), record.get("base_sha"), identity
        ):
            raise ManualCIError(
                "The PR revision or authorization changed; post a new /ci run request.",
                reason="expired",
            )
        if run.get("status") == "completed":
            client.request(f"repos/{repo}/actions/runs/{run_id}/rerun", method="POST")
            return
        if time.monotonic() >= deadline:
            raise ManualCIError(
                "Automatic CI is still running; rerun this request after it completes.",
                reason="busy",
            )
        time.sleep(10)


def authorize_with_feedback(client, event: dict) -> dict | None:
    feedback = feedback_for_event(client, event)
    try:
        if command_event(event) and parse_command(event["comment"].get("body")) is None:
            raise ManualCIError("Unsupported /ci run command.", reason="syntax")
        if feedback:
            feedback.received()
        return authorize(client, event)
    except (
        ManualCIError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        if feedback:
            feedback.rejected(getattr(error, "reason", "failed"))
        raise


def rerun_with_feedback(client, record: dict, event: dict) -> None:
    feedback = feedback_for_event(client, event, record=record)
    try:
        rerun(client, record)
    except (
        ManualCIError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        if feedback:
            reason = getattr(error, "reason", "trigger-failed")
            feedback.rejected(
                "trigger-failed" if reason == "failed" else reason,
                run_id=record.get("pr_run_id"),
            )
        raise
    if feedback:
        feedback.started(record["pr_run_id"])


def feedback_event() -> dict:
    """Feedback context is optional; reading it must not block an authorized rerun."""
    try:
        return json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    except (KeyError, ValueError, OSError):
        print("::warning::CI command feedback context is unavailable.")
        return {}


def write_outputs(values: dict) -> None:
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            for key, value in values.items():
                stream.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    authorize_parser = commands.add_parser("authorize")
    authorize_parser.add_argument("--event", type=Path, required=True)
    authorize_parser.add_argument("--output", type=Path, required=True)
    resolver = commands.add_parser("resolve")
    resolver.add_argument("--event", type=Path, required=True)
    resolver.add_argument("--source", required=True)
    resolver.add_argument("--base", required=True)
    resolver.add_argument("--output", type=Path, required=True)
    rerunner = commands.add_parser("rerun")
    rerunner.add_argument("--authorization", type=Path, required=True)
    feedback_parser = commands.add_parser("feedback")
    feedback_parser.add_argument("--event", type=Path, required=True)
    feedback_parser.add_argument("--reason", choices=["controller"], required=True)
    args = parser.parse_args()
    if args.command == "feedback":
        try:
            event = json.loads(args.event.read_text())
            feedback = feedback_for_event(GitHubClient(), event)
            if feedback:
                feedback.rejected(args.reason)
        except (KeyError, TypeError, ValueError, OSError):
            print("::warning::CI command feedback context is unavailable.")
        return 0
    try:
        client = GitHubClient()
        if args.command == "rerun":
            record = json.loads(args.authorization.read_text())
            rerun_with_feedback(client, record, feedback_event())
            print(f"Requested rerun of pull-request CI {record['pr_run_id']}.")
            return 0
        event = json.loads(args.event.read_text())
        result = (
            authorize_with_feedback(client, event)
            if args.command == "authorize"
            else resolve(client, event, source=args.source, base=args.base)
        )
        if args.command == "authorize":
            write_outputs({"authorized": str(result is not None).lower()})
            if result is None:
                return 0
            write_outputs({"pr_run_id": result["pr_run_id"]})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("CI authorization resolved for the current immutable PR revision.")
        return 0
    except (
        ManualCIError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"CI request rejected: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
