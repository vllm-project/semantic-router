#!/usr/bin/env python3
"""Publish PR qualification state from trusted code without holding a runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from manual_ci import (
    MERGE_EVIDENCE_UNAVAILABLE,
    PAGE_SIZE,
    PR_QUALIFICATION_CONTEXT,
    GitHubClient,
    ManualCIError,
    api_failure,
    latest_qualification_status,
    main_contains,
    pending_rerun_status,
    plan_for_run,
    qualification_status_url,
    source_identity,
    trusted_queue_pull,
    write_qualification_status,
)
from pr_ci_events import resolve_event
from pr_ci_pending import fingerprint, provenance, recover_pending, repair
from pr_ci_shared import (
    APP_ID,
    CAPABILITIES,
    CHECK,
    COMMENT_ITEM_LIMIT,
    EXECUTION_CHECK,
    IDENTIFIER,
    MARKER,
    MAX_IDENTIFIERS,
    MAX_PAGES,
    PREFIX,
    SHA,
    artifact,
    current_pull,
    identifiers,
    latest_run,
    owned_checks,
    pages,
    presentation,
    publisher_ready,
    qualification_only_rules,
    validate_plan,
    verdict,
)
from pr_ci_status_suites import read_rollup, read_suites, successful

# Preserve the existing module-level evidence helper API.
__all__ = [
    "APP_ID",
    "CAPABILITIES",
    "CHECK",
    "COMMENT_ITEM_LIMIT",
    "IDENTIFIER",
    "MARKER",
    "MAX_IDENTIFIERS",
    "MAX_PAGES",
    "PREFIX",
    "SHA",
    "artifact",
    "current_pull",
    "identifiers",
    "latest_run",
    "owned_checks",
    "pages",
    "presentation",
    "publisher_ready",
    "validate_plan",
    "verdict",
]


class StatusClient(GitHubClient):
    def write(self, endpoint: str, payload: dict, *, method="POST"):
        result = self.gh(
            ["gh", "api", endpoint, "--method", method, "--input", "-"],
            read=False,
            input=json.dumps(payload),
            text=True,
        )
        if result.returncode:
            raise api_failure(result, "Could not publish CI status")
        return json.loads(result.stdout) if result.stdout.strip() else None


def qualifies_for_merge(repo: str, pull: dict, plan: dict | None) -> bool:
    # A new Ready event must not promote the old draft run's receipt.
    return (
        plan is not None
        and plan.get("draft") is False
        and (pull.get("draft") is False or trusted_queue_pull(pull, repo))
    )


def current_publication_pull(client, repo: str, pull: dict, run: dict) -> dict | None:
    fresh = client.request(f"repos/{repo}/actions/runs/{run['id']}")
    current = client.request(f"repos/{repo}/pulls/{pull['number']}")
    if (
        current.get("state") != "open"
        or current["head"]["sha"] != run["head_sha"]
        or current.get("merge_commit_sha") != pull.get("merge_commit_sha")
        or current.get("base", {}).get("sha") != pull.get("base", {}).get("sha")
        or fresh.get("run_attempt") != run["run_attempt"]
        or fresh.get("status") != run["status"]
        or not latest_run(client, repo, run)
    ):
        return None
    return current


def publish_qualification_status(client, repo, pull, run, plan, result) -> dict:
    payload = result["qualification_status"]
    head = run["head_sha"]
    try:
        previous = latest_qualification_status(client, repo, head)
        fence = pending_rerun_status(repo, run["id"], run["run_attempt"])
        if all(previous.get(key) == value for key, value in fence.items()):
            result["qualification_status_ignored"] = "waiting for rerun"
            return result
        if (
            result.get("state") == "running"
            and previous.get("target_url") == payload["target_url"]
            and previous.get("state") in {"success", "failure", "error"}
        ):
            result["qualification_status_ignored"] = "current attempt already completed"
            return result
        current = current_publication_pull(client, repo, pull, run)
        if current is None or (
            payload["state"] == "success"
            and not qualifies_for_merge(repo, current, plan)
        ):
            result["qualification_status_ignored"] = "superseded"
            return result
        if any(previous.get(key) != value for key, value in payload.items()):
            write_qualification_status(client, repo, head, payload)
    except (
        ManualCIError,
        KeyError,
        TypeError,
        OSError,
        ValueError,
        subprocess.TimeoutExpired,
    ):
        result["qualification_status_error"] = "Could not publish PR Qualification"
    return result


def publish(
    client,
    repo: str,
    pull: dict,
    run: dict,
    state: str,
    plan: dict | None,
    checks: list[dict],
    *,
    dry_run=False,
    reason: str | None = None,
) -> dict:
    if state == "success" and not qualifies_for_merge(repo, pull, plan):
        return {"ignored": "draft CI does not qualify this revision for merge"}
    title, summary = presentation(state, repo, pull, run, plan, reason)
    external = f"{PREFIX}{pull['number']}:{run['id']}:{run['run_attempt']}"
    payload = {
        "name": CHECK,
        "head_sha": run["head_sha"],
        "external_id": external,
        "details_url": f"https://github.com/{repo}/actions/runs/{run['id']}",
        "status": "completed" if state in {"success", "failure"} else "in_progress",
        "output": {"title": title, "summary": summary},
    }
    if payload["status"] == "completed":
        payload["conclusion"] = state
    result = {
        "pr": pull["number"],
        "state": state,
    }
    if state != "legacy":
        result["qualification_status"] = {
            "context": PR_QUALIFICATION_CONTEXT,
            "state": state if state in {"success", "failure"} else "pending",
            "target_url": qualification_status_url(repo, run["id"], run["run_attempt"]),
            "description": title[:140],
        }
        if not dry_run:
            # Publish from the verified attempt before touching compatibility output.
            publish_qualification_status(client, repo, pull, run, plan, result)
            if result.get("qualification_status_ignored") == "superseded":
                return {"ignored": "superseded while preparing status"}
    legacy_gate = state != "legacy" and not qualification_only_rules(
        client, repo, pull["base"]["ref"]
    )
    result["check"] = payload if legacy_gate else None
    comments = pages(client, f"repos/{repo}/issues/{pull['number']}/comments")
    owned = [
        item
        for item in comments
        if item.get("user", {}).get("login") == "github-actions[bot]"
        and item.get("user", {}).get("type") == "Bot"
        and item.get("body", "").startswith(MARKER)
    ]
    comment_body = f"{MARKER}\n{summary}\n"
    result["comment"] = (
        comment_body
        if owned
        or state in {"awaiting-approval", "legacy", "stale"}
        or (plan and plan.get("deferred_verifications"))
        else None
    )
    if reason == MERGE_EVIDENCE_UNAVAILABLE:
        result["reason_codes"] = [reason]
    if dry_run:
        return result
    visible_checks = (
        owned_checks(client, repo, pull, latest=True) if legacy_gate else []
    )
    # Recheck after reading artifacts/comments; an old completion must never
    # overwrite the status of a newer run, attempt, or contributor commit.
    current = current_publication_pull(client, repo, pull, run)
    if current is None:
        return {"ignored": "superseded while preparing status"}
    if state == "success" and not qualifies_for_merge(repo, current, plan):
        return {"ignored": "PR became draft while preparing its Gate"}
    if legacy_gate:
        fresh_checks = owned_checks(client, repo, pull)
        current_checks = [c for c in fresh_checks if c.get("external_id") == external]
        if payload["status"] == "in_progress" and any(
            c.get("external_id") == external and c.get("status") == "completed"
            for c in [*checks, *fresh_checks, *visible_checks]
        ):
            return {"ignored": "current completed verdict cannot return to pending"}
        visible_current = [
            c for c in visible_checks if c.get("external_id") == external
        ]
        if (
            state == "awaiting-approval"
            and current_checks
            and (not visible_current or any(provenance(c) for c in current_checks))
        ):
            try:
                recovered = repair(
                    client, repo, run["id"], (run["id"], run["run_attempt"])
                )
            except (
                ManualCIError,
                KeyError,
                ValueError,
                TypeError,
                OSError,
                subprocess.TimeoutExpired,
            ):
                return {
                    "ignored": "pending publication evidence unavailable or changed"
                }
            return {**result, **recovered}
        # PATCH cannot move a check into a rerun's attempt. Only reuse a
        # visible check for this exact attempt, including when repairing an
        # external_id that the old publisher rewrote on a superseded check.
        existing = [
            item for item in visible_checks if item.get("external_id") == external
        ]
        current_id = max((item["id"] for item in existing), default=None)
        if existing:
            if payload["status"] == "in_progress":
                target = max(existing, key=lambda item: item["id"])
                observed = client.request(f"repos/{repo}/check-runs/{current_id}")
                if observed.get("status") == "completed" or fingerprint(
                    observed
                ) != fingerprint(target):
                    return {"ignored": "pending publication target changed"}
            update = dict(payload)
            update.pop("head_sha")
            client.write(
                f"repos/{repo}/check-runs/{current_id}",
                update,
                method="PATCH",
            )
        else:
            client.write(f"repos/{repo}/check-runs", payload)
        # Publish the authoritative check first, so retiring old pending
        # checks cannot leave a window without the current qualification.
        run_prefix = f"{PREFIX}{pull['number']}:{run['id']}:"
        for item in checks:
            if (
                item["id"] != current_id
                and item.get("status") == "in_progress"
                and str(item.get("external_id", "")).startswith(run_prefix)
            ):
                client.write(
                    f"repos/{repo}/check-runs/{item['id']}",
                    {
                        "status": "completed",
                        "conclusion": "neutral",
                        "output": {
                            "title": "Superseded qualification status",
                            "summary": (
                                "The PR Gate for the current "
                                f"[CI attempt {run['run_attempt']}]({payload['details_url']}) "
                                "replaces this pending check."
                            ),
                        },
                    },
                    method="PATCH",
                )
    if result["comment"]:
        if owned:
            comment = max(owned, key=lambda item: item["id"])
            if comment.get("body") != comment_body:
                client.write(
                    f"repos/{repo}/issues/comments/{comment['id']}",
                    {"body": comment_body},
                    method="PATCH",
                )
        else:
            client.write(
                f"repos/{repo}/issues/{pull['number']}/comments", {"body": comment_body}
            )
    return result


def reconcile(client, repo: str, run_id: int, *, dry_run=False) -> dict:
    run = client.request(f"repos/{repo}/actions/runs/{run_id}")
    workflow = client.request(f"repos/{repo}/actions/workflows/pr.yml")
    if (
        run.get("event") != "pull_request"
        or run.get("path") != ".github/workflows/pr.yml"
        or run.get("workflow_id") != workflow.get("id")
        or not latest_run(client, repo, run)
    ):
        return {"ignored": "not the current pull-request workflow"}
    pull = current_pull(client, repo, run)
    if not pull:
        return {"ignored": "no matching current open pull request"}
    checks = owned_checks(client, repo, pull)
    if run.get("status") != "completed":
        # No earlier verdict needs fencing on a fresh head. Once legacy
        # publishing is off, the commit status identifies earlier attempts.
        if not checks and not latest_qualification_status(
            client, repo, run["head_sha"]
        ):
            return {"ignored": "first attempt is still preparing its plan"}
        return publish(
            client, repo, pull, run, "running", None, checks, dry_run=dry_run
        )
    jobs = list(
        pages(
            client,
            f"repos/{repo}/actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs",
            "jobs",
        )
    )
    plan = None
    reason = None
    try:
        plan = artifact(client, repo, run, "ci-plan", "plan.json", 16 * 1024 * 1024)
        validate_plan(client, repo, run, plan)
        if not publisher_ready(client, repo, plan["base_sha"]):
            # Old PR sources keep the original required Actions job. Never add a
            # competing check with the same required name during migration.
            failures = [
                job
                for job in jobs
                if job.get("conclusion") not in {"success", "skipped"}
            ]
            if plan["deferred_verifications"] and all(
                job.get("name") in {"CI / Gate", "PR Gate"} for job in failures
            ):
                return publish(
                    client, repo, pull, run, "legacy", plan, checks, dry_run=dry_run
                )
            return {"ignored": "legacy workflow owns PR Gate"}
        publisher_ids = {check["id"] for check in checks}
        jobs = [job for job in jobs if job.get("id") not in publisher_ids]
        report = artifact(client, repo, run, "ci-gate", "gate.json", 256 * 1024)
        state = verdict(run, plan, report, jobs)
        if state != "failure" and not main_contains(client, repo, plan["base_sha"]):
            state = "stale"
    except (ManualCIError, KeyError, ValueError, TypeError) as error:
        if not checks and not any(job.get("name") == EXECUTION_CHECK for job in jobs):
            raise
        state = "failure"
        if (
            isinstance(error, ManualCIError)
            and error.reason == MERGE_EVIDENCE_UNAVAILABLE
        ):
            reason = MERGE_EVIDENCE_UNAVAILABLE
        plan = None  # Never interpolate malformed artifact content into a comment.
    return publish(
        client, repo, pull, run, state, plan, checks, dry_run=dry_run, reason=reason
    )


def required_checks(client, repo: str, head: str, number: int):
    """Resolve requiredness by check node identity, never by display name."""
    checks = list(
        pages(
            client,
            f"repos/{repo}/commits/{head}/check-runs?check_name=PR%20Gate&filter=all",
            "check_runs",
        )
    )
    if len(checks) > MAX_IDENTIFIERS or any(not item.get("node_id") for item in checks):
        return {}, False
    query = "query($ids:[ID!]!,$number:Int!){nodes(ids:$ids){... on CheckRun{databaseId isRequired(pullRequestNumber:$number)}}}"
    required = {}
    for start in range(0, len(checks), PAGE_SIZE):
        batch = checks[start : start + PAGE_SIZE]
        response = client.gh(
            [
                "gh",
                "api",
                "graphql",
                "-f",
                f"query={query}",
                "-F",
                f"number={number}",
                *[
                    argument
                    for item in batch
                    for argument in ("-f", f"ids[]={item['node_id']}")
                ],
            ],
            read=True,
            text=True,
        )
        if response.returncode:
            return {}, False
        data = json.loads(response.stdout)
        if data.get("errors"):
            return {}, False
        nodes = data["data"]["nodes"]
        if len(nodes) != len(batch):
            return {}, False
        for check, node in zip(batch, nodes, strict=True):
            if (
                not node
                or node.get("databaseId") != check["id"]
                or type(node.get("isRequired")) is not bool
            ):
                return {}, False
            required[check["id"]] = node["isRequired"]
    return required, True


def diagnostic_run_matches(run: dict, selected: dict, workflow: dict, repo: str):
    return (
        run.get("event") == "pull_request"
        and run.get("path") == ".github/workflows/pr.yml"
        and run.get("workflow_id") == workflow.get("id")
        and run.get("head_sha") == selected.get("head_sha")
        and run.get("head_branch") == selected.get("head_branch")
        and (run.get("head_repository") or {}).get("full_name")
        == (selected.get("head_repository") or {}).get("full_name")
        and (run.get("repository") or {}).get("full_name") == repo
    )


def diagnostic_native(client, repo, check, run, workflow, snapshots):
    """Bind a cancelled native job; its failure cause remains unproven."""
    external = check.get("external_id")
    if (
        check.get("app", {}).get("id") != APP_ID
        or not isinstance(external, str)
        or not re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", external
        )
        or check.get("head_sha") != run["head_sha"]
    ):
        return False
    suite = check.get("check_suite", {}).get("id")
    if type(suite) is not int or suite <= 0:
        return False
    candidates = list(
        pages(
            client, f"repos/{repo}/actions/runs?check_suite_id={suite}", "workflow_runs"
        )
    )
    if len(candidates) > MAX_IDENTIFIERS:
        raise ManualCIError("Diagnostic run inventory exceeds the limit.")
    matches = [
        candidate
        for candidate in candidates
        if candidate.get("check_suite_id") == suite
        and diagnostic_run_matches(candidate, run, workflow, repo)
    ]
    if len(matches) != 1:
        return False
    candidate = client.request(f"repos/{repo}/actions/runs/{int(matches[0]['id'])}")
    if (
        not diagnostic_run_matches(candidate, run, workflow, repo)
        or candidate.get("check_suite_id") != suite
    ):
        return False
    snapshots[candidate["id"]] = candidate
    if (
        candidate.get("status") != "completed"
        or candidate.get("conclusion") != "cancelled"
    ):
        return False
    attempt = int(candidate["run_attempt"])
    jobs = list(
        pages(
            client,
            f"repos/{repo}/actions/runs/{candidate['id']}/attempts/{attempt}/jobs",
            "jobs",
        )
    )
    gates = [
        job
        for job in jobs
        if job.get("id") == check["id"]
        and job.get("name") == CHECK
        and job.get("conclusion") == "failure"
    ]
    if len(gates) != 1:
        return False
    return {
        "run_id": candidate["id"],
        "attempt": attempt,
        "other_failed_jobs": any(
            job.get("conclusion") in {"failure", "timed_out", "startup_failure"}
            and job.get("id") != check["id"]
            and job.get("name") != "CI / Gate"
            for job in jobs
        ),
    }


def diagnostic_check(check, role, required):
    """Only typed IDs and allowlisted labels cross the stdout boundary."""
    external = check.get("external_id")
    valid_external = isinstance(external, str) and (
        re.fullmatch(
            r"semantic-router-ci:[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*", external
        )
        or re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", external
        )
    )

    def identity(value):
        return value if type(value) is int and value > 0 else None

    return {
        "id": identity(check.get("id")),
        "suite_id": identity(check.get("check_suite", {}).get("id")),
        "app_id": identity(check.get("app", {}).get("id")),
        "external_id": external if valid_external else None,
        "status": (
            check.get("status")
            if check.get("status") in {"queued", "in_progress", "completed"}
            else None
        ),
        "conclusion": (
            check.get("conclusion")
            if check.get("conclusion")
            in {
                "success",
                "failure",
                "cancelled",
                "skipped",
                "neutral",
                "timed_out",
                "action_required",
                "stale",
                "startup_failure",
            }
            else None
        ),
        "role": role,
        "required": required if type(required) is bool else None,
    }


def resolve_requiredness(client, repo: str, head: str, number: int, checks: list[dict]):
    """Resolve requiredness by node identity; unknown is never inferred."""
    try:
        lookup = getattr(client, "required_checks", None)
        required, complete = (
            lookup(repo, head, number)
            if lookup
            else required_checks(client, repo, head, number)
        )
        complete = bool(complete) and all(
            type(required.get(check["id"])) is bool for check in checks
        )
    except (
        ManualCIError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ):
        return {}, False
    return (required if complete else {}), complete


def diagnose(
    client, repo: str, number: int, head: str, run_id: int, attempt: int
) -> dict:
    """Read-only evidence, not authorization to retire checks or recover a PR."""
    result = {
        "pr": number,
        "head": head,
        "run_id": run_id,
        "attempt": attempt,
        "state": "evidence_unavailable",
        "reason_codes": [],
        "actionable": False,
        "guidance": "Read-only evidence; do not change or retire checks based on this report.",
        "evidence": {
            "snapshot_current": False,
            "competition": False,
            "native_failures_verified": False,
            "other_failed_jobs": False,
            "cancellation_only": None,
            "full_qualification": False,
            "current_source": False,
            "requiredness_complete": False,
        },
        "checks": {"all": [], "latest": []},
        "suites": [],
        "run_suite_id": None,
        "rollup_state": None,
    }

    def finish(state, reason):
        result.update(state=state)
        result["reason_codes"].append(reason)
        return result

    if (
        not SHA.fullmatch(head)
        or min(number, run_id, attempt) <= 0
        or not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repo)
    ):
        return finish("ignored", "invalid_identity")
    snapshots = {}
    suite_snapshot = None
    rollup_snapshot = None
    try:
        run = client.request(f"repos/{repo}/actions/runs/{run_id}")
        workflow = client.request(f"repos/{repo}/actions/workflows/pr.yml")
        pull = client.request(f"repos/{repo}/pulls/{number}")
        main_ref = client.request(f"repos/{repo}/git/ref/heads/main")
        if pull.get("number") != number:
            return finish("ignored", "pull_mismatch")
        if pull.get("state") != "open":
            return finish("ignored", "pull_not_open")
        if (
            run.get("id") != run_id
            or run.get("head_sha") != head
            or run.get("run_attempt") != attempt
            or not diagnostic_run_matches(run, run, workflow, repo)
            or pull.get("head", {}).get("sha") != head
            or not latest_run(client, repo, run)
        ):
            return finish("ignored", "identity_mismatch_or_superseded")
        bound = current_pull(client, repo, run)
        if not bound or bound.get("number") != number:
            return finish("ignored", "pull_mismatch")
        snapshots[run_id] = run
        suite_id = run.get("check_suite_id")
        if type(suite_id) is int and suite_id > 0:
            result["run_suite_id"] = suite_id
        source_snapshot = None
        raw = {}
        for check_filter in ("all", "latest"):
            raw[check_filter] = list(
                pages(
                    client,
                    f"repos/{repo}/commits/{head}/check-runs?check_name=PR%20Gate&filter={check_filter}",
                    "check_runs",
                )
            )
            if len(raw[check_filter]) > MAX_IDENTIFIERS:
                raise ManualCIError("Diagnostic check inventory exceeds the limit.")
            if any(
                check.get("name") != CHECK or check.get("head_sha") != head
                for check in raw[check_filter]
            ):
                raise ManualCIError(
                    "Diagnostic checks do not match the requested identity."
                )
        required, complete = resolve_requiredness(
            client, repo, head, number, raw["all"] + raw["latest"]
        )
        result["evidence"]["requiredness_complete"] = bool(complete)
        if not complete:
            required = {}
            result["reason_codes"].append("requiredness_unknown")
        external = f"{PREFIX}{number}:{run_id}:{attempt}"
        owned = owned_checks(client, repo, pull)
        owned_ids = {check["id"] for check in owned}
        replacements = [
            check
            for check in raw["all"]
            if check.get("id") in owned_ids
            and check.get("external_id") == external
            and check.get("head_sha") == head
        ]
        failures = [
            check
            for check in raw["all"]
            if check.get("status") == "completed"
            and check.get("conclusion") == "failure"
            and check.get("id") not in owned_ids
        ]
        native = {}
        for check in failures:
            native[check["id"]] = diagnostic_native(
                client, repo, check, run, workflow, snapshots
            )
        for check_filter, checks in raw.items():
            result["checks"][check_filter] = [
                diagnostic_check(
                    check,
                    (
                        "publisher"
                        if check.get("id") in owned_ids
                        else "native" if native.get(check.get("id")) else "unknown"
                    ),
                    required.get(check.get("id")),
                )
                for check in checks
            ]
        green = bool(replacements) and all(
            check.get("status") == "completed" and check.get("conclusion") == "success"
            for check in replacements
        )
        evidence = result["evidence"]
        evidence["competition"] = green and bool(failures)
        evidence["native_failures_verified"] = bool(failures) and all(native.values())
        evidence["other_failed_jobs"] = any(
            item and item["other_failed_jobs"] for item in native.values()
        )
        if any(native.values()):
            result["reason_codes"].append("cancellation_cause_unproven")
        state, reason = "evidence_unavailable", "qualification_evidence_unavailable"
        if run.get("status") != "completed":
            state, reason = "running", "qualification_incomplete"
        elif run.get("conclusion") != "success" or (
            replacements
            and any(check.get("conclusion") == "failure" for check in replacements)
        ):
            state, reason = "failed", "current_qualification_failed"
        try:
            jobs = list(
                pages(
                    client,
                    f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs",
                    "jobs",
                )
            )
            plan = artifact(client, repo, run, "ci-plan", "plan.json", 16 * 1024 * 1024)
            validate_plan(client, repo, run, plan)
            report = artifact(client, repo, run, "ci-gate", "gate.json", 256 * 1024)
            qualification = verdict(
                run,
                plan,
                report,
                [job for job in jobs if job.get("id") not in owned_ids],
            )
            evidence["full_qualification"] = (
                run.get("status") == "completed" and qualification == "success"
            )
            if run.get("status") == "completed" and qualification == "failure":
                state, reason = "failed", "current_qualification_failed"
            elif qualification == "awaiting-approval":
                state, reason = "deferred", "qualification_deferred"
            elif run.get("status") == "completed" and replacements and not green:
                state, reason = "failed", "current_qualification_failed"
            elif evidence["full_qualification"]:
                try:
                    suite_ids = [
                        run.get("check_suite_id"),
                        *(
                            check.get("check_suite", {}).get("id")
                            for check in replacements
                        ),
                    ]
                    suite_snapshot = read_suites(client, repo, head, suite_ids, APP_ID)
                    rollup_snapshot = read_rollup(client, repo, head)
                    result["suites"] = list(suite_snapshot.values())
                    result["rollup_state"] = rollup_snapshot
                except (
                    ManualCIError,
                    KeyError,
                    ValueError,
                    TypeError,
                    OSError,
                    subprocess.TimeoutExpired,
                ):
                    # Missing health evidence must not skip source revalidation.
                    suite_snapshot = None
                    result["reason_codes"].append("check_state_evidence_unavailable")
                try:
                    identity = source_identity(client, repo, number)
                    source_snapshot = plan_for_run(client, repo, run, identity)
                    evidence["current_source"] = source_snapshot is not None
                except ManualCIError as error:
                    if error.reason == MERGE_EVIDENCE_UNAVAILABLE:
                        raise
                    source_snapshot = False
                    evidence["current_source"] = False
                if not evidence["current_source"]:
                    state, reason = (
                        "stale_blockers" if evidence["competition"] else "source_stale"
                    ), "source_stale"
                elif failures and (
                    not evidence["native_failures_verified"]
                    or evidence["other_failed_jobs"]
                ):
                    state, reason = "uncertain", "native_failure_evidence_incomplete"
                elif evidence["competition"]:
                    confirmed = (
                        complete
                        and any(required.get(c["id"]) for c in replacements)
                        and any(required.get(c["id"]) for c in failures)
                    )
                    state, reason = (
                        "competing_required" if confirmed else "suspected_competition"
                    ), "competing_same_head_checks"
                else:
                    state, reason = (
                        "healthy" if green else "uncertain"
                    ), "no_competing_native_failure"
                    if green:
                        visible = {check["id"] for check in raw["latest"]}
                        required_green = [
                            check
                            for check in replacements
                            if required.get(check["id"]) is True
                            and check["id"] in visible
                        ]
                        if not complete or suite_snapshot is None:
                            state, reason = (
                                "evidence_unavailable",
                                "health_evidence_incomplete",
                            )
                        elif not required_green:
                            state, reason = (
                                "uncertain",
                                "current_required_gate_not_visible",
                            )
                        elif not successful(suite_snapshot[run["check_suite_id"]]):
                            state, reason = (
                                "uncertain",
                                "qualification_suite_not_successful",
                            )
                        elif any(
                            not successful(suite_snapshot[check["check_suite"]["id"]])
                            for check in required_green
                        ):
                            state, reason = (
                                "suite_mismatch",
                                "required_gate_suite_not_successful",
                            )
                        elif rollup_snapshot != "SUCCESS":
                            state, reason = (
                                "rollup_blocked",
                                "commit_rollup_not_successful",
                            )
        except (ManualCIError, KeyError, ValueError, TypeError) as error:
            if (
                isinstance(error, ManualCIError)
                and error.reason == MERGE_EVIDENCE_UNAVAILABLE
            ):
                result["reason_codes"].append(MERGE_EVIDENCE_UNAVAILABLE)
        # The last GETs must still bind the PR and every examined run attempt.
        for identity, snapshot in snapshots.items():
            fresh = client.request(f"repos/{repo}/actions/runs/{identity}")
            if any(
                fresh.get(key) != snapshot.get(key)
                for key in (
                    "id",
                    "run_attempt",
                    "status",
                    "conclusion",
                    "head_sha",
                    "head_branch",
                    "head_repository",
                    "repository",
                    "path",
                    "workflow_id",
                    "check_suite_id",
                )
            ):
                return finish("stale", "snapshot_changed")
        if client.request(f"repos/{repo}/git/ref/heads/main") != main_ref:
            return finish("stale", "snapshot_changed")
        for check_filter, checks in raw.items():
            fresh_checks = list(
                pages(
                    client,
                    f"repos/{repo}/commits/{head}/check-runs?check_name=PR%20Gate&filter={check_filter}",
                    "check_runs",
                )
            )

            if len(fresh_checks) > MAX_IDENTIFIERS:
                return finish("evidence_unavailable", "check_inventory_incomplete")

            def snapshot(items):
                if any(
                    item.get("name") != CHECK or item.get("head_sha") != head
                    for item in items
                ):
                    return None
                return {
                    item["id"]: diagnostic_check(item, "unknown", None)
                    for item in items
                }

            if snapshot(fresh_checks) != snapshot(checks):
                return finish("stale", "snapshot_changed")
        fresh_required, fresh_complete = resolve_requiredness(
            client, repo, head, number, raw["all"] + raw["latest"]
        )
        if (fresh_required, fresh_complete) != (required, complete):
            return finish("stale", "requiredness_changed")
        fresh_pull = client.request(f"repos/{repo}/pulls/{number}")
        if any(
            fresh_pull.get(key) != pull.get(key)
            for key in ("number", "state", "head", "base", "merge_commit_sha")
        ) or not latest_run(client, repo, run):
            return finish("stale", "snapshot_changed")
        if suite_snapshot is not None:
            if (
                read_suites(client, repo, head, suite_snapshot, APP_ID)
                != suite_snapshot
            ):
                return finish("stale", "check_suite_changed")
            if read_rollup(client, repo, head) != rollup_snapshot:
                return finish("stale", "commit_rollup_changed")
        if source_snapshot is not None:
            try:
                fresh_identity = source_identity(client, repo, number)
                fresh_source = plan_for_run(client, repo, run, fresh_identity)
            except ManualCIError as error:
                if error.reason == MERGE_EVIDENCE_UNAVAILABLE:
                    evidence["current_source"] = False
                    return finish("evidence_unavailable", MERGE_EVIDENCE_UNAVAILABLE)
                fresh_source = False
            if bool(fresh_source) != bool(source_snapshot):
                return finish("stale", "source_changed")
        evidence["snapshot_current"] = True
        return finish(state, reason)
    except (
        ManualCIError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        subprocess.TimeoutExpired,
    ):
        return finish("evidence_unavailable", "read_evidence_unavailable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--head")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--resolve-event", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("publish", "pending"))
    args = parser.parse_args()
    if args.resolve_event:
        if not args.event or args.diagnose or args.run_id or args.mode or args.dry_run:
            parser.error("--resolve-event requires only --event and optional --output")
        try:
            event = json.loads(args.event.read_text(encoding="utf-8"))
            result = resolve_event(
                StatusClient(), os.environ["GITHUB_REPOSITORY"], event
            )
        except (
            ManualCIError,
            KeyError,
            ValueError,
            TypeError,
            OSError,
            subprocess.TimeoutExpired,
        ):
            result = {"ignored": "no verified qualification identity for this event"}
        if args.output:
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"run_id={result.get('run_id', '')}\nmode={result.get('mode', '')}\n"
                )
        print(json.dumps(result, indent=2))
        return 0
    if args.output:
        parser.error("--output requires --resolve-event")
    if args.diagnose:
        if (
            args.event
            or args.mode
            or any(
                value is None
                for value in (args.pr, args.head, args.run_id, args.attempt)
            )
        ):
            parser.error(
                "--diagnose requires --pr, --head, --run-id and --attempt; not --event"
            )
        result = diagnose(
            StatusClient(),
            os.environ.get("GITHUB_REPOSITORY", ""),
            args.pr,
            args.head,
            args.run_id,
            args.attempt,
        )
        print(json.dumps(result, indent=2))
        return 0
    if any(value is not None for value in (args.pr, args.head, args.attempt)):
        parser.error("--pr, --head and --attempt require --diagnose")
    try:
        event = json.loads(args.event.read_text(encoding="utf-8")) if args.event else {}
        community = (event.get("workflow_run") or {}).get("path") == (
            ".github/workflows/community.yml"
        )
        if args.mode == "pending" or (community and args.mode is None):
            result = recover_pending(
                StatusClient(),
                os.environ["GITHUB_REPOSITORY"],
                event,
                args.run_id,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, indent=2))
            return 0
        if community:
            raise ManualCIError("Community events require pending-only publication.")
        run_id = args.run_id or int(
            event.get("inputs", {}).get("run_id") or event["workflow_run"]["id"]
        )
        result = reconcile(
            StatusClient(),
            os.environ["GITHUB_REPOSITORY"],
            run_id,
            dry_run=args.dry_run,
        )
        if "qualification_status_error" in result:
            print("::warning::Could not publish PR Qualification")
        print(json.dumps(result, indent=2))
        return 0
    except (
        ManualCIError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"CI status could not be reconciled: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
