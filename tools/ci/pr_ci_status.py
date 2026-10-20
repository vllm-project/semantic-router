#!/usr/bin/env python3
"""Publish PR qualification state from trusted code without holding a runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlencode

from manual_ci import PAGE_SIZE, GitHubClient, ManualCIError, read_artifact

CHECK = "PR Gate"
MARKER = "<!-- semantic-router-ci-status:v1 -->"
PREFIX = "semantic-router-ci:"
APP_ID = 15368  # GitHub Actions; also the branch-protection check owner.
SHA = re.compile(r"[0-9a-f]{40}")
IDENTIFIER = re.compile(r"[a-zA-Z0-9_.-]{1,160}")
CAPABILITIES = (".github/workflows/ci-status.yml", "tools/ci/pr_ci_status.py")
MAX_PAGES = 100
MAX_IDENTIFIERS = 1000
COMMENT_ITEM_LIMIT = 30


class StatusClient(GitHubClient):
    def write(self, endpoint: str, payload: dict, *, method="POST"):
        result = subprocess.run(
            ["gh", "api", endpoint, "--method", method, "--input", "-"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise ManualCIError(
                "Could not publish CI status; check workflow permissions."
            )
        return json.loads(result.stdout) if result.stdout.strip() else None


def pages(client, endpoint: str, key: str | None = None):
    separator = "&" if "?" in endpoint else "?"
    for page in range(1, MAX_PAGES + 1):
        response = client.request(
            f"{endpoint}{separator}per_page={PAGE_SIZE}&page={page}"
        )
        items = response[key] if key else response
        if not isinstance(items, list):
            raise ManualCIError("Invalid GitHub collection response.")
        yield from items
        if len(items) < PAGE_SIZE:
            return
    raise ManualCIError("GitHub collection exceeds the reconciliation limit.")


def current_pull(client, repo: str, run: dict) -> dict | None:
    # Both workflow_run.pull_requests and commits/{sha}/pulls can be empty for
    # fork commits. Resolve by source owner/branch, then bind the exact SHA.
    source_repo = (run.get("head_repository") or {}).get("full_name", "")
    if not isinstance(source_repo, str) or source_repo.count("/") != 1:
        raise ManualCIError("Invalid pull-request source repository.")
    query = urlencode(
        {
            "state": "open",
            "base": "main",
            "head": f"{source_repo.split('/')[0]}:{run['head_branch']}",
        }
    )
    candidates = pages(client, f"repos/{repo}/pulls?{query}")
    matches = []
    for candidate in candidates:
        if candidate.get("state") != "open":
            continue
        pull = client.request(f"repos/{repo}/pulls/{int(candidate['number'])}")
        head, base = pull.get("head", {}), pull.get("base", {})
        if (
            pull.get("state") == "open"
            and base.get("ref") == "main"
            and (base.get("repo") or {}).get("full_name") == repo
            and head.get("sha") == run["head_sha"]
            and head.get("ref") == run.get("head_branch")
            and (head.get("repo") or {}).get("full_name")
            == (run.get("head_repository") or {}).get("full_name")
        ):
            matches.append(pull)
    if len(matches) > 1:
        raise ManualCIError(
            "The CI run is associated with multiple open pull requests."
        )
    return matches[0] if matches else None


def latest_run(client, repo: str, run: dict) -> bool:
    query = urlencode({"event": "pull_request", "head_sha": run["head_sha"]})
    candidates = [
        item
        for item in pages(
            client,
            f"repos/{repo}/actions/workflows/pr.yml/runs?{query}",
            "workflow_runs",
        )
        if item.get("head_branch") == run.get("head_branch")
        and (item.get("head_repository") or {}).get("full_name")
        == (run.get("head_repository") or {}).get("full_name")
    ]
    return (
        bool(candidates)
        and max(candidates, key=lambda item: (item["created_at"], item["id"]))["id"]
        == run["id"]
    )


def artifact(client, repo: str, run: dict, name: str, filename: str, limit: int):
    items = list(
        pages(
            client,
            f"repos/{repo}/actions/runs/{run['id']}/artifacts?name={name}",
            "artifacts",
        )
    )
    matches = [
        item for item in items if item.get("name") == name and not item.get("expired")
    ]
    if len(matches) != 1:
        raise ManualCIError(f"Missing or ambiguous {name} evidence.")
    return read_artifact(client, repo, matches[0], filename, limit)


def identifiers(value) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_IDENTIFIERS
        or any(
            not isinstance(item, str) or not IDENTIFIER.fullmatch(item)
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise ManualCIError("Invalid qualification inventory.")
    return value


def validate_plan(client, repo: str, run: dict, plan: dict) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if plan.get("schema_version") != 1 or plan.get("plan_sha256") != digest:
        raise ManualCIError("Invalid qualification plan digest.")
    if any(
        not isinstance(plan.get(key), str) or not SHA.fullmatch(plan[key])
        for key in ("source_sha", "base_sha")
    ):
        raise ManualCIError("Invalid qualification source revision.")
    commit = client.request(f"repos/{repo}/git/commits/{plan['source_sha']}")
    if [item.get("sha") for item in commit.get("parents", [])] != [
        plan["base_sha"],
        run["head_sha"],
    ]:
        raise ManualCIError(
            "Qualification plan does not belong to this PR merge revision."
        )
    selected = identifiers(plan.get("expected_verification_ids"))
    deferred = identifiers(plan.get("deferred_verifications"))
    required = identifiers(plan.get("required_verification_ids"))
    identifiers(plan.get("deferred_images"))
    if set(selected) & set(deferred) or set(required) != set(selected) | set(deferred):
        raise ManualCIError("Qualification inventories do not partition the plan.")


def publisher_ready(client, repo: str, base: str) -> bool:
    tree = client.request(f"repos/{repo}/git/trees/{base}?recursive=1")
    if tree.get("truncated"):
        raise ManualCIError("Cannot establish the trusted base's status publisher.")
    paths = {
        item.get("path") for item in tree.get("tree", []) if item.get("type") == "blob"
    }
    return set(CAPABILITIES) <= paths


def verdict(run: dict, plan: dict, report: dict, jobs: list[dict]) -> str:
    """A successful automatic workflow is not necessarily full qualification."""
    if run.get("conclusion") != "success" or any(
        job.get("conclusion") not in {"success", "skipped"} for job in jobs
    ):
        return "failure"
    by_name = {job.get("name"): job for job in jobs}
    if any(
        by_name.get(name, {}).get("conclusion") != "success"
        for name in ("CI / Gate", "CI Execution")
    ):
        raise ManualCIError("Required CI reconciliation jobs did not succeed.")
    expected = {
        "schema_version": 1,
        "source_sha": plan["source_sha"],
        "base_sha": plan["base_sha"],
        "plan_sha256": plan["plan_sha256"],
        "run_id": int(run["id"]),
        "run_attempt": int(run["run_attempt"]),
        "required_verification_ids": plan["required_verification_ids"],
        "deferred_verifications": plan["deferred_verifications"],
        "deferred_images": plan["deferred_images"],
    }
    if (
        any(report.get(key) != value for key, value in expected.items())
        or report.get("errors") != []
    ):
        raise ManualCIError(
            "CI reconciliation receipt is missing, stale or inconsistent."
        )
    waiting = bool(plan["deferred_verifications"] or plan["deferred_images"])
    expected_status = "awaiting-approval" if waiting else "success"
    if report.get("status") != expected_status:
        raise ManualCIError("CI receipt does not match the remaining qualification.")
    return expected_status


def presentation(
    state: str, repo: str, pull: dict, run: dict, plan: dict | None
) -> tuple[str, str]:
    url = f"https://github.com/{repo}/actions/runs/{run['id']}"
    titles = {
        "running": "CI is running",
        "awaiting-approval": "Waiting for committer approval — /ci run",
        "stale": "Update the branch before requesting CI",
        "failure": "CI failed — inspect the failing checks",
        "success": "All required CI qualification passed",
        "legacy": "Waiting for committer approval — /ci run",
    }
    lines = [f"**{titles[state]}**", ""]
    if state in {"awaiting-approval", "legacy"}:
        lines += [
            (
                "Automatic checks passed. Required expensive checks have not run yet; this is not a test failure."
                if state == "awaiting-approval"
                else "This older run has required checks awaiting approval. Its existing Gate remains authoritative; inspect that run for any additional evidence failures."
            ),
            "",
            "A committer with repository **write, maintain or admin** permission can start the required checks by commenting:",
            "",
            "```text",
            "/ci run",
            "```",
            "",
            "Contributors without that permission do not need to request authorization themselves.",
        ]
        if state == "legacy":
            lines += [
                "",
                "This run uses the older gate display. Update the branch from main to use the waiting status, then request CI after its Plan job finishes.",
            ]
    elif state == "stale":
        lines += [
            "The tested merge revision is no longer current. Update the branch from main, wait for the new Plan job, then ask a committer to comment `/ci run`."
        ]
    elif state == "failure":
        lines += [
            "A check failed or its required evidence could not be verified. Open the run below for the failure; approval alone does not turn a failed test green."
        ]
        if plan and plan.get("deferred_verifications"):
            lines += [
                "After fixing the failure, a committer must also comment `/ci run` to authorize the remaining qualification."
            ]
    elif state == "running":
        lines += [
            "The current CI attempt is running. This check will update when its results are available."
        ]
    else:
        lines += [
            "All selected and required checks have completed successfully for this revision."
        ]
    if (
        plan
        and plan.get("deferred_verifications")
        and state not in {"running", "success"}
    ):
        lines += [
            "",
            "Pending qualification: "
            + ", ".join(
                f"`{item}`"
                for item in plan["deferred_verifications"][:COMMENT_ITEM_LIMIT]
            )
            + (
                f" (and {len(plan['deferred_verifications']) - COMMENT_ITEM_LIMIT} more; see the CI plan)"
                if len(plan["deferred_verifications"]) > COMMENT_ITEM_LIMIT
                else ""
            )
            + ".",
        ]
    if plan and plan.get("deferred_images") and state not in {"running", "success"}:
        images = plan["deferred_images"]
        lines += [
            "",
            "Pending image builds: "
            + ", ".join(f"`{item}`" for item in images[:COMMENT_ITEM_LIMIT])
            + (
                f" (and {len(images) - COMMENT_ITEM_LIMIT} more; see the CI plan)"
                if len(images) > COMMENT_ITEM_LIMIT
                else ""
            )
            + ".",
        ]
    lines += [
        "",
        f"Revision `{pull['head']['sha'][:12]}` · [CI run, attempt {run['run_attempt']}]({url})",
    ]
    if state in {"awaiting-approval", "legacy", "stale"}:
        lines += [
            "",
            "Approval applies to this revision only. New commits or a changed merge base require fresh qualification.",
        ]
    return titles[state], "\n".join(lines)


def owned_checks(client, repo: str, pull: dict) -> list[dict]:
    return [
        item
        for item in pages(
            client,
            f"repos/{repo}/commits/{pull['head']['sha']}/check-runs?check_name=PR%20Gate&filter=all",
            "check_runs",
        )
        if item.get("name") == CHECK
        and item.get("app", {}).get("id") == APP_ID
        and str(item.get("external_id", "")).startswith(f"{PREFIX}{pull['number']}:")
    ]


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
) -> dict:
    title, summary = presentation(state, repo, pull, run, plan)
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
    comments = list(pages(client, f"repos/{repo}/issues/{pull['number']}/comments"))
    owned = [
        item
        for item in comments
        if item.get("user", {}).get("login") == "github-actions[bot]"
        and item.get("user", {}).get("type") == "Bot"
        and item.get("body", "").startswith(MARKER)
    ]
    comment_body = f"{MARKER}\n{summary}\n"
    result = {
        "pr": pull["number"],
        "state": state,
        "check": None if state == "legacy" else payload,
        "comment": (
            comment_body
            if owned
            or state in {"awaiting-approval", "legacy", "stale"}
            or (plan and plan.get("deferred_verifications"))
            else None
        ),
    }
    if dry_run:
        return result
    # Recheck after reading artifacts/comments; an old completion must never
    # overwrite the status of a newer run, attempt, or contributor commit.
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
        return {"ignored": "superseded while preparing status"}
    if state != "legacy":
        existing = [item for item in checks if item.get("external_id") == external]
        if not existing:
            # An approval rerun should resolve its own pending check, without
            # reopening a completed check or mutating another workflow run.
            run_prefix = f"{PREFIX}{pull['number']}:{run['id']}:"
            existing = [
                item
                for item in checks
                if item.get("status") == "in_progress"
                and str(item.get("external_id", "")).startswith(run_prefix)
            ]
        if existing:
            update = dict(payload)
            update.pop("head_sha")
            client.write(
                f"repos/{repo}/check-runs/{
                    max(existing, key=lambda item: item['id'])['id']
                }",
                update,
                method="PATCH",
            )
        else:
            client.write(f"repos/{repo}/check-runs", payload)
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
        # On the first attempt, the required check is already Expected/pending.
        # Do not invent the source identity before Plan has produced it.
        if not checks:
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
        if state != "failure" and pull.get("merge_commit_sha") != plan["source_sha"]:
            state = "stale"
    except (ManualCIError, KeyError, ValueError, TypeError):
        if not checks and not any(job.get("name") == "CI Execution" for job in jobs):
            raise
        state = "failure"
        plan = None  # Never interpolate malformed artifact content into a comment.
    return publish(client, repo, pull, run, state, plan, checks, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        event = json.loads(args.event.read_text()) if args.event else {}
        run_id = args.run_id or int(
            event.get("inputs", {}).get("run_id") or event["workflow_run"]["id"]
        )
        result = reconcile(
            StatusClient(),
            os.environ["GITHUB_REPOSITORY"],
            run_id,
            dry_run=args.dry_run,
        )
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
