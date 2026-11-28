"""Shared, read-only PR qualification and publisher identity contracts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from urllib.parse import quote, urlencode

import manual_ci as _manual_ci
from manual_ci import (
    MERGE_EVIDENCE_UNAVAILABLE,
    ManualCIError,
    merge_revision_matches,
    read_artifact,
)

CAPABILITIES = _manual_ci.PUBLISHER_FILES
MAX_PAGES = _manual_ci.MAX_PAGES
latest_run = _manual_ci.latest_run
pages = _manual_ci.pages
publisher_ready = _manual_ci.status_publisher_available

CHECK = "PR Gate"
EXECUTION_CHECK = "CI Execution"
MARKER = "<!-- semantic-router-ci-status:v1 -->"
PREFIX = "semantic-router-ci:"
APP_ID = 15368  # GitHub Actions; also the branch-protection check owner.
SHA = re.compile(r"[0-9a-f]{40}")
IDENTIFIER = re.compile(r"[a-zA-Z0-9_.-]{1,160}")
MAX_IDENTIFIERS = 1000
COMMENT_ITEM_LIMIT = 30


def qualification_only_rules(client, repo: str, branch: str) -> bool:
    """Keep the compatibility publisher until both replacement checks are required."""
    base = quote(branch, safe="")
    try:
        required = [
            (check["context"], check.get("integration_id"))
            for rule in pages(client, f"repos/{repo}/rules/branches/{base}")
            if rule["type"] == "required_status_checks"
            for check in rule["parameters"]["required_status_checks"]
        ]
        if any(name == CHECK for name, _ in required) or not {
            (_manual_ci.PR_QUALIFICATION_CONTEXT, APP_ID),
            (EXECUTION_CHECK, APP_ID),
        }.issubset(required):
            return False
        # Classic branch protection applies independently of rulesets.
        protection = client.request(f"repos/{repo}/branches/{base}")["protection"]
        return protection["enabled"] is False
    except (
        ManualCIError,
        KeyError,
        TypeError,
        OSError,
        ValueError,
        subprocess.TimeoutExpired,
    ):
        return False


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
    if not merge_revision_matches(
        client, repo, plan["source_sha"], plan["base_sha"], run["head_sha"]
    ):
        raise ManualCIError(
            "Qualification plan does not belong to this PR merge revision."
        )
    selected = identifiers(plan.get("expected_verification_ids"))
    deferred = identifiers(plan.get("deferred_verifications"))
    required = identifiers(plan.get("required_verification_ids"))
    identifiers(plan.get("deferred_images"))
    if set(selected) & set(deferred) or set(required) != set(selected) | set(deferred):
        raise ManualCIError("Qualification inventories do not partition the plan.")


# Supports strict CI Execution qualification.
# The CI Plan workflow reads this capability from the tested base.
def verdict(run: dict, plan: dict, report: dict, jobs: list[dict]) -> str:
    """A successful automatic workflow is not necessarily full qualification."""
    by_name = {job.get("name"): job for job in jobs}
    execution = by_name.get("CI Execution", {})
    waiting = bool(plan["deferred_verifications"] or plan["deferred_images"])
    # The native check rejects incomplete qualification even when every
    # dispatched check passed. The receipt still determines the published state.
    execution_blocked = (waiting or plan.get("draft") is True) and execution.get(
        "conclusion"
    ) == "failure"
    expected_conclusion = "failure" if execution_blocked else "success"
    if run.get("conclusion") != expected_conclusion or any(
        job.get("conclusion") not in {"success", "skipped"}
        for job in jobs
        if not (execution_blocked and job is execution)
    ):
        return "failure"
    if (
        by_name.get("CI / Gate", {}).get("conclusion") != "success"
        or execution.get("conclusion") != expected_conclusion
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
    expected_status = "awaiting-approval" if waiting else "success"
    if report.get("status") != expected_status:
        raise ManualCIError("CI receipt does not match the remaining qualification.")
    return expected_status


def presentation(
    state: str,
    repo: str,
    pull: dict,
    run: dict,
    plan: dict | None,
    reason: str | None = None,
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
    if reason == MERGE_EVIDENCE_UNAVAILABLE:
        titles["failure"] = "CI source evidence unavailable"
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
            (
                "The original planned merge commit could not be read. Qualification stays blocked because its source equivalence could not be verified. Retain the Plan and run evidence for investigation."
                if reason == MERGE_EVIDENCE_UNAVAILABLE
                else "A check failed or its required evidence could not be verified. Open the run below for the failure; approval alone does not turn a failed test green."
            )
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
            "Approval applies to this revision only. New commits require fresh qualification.",
        ]
    return titles[state], "\n".join(lines)


def owned_checks(client, repo: str, pull: dict, *, latest=False) -> list[dict]:
    check_filter = "latest" if latest else "all"
    return [
        item
        for item in pages(
            client,
            f"repos/{repo}/commits/{pull['head']['sha']}/check-runs?check_name=PR%20Gate&filter={check_filter}",
            "check_runs",
        )
        if item.get("name") == CHECK
        and item.get("app", {}).get("id") == APP_ID
        and str(item.get("external_id", "")).startswith(f"{PREFIX}{pull['number']}:")
    ]
