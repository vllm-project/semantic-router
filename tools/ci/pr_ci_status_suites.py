"""Read-only, head-bound check-suite and commit-rollup evidence."""

from __future__ import annotations

import json

from manual_ci import ManualCIError

MAX_SUITES = 100
STATUSES = {"queued", "in_progress", "completed", "waiting", "pending", "requested"}
CONCLUSIONS = {
    "success",
    "failure",
    "neutral",
    "cancelled",
    "skipped",
    "timed_out",
    "action_required",
    "stale",
    "startup_failure",
}
ROLLUP_STATES = {"SUCCESS", "FAILURE", "ERROR", "PENDING", "EXPECTED"}


def read_suites(client, repo, head, identities, app_id):
    if any(type(identity) is not int or identity <= 0 for identity in identities):
        raise ManualCIError("Invalid diagnostic check-suite identity.")
    identities = set(identities)
    if len(identities) > MAX_SUITES:
        raise ManualCIError("Diagnostic check-suite inventory exceeds the limit.")
    suites = {}
    for identity in sorted(identities):
        suite = client.request(f"repos/{repo}/check-suites/{identity}")
        if (
            not isinstance(suite, dict)
            or type(suite.get("id")) is not int
            or suite["id"] != identity
            or suite.get("head_sha") != head
            or not isinstance(suite.get("app"), dict)
            or suite["app"].get("id") != app_id
            or suite.get("status") not in STATUSES
            or suite.get("conclusion") not in CONCLUSIONS | {None}
            or (suite["status"] == "completed" and suite.get("conclusion") is None)
            or (suite["status"] != "completed" and suite.get("conclusion") is not None)
        ):
            raise ManualCIError(
                "Check-suite evidence does not match the requested head."
            )
        suites[identity] = {
            "id": identity,
            "status": suite["status"],
            "conclusion": suite["conclusion"],
        }
    return suites


def read_rollup(client, repo, head):
    owner, name = repo.split("/")
    query = (
        "query($owner:String!,$repo:String!,$head:GitObjectID!){"
        "repository(owner:$owner,name:$repo){object(oid:$head){"
        "... on Commit{oid statusCheckRollup{state}}}}}"
    )
    response = client.gh(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-f",
            f"owner={owner}",
            "-f",
            f"repo={name}",
            "-f",
            f"head={head}",
        ],
        read=True,
        text=True,
    )
    if response.returncode:
        raise ManualCIError("Commit rollup evidence is unavailable.")
    result = json.loads(response.stdout)
    if not isinstance(result, dict) or result.get("errors"):
        raise ManualCIError("Commit rollup evidence is incomplete.")
    data = result.get("data")
    repository = data.get("repository") if isinstance(data, dict) else None
    commit = repository.get("object") if isinstance(repository, dict) else None
    if (
        not isinstance(commit, dict)
        or commit.get("oid") != head
        or not isinstance(commit.get("statusCheckRollup"), dict)
        or commit["statusCheckRollup"].get("state") not in ROLLUP_STATES
    ):
        raise ManualCIError("Commit rollup does not match the requested head.")
    return commit["statusCheckRollup"]["state"]


def successful(suite):
    return suite["status"] == "completed" and suite["conclusion"] == "success"
