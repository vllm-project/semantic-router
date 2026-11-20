"""Read-only resolution of trusted events to one qualification concurrency key."""

from __future__ import annotations

from urllib.parse import urlencode

from manual_ci import ManualCIError
from pr_ci_shared import current_pull, latest_run, pages


def positive_id(value):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        result = int(value)
        if result > 0:
            return result
    raise ManualCIError("Invalid qualification run identity.")


def community_run(client, repo, event):
    payload = event.get("workflow_run") or {}
    identity = positive_id(payload.get("id"))
    run = client.request(f"repos/{repo}/actions/runs/{identity}")
    workflow = client.request(f"repos/{repo}/actions/workflows/community.yml")
    if (
        event.get("action") != "completed"
        or run.get("id") != identity
        or run.get("path") != ".github/workflows/community.yml"
        or run.get("workflow_id") != workflow.get("id")
        or (run.get("repository") or {}).get("full_name") != repo
        or run.get("event") not in {"pull_request", "pull_request_review"}
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or any(
            payload.get(key) != run.get(key)
            for key in ("head_sha", "head_branch", "run_attempt")
        )
        or (payload.get("head_repository") or {}).get("full_name")
        != (run.get("head_repository") or {}).get("full_name")
    ):
        raise ManualCIError("Community event is not a current successful completion.")
    positive_id(run.get("run_attempt"))
    return run


def resolve_event(client, repo, event):
    """Return only typed run/mode outputs; never execute or write PR content."""
    if not isinstance(event, dict):
        raise ManualCIError("Invalid status event.")
    payload = event.get("workflow_run")
    if payload is None:
        run_id = positive_id((event.get("inputs") or {}).get("run_id"))
        return {"run_id": run_id, "mode": "publish"}
    identity = positive_id(payload.get("id"))
    actual = client.request(f"repos/{repo}/actions/runs/{identity}")
    if actual.get("path") == ".github/workflows/pr.yml":
        workflow = client.request(f"repos/{repo}/actions/workflows/pr.yml")
        if (
            actual.get("id") != identity
            or actual.get("workflow_id") != workflow.get("id")
            or actual.get("event") != "pull_request"
            or (actual.get("repository") or {}).get("full_name") != repo
        ):
            raise ManualCIError("Invalid Pull Request workflow identity.")
        return {"run_id": identity, "mode": "publish"}
    observed = community_run(client, repo, event)
    pull = current_pull(client, repo, observed)
    if pull is None:
        raise ManualCIError("Community event has no unique current open PR.")
    workflow = client.request(f"repos/{repo}/actions/workflows/pr.yml")
    query = urlencode({"event": "pull_request", "head_sha": observed["head_sha"]})
    candidates = [
        run
        for run in pages(
            client,
            f"repos/{repo}/actions/workflows/pr.yml/runs?{query}",
            "workflow_runs",
        )
        if run.get("head_branch") == observed.get("head_branch")
        and (run.get("head_repository") or {}).get("full_name")
        == (observed.get("head_repository") or {}).get("full_name")
    ]
    if not candidates:
        raise ManualCIError("No current qualification run.")
    selected = max(candidates, key=lambda run: (run["created_at"], run["id"]))
    run = client.request(f"repos/{repo}/actions/runs/{positive_id(selected['id'])}")
    if (
        run.get("workflow_id") != workflow.get("id")
        or run.get("path") != ".github/workflows/pr.yml"
        or run.get("event") != "pull_request"
        or (run.get("repository") or {}).get("full_name") != repo
        or run.get("head_sha") != pull["head"]["sha"]
        or run.get("head_branch") != observed.get("head_branch")
        or (run.get("head_repository") or {}).get("full_name")
        != (observed.get("head_repository") or {}).get("full_name")
        or not latest_run(client, repo, run)
    ):
        raise ManualCIError("Qualification identity changed while resolving event.")
    return {"run_id": positive_id(run["id"]), "mode": "pending"}
