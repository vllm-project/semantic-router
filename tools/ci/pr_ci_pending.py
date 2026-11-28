"""Bounded recovery of publisher-owned, current-attempt pending visibility."""

from __future__ import annotations

import re
import subprocess

from manual_ci import ManualCIError, plan_for_run, source_identity
from pr_ci_events import positive_id, resolve_event
from pr_ci_shared import (
    APP_ID,
    CHECK,
    MAX_IDENTIFIERS,
    PREFIX,
    artifact,
    current_pull,
    latest_run,
    pages,
    presentation,
    publisher_ready,
    validate_plan,
    verdict,
)

MARKER = "semantic-router-pending:v1:"
PATTERN = re.compile(
    r"semantic-router-pending:v1:(claim|replacement|retired):"
    r"([1-9][0-9]{0,19}):([1-9][0-9]{0,19}):([1-9][0-9]{0,19})"
)


def provenance(check):
    details = check.get("output") or {}
    if not isinstance(details, dict):
        raise ManualCIError("Invalid pending recovery output.")
    text = details.get("text", "")
    if text is None:
        return None
    if not isinstance(text, str):
        raise ManualCIError("Invalid pending recovery provenance.")
    match = PATTERN.fullmatch(text)
    if match:
        return match[1], int(match[2]), int(match[3]), int(match[4])
    if text.startswith(MARKER):
        raise ManualCIError("Invalid pending recovery provenance.")
    return None


def marker(phase, trigger, linked):
    return f"{MARKER}{phase}:{trigger[0]}:{trigger[1]}:{linked}"


def pending(check):
    return check.get("status") == "in_progress" and check.get("conclusion") is None


def fingerprint(check):
    return (
        check.get("id"),
        check.get("name"),
        check.get("head_sha"),
        check.get("app", {}).get("id"),
        check.get("external_id"),
        check.get("status"),
        check.get("conclusion"),
        check.get("check_suite", {}).get("id"),
        provenance(check),
    )


def verify_pending(client, repo, expected):
    current = client.request(f"repos/{repo}/check-runs/{expected['id']}")
    if (
        not isinstance(current, dict)
        or not pending(current)
        or fingerprint(current) != fingerprint(expected)
    ):
        raise ManualCIError("Pending write target changed.")


def inventory(client, repo, pull, run):
    result = {}
    for view in ("all", "latest"):
        checks = list(
            pages(
                client,
                f"repos/{repo}/commits/{run['head_sha']}/check-runs?check_name=PR%20Gate&filter={view}",
                "check_runs",
            )
        )
        if len(checks) > MAX_IDENTIFIERS or any(
            check.get("name") != CHECK
            or check.get("head_sha") != run["head_sha"]
            or type(check.get("id")) is not int
            or check["id"] <= 0
            for check in checks
        ):
            raise ManualCIError("Pending check inventory is incomplete.")
        result[view] = {
            check["id"]: check
            for check in checks
            if check.get("app", {}).get("id") == APP_ID
            and check.get("external_id")
            == f"{PREFIX}{pull['number']}:{run['id']}:{run['run_attempt']}"
        }
        if view == "all":
            result["publisher_ids"] = {
                check["id"]
                for check in checks
                if check.get("app", {}).get("id") == APP_ID
                and str(check.get("external_id", "")).startswith(
                    f"{PREFIX}{pull['number']}:"
                )
            }
    if not set(result["latest"]) <= set(result["all"]) or any(
        fingerprint(result["latest"][identity]) != fingerprint(result["all"][identity])
        for identity in result["latest"]
    ):
        raise ManualCIError("Pending visibility evidence changed.")
    for check in result["all"].values():
        meta = provenance(check)
        retired = (
            meta is not None
            and meta[0] == "retired"
            and check.get("status") == "completed"
            and check.get("conclusion") == "neutral"
            and meta[3] != check["id"]
            and meta[3] in result["all"]
        )
        if not retired and not pending(check):
            raise ManualCIError("A current completed or ambiguous verdict exists.")
    return result


def collect(client, repo, run_id):
    run = client.request(f"repos/{repo}/actions/runs/{run_id}")
    workflow = client.request(f"repos/{repo}/actions/workflows/pr.yml")
    if (
        run.get("id") != run_id
        or run.get("path") != ".github/workflows/pr.yml"
        or run.get("workflow_id") != workflow.get("id")
        or run.get("event") != "pull_request"
        or (run.get("repository") or {}).get("full_name") != repo
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or type(run.get("run_attempt")) is not int
        or run["run_attempt"] <= 0
        or not latest_run(client, repo, run)
    ):
        raise ManualCIError("Pending qualification is not current.")
    pull = current_pull(client, repo, run)
    if pull is None:
        raise ManualCIError("Pending qualification has no current open PR.")
    checks = inventory(client, repo, pull, run)
    if not any(pending(c) for c in checks["all"].values()):
        raise ManualCIError("No current pending check exists.")
    jobs = list(
        pages(
            client,
            f"repos/{repo}/actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs",
            "jobs",
        )
    )
    owned = checks["publisher_ids"]
    plan = artifact(client, repo, run, "ci-plan", "plan.json", 16 * 1024 * 1024)
    validate_plan(client, repo, run, plan)
    report = artifact(client, repo, run, "ci-gate", "gate.json", 256 * 1024)
    if (
        verdict(run, plan, report, [j for j in jobs if j.get("id") not in owned])
        != "awaiting-approval"
        or not publisher_ready(client, repo, plan["base_sha"])
        or plan_for_run(
            client, repo, run, source_identity(client, repo, pull["number"])
        )
        != plan
    ):
        raise ManualCIError("Qualification does not permit pending recovery.")
    snapshot = {"run": run, "pull": pull, "plan": plan}
    final_run = client.request(f"repos/{repo}/actions/runs/{run_id}")
    final_pull = client.request(f"repos/{repo}/pulls/{pull['number']}")
    if signature({"run": final_run, "pull": final_pull, "plan": plan}) != signature(
        snapshot
    ) or not latest_run(client, repo, final_run):
        raise ManualCIError("Pending qualification changed during collection.")
    # Receipt/source reads may outlive a pending verdict. Never discard a
    # completed check observed while collecting the evidence for a write.
    checks = inventory(client, repo, pull, run)
    return {**snapshot, **checks}


def signature(snapshot):
    run, pull = snapshot["run"], snapshot["pull"]
    return (
        tuple(
            run.get(k)
            for k in (
                "id",
                "workflow_id",
                "path",
                "event",
                "head_sha",
                "head_branch",
                "run_attempt",
                "check_suite_id",
                "status",
                "conclusion",
            )
        ),
        (run.get("repository") or {}).get("full_name"),
        (run.get("head_repository") or {}).get("full_name"),
        tuple(pull.get(k) for k in ("number", "state", "merge_commit_sha")),
        tuple(
            (
                pull[side].get("sha"),
                pull[side].get("ref"),
                (pull[side].get("repo") or {}).get("full_name"),
            )
            for side in ("head", "base")
        ),
        snapshot["plan"],
    )


def output(snapshot, text):
    title, summary = presentation(
        "awaiting-approval",
        snapshot["repo"],
        snapshot["pull"],
        snapshot["run"],
        snapshot["plan"],
    )
    return {"title": title, "summary": summary, "text": text}


def repair(client, repo, run_id, trigger, *, event=None, dry_run=False):
    original_snapshot = collect(client, repo, run_id)
    baseline = signature(original_snapshot)

    def fresh():
        if event is not None:
            resolution = resolve_event(client, repo, event)
            if resolution != {"run_id": run_id, "mode": "pending"}:
                raise ManualCIError("Pending event identity changed.")
        snapshot = collect(client, repo, run_id)
        if signature(snapshot) != baseline:
            raise ManualCIError("Pending source identity changed.")
        snapshot["repo"] = repo
        return snapshot

    def result(kind, identity):
        return {
            "pr": original_snapshot["pull"]["number"],
            "run_id": run_id,
            "attempt": original_snapshot["run"]["run_attempt"],
            "state": "awaiting-approval",
            "recovery": kind,
            "check_id": identity,
        }

    snapshot = fresh()
    active = [c for c in snapshot["all"].values() if pending(c)]
    visible = [c for c in active if c["id"] in snapshot["latest"]]
    if len(visible) > 1:
        raise ManualCIError("Multiple current pending checks are visible.")
    # A persisted claim is an at-most-once checkpoint, including a timed-out POST.
    claims = [c for c in active if (provenance(c) or (None,))[0] == "claim"]
    replacements = [c for c in active if (provenance(c) or (None,))[0] == "replacement"]
    origin = replacement = None
    if claims:
        if len(claims) != 1:
            raise ManualCIError("Ambiguous pending recovery claims.")
        origin = claims[0]
        claim = provenance(origin)
        if claim[3] != origin["id"]:
            raise ManualCIError("Invalid pending recovery claim target.")
        matches = [
            c for c in replacements if provenance(c)[1:] == (*claim[1:3], origin["id"])
        ]
        if len(matches) > 1 or len(active) != 1 + len(matches):
            raise ManualCIError("Ambiguous pending recovery replacement.")
        if not matches:
            return result("unconfirmed", origin["id"])
        replacement = matches[0]
        trigger = claim[1:3]
    else:
        if len(active) != 1:
            raise ManualCIError("Ambiguous current pending check.")
        candidate = active[0]
        if visible:
            return {"ignored": "current pending Gate is already visible"}
        meta = provenance(candidate)
        if meta:
            parent = snapshot["all"].get(meta[3])
            receipt = provenance(parent) if parent else None
            if meta[0] != "replacement" or receipt != (
                "retired",
                *meta[1:3],
                candidate["id"],
            ):
                raise ManualCIError("Pending replacement visibility was not confirmed.")
            if meta[1:3] == trigger:
                return result("unconfirmed", candidate["id"])
        origin = candidate
        if dry_run:
            return result("eligible", origin["id"])
        snapshot = fresh()
        current = snapshot["all"].get(origin["id"])
        if current != origin or visible_ids(snapshot):
            raise ManualCIError("Pending inventory changed before claim.")
        claim_text = marker("claim", trigger, origin["id"])
        verify_pending(client, repo, current)
        client.write(
            f"repos/{repo}/check-runs/{origin['id']}",
            {"output": output(snapshot, claim_text)},
            method="PATCH",
        )
        snapshot = fresh()
        current = snapshot["all"].get(origin["id"])
        if (
            current is None
            or provenance(current) != ("claim", *trigger, origin["id"])
            or len([c for c in snapshot["all"].values() if pending(c)]) != 1
            or visible_ids(snapshot)
        ):
            raise ManualCIError("Pending claim could not be verified.")
        payload = {
            "name": CHECK,
            "head_sha": snapshot["run"]["head_sha"],
            "external_id": current["external_id"],
            "status": "in_progress",
            "details_url": f"https://github.com/{repo}/actions/runs/{run_id}",
            "output": output(snapshot, marker("replacement", trigger, origin["id"])),
        }
        verify_pending(client, repo, current)
        client.write(f"repos/{repo}/check-runs", payload)
        snapshot = fresh()
        matches = [
            c
            for c in snapshot["all"].values()
            if pending(c) and provenance(c) == ("replacement", *trigger, origin["id"])
        ]
        if len(matches) != 1:
            return result("unconfirmed", origin["id"])
        replacement = matches[0]
    if dry_run:
        return result("eligible", replacement["id"])
    snapshot = fresh()
    current = snapshot["all"].get(origin["id"])
    target = snapshot["all"].get(replacement["id"])
    if target is None or target["id"] not in snapshot["latest"]:
        return result("unconfirmed", replacement["id"])
    if (
        current is None
        or not pending(current)
        or not pending(target)
        or provenance(current) != ("claim", *trigger, origin["id"])
        or provenance(target) != ("replacement", *trigger, origin["id"])
        or visible_ids(snapshot) != {target["id"]}
    ):
        raise ManualCIError("Pending retirement evidence changed.")
    verify_pending(client, repo, target)
    verify_pending(client, repo, current)
    client.write(
        f"repos/{repo}/check-runs/{origin['id']}",
        {
            "status": "completed",
            "conclusion": "neutral",
            "output": {
                "title": "Superseded pending Gate",
                "summary": f"Pending Gate check {target['id']} is visible. Qualification still awaits approval.",
                "text": marker("retired", trigger, target["id"]),
            },
        },
        method="PATCH",
    )
    snapshot = fresh()
    if visible_ids(snapshot) != {target["id"]}:
        return result("unconfirmed", target["id"])
    verify_pending(client, repo, snapshot["all"][target["id"]])
    return result("restored", target["id"])


def visible_ids(snapshot):
    return {
        c["id"]
        for c in snapshot["all"].values()
        if pending(c) and c["id"] in snapshot["latest"]
    }


def recover_pending(client, repo, event, run_id=None, *, dry_run=False):
    try:
        resolved = resolve_event(client, repo, event)
        if resolved.get("mode") != "pending" or (
            run_id is not None and resolved["run_id"] != run_id
        ):
            return {"ignored": "event is not the current pending recovery source"}
        trigger = event["workflow_run"]
        return repair(
            client,
            repo,
            resolved["run_id"],
            (positive_id(trigger["id"]), positive_id(trigger["run_attempt"])),
            event=event,
            dry_run=dry_run,
        )
    except (
        ManualCIError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        subprocess.TimeoutExpired,
    ):
        return {"ignored": "pending recovery evidence unavailable or changed"}
