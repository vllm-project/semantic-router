"""Carry validated automatic receipts within the original PR workflow run."""

from __future__ import annotations

import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

from check_ci_gate import receipt_artifact_names
from ci_plan import component_batches, digest
from ci_results import make_receipt
from execution_batches import (
    e2e_batches,
    expected_dispatch_jobs,
    platform_batches,
    scheduled_verifications,
)
from manual_ci import PLAN_LIMIT, GitHubClient, ManualCIError, read_artifact

PAGE_SIZE = 100


def select_receipts(plan: dict, previous: dict, receipts: list[dict]) -> list[dict]:
    """Authorization is not part of a contract; source, base and evidence are."""
    if (
        previous.get("schema_version") != 1
        or previous.get("profile") != "pr"
        or previous.get("source_sha") != plan["source_sha"]
        or previous.get("base_sha") != plan["base_sha"]
        or previous.get("plan_sha256")
        != digest(
            {key: value for key, value in previous.items() if key != "plan_sha256"}
        )
    ):
        return []
    old = {record["id"]: record for record in previous["verifications"]}
    selected = []
    for record in plan["verifications"]:
        # Automatic contracts currently have no image dependencies. Do not reuse
        # image consumers without also preserving their exact build receipts.
        if (
            record["execution"] != "automatic"
            or record["images"]
            or old.get(record["id"]) != record
            or record["contract_sha256"]
            != digest(
                {
                    key: value
                    for key, value in record.items()
                    if key != "contract_sha256"
                }
            )
        ):
            continue
        matches = [receipt for receipt in receipts if receipt.get("id") == record["id"]]
        if len(matches) != 1:
            continue
        receipt = matches[0]
        try:
            # Rebuild through the existing evidence validator, requiring hosted
            # Actions observations and exact receipt identity, not just success.
            expected = make_receipt(
                record,
                receipt["evidence"],
                source_sha=plan["source_sha"],
                execution_platform=record["platform"],
                environ={
                    "GITHUB_ACTIONS": "true",
                    "CI_RUNNER_ENVIRONMENT": record["runner_environment"],
                },
            )
            if receipt == expected and receipt["result"] == "success":
                selected.append(receipt)
        except (KeyError, ValueError, TypeError):
            continue
    return selected


def previous_receipts(client, repo: str, run_id: int, previous: dict) -> list[dict]:
    receipts = []
    names = receipt_artifact_names(previous)
    page = 1
    while True:
        artifacts = client.request(
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page={PAGE_SIZE}&page={page}"
        )["artifacts"]
        for artifact in artifacts:
            if artifact.get("name") not in names or artifact.get("expired"):
                continue
            if (
                artifact.get("workflow_run", {}).get("id") != run_id
                or not 0 < artifact.get("size_in_bytes", 0) <= PLAN_LIMIT
            ):
                continue
            raw = client.raw(
                f"repos/{repo}/actions/artifacts/{int(artifact['id'])}/zip"
            )
            if len(raw) > PLAN_LIMIT:
                continue
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                if sum(entry.file_size for entry in entries) > PLAN_LIMIT:
                    continue
                # Never extract archive paths or execute artifact contents.
                for entry in entries:
                    if entry.filename.endswith(".json"):
                        receipt = json.loads(archive.read(entry))
                        if isinstance(receipt, dict):
                            receipts.append(receipt)
        if len(artifacts) < PAGE_SIZE:
            return receipts
        page += 1


def reuse_automatic_results(
    plan: dict, output: Path, *, client=None, environ=None
) -> None:
    env = os.environ if environ is None else environ
    client = GitHubClient() if client is None else client
    try:
        repo = env["GITHUB_REPOSITORY"]
        run_id, attempt = int(env["GITHUB_RUN_ID"]), int(env["GITHUB_RUN_ATTEMPT"])
        event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text())
        run = client.request(
            f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt - 1}"
        )
        workflow = client.request(f"repos/{repo}/actions/workflows/pr.yml")
        if (
            env.get("GITHUB_EVENT_NAME") != "pull_request"
            or plan["profile"] != "pr"
            or run.get("id") != run_id
            or run.get("run_attempt") != attempt - 1
            or run.get("status") != "completed"
            or run.get("event") != "pull_request"
            or run.get("repository", {}).get("full_name") != repo
            or run.get("path") != ".github/workflows/pr.yml"
            or run.get("workflow_id") != workflow.get("id")
            or run.get("head_sha") != event["pull_request"]["head"]["sha"]
            or env.get("GITHUB_SHA") != plan["source_sha"]
        ):
            return
        artifacts = client.request(
            f"repos/{repo}/actions/runs/{run_id}/artifacts?name=ci-plan&per_page={PAGE_SIZE}"
        )["artifacts"]
        candidates = [
            artifact
            for artifact in artifacts
            if artifact.get("name") == "ci-plan"
            and artifact.get("workflow_run", {}).get("id") == run_id
            and not artifact.get("expired")
        ]
        if len(candidates) != 1:
            return
        previous = read_artifact(client, repo, candidates[0], "plan.json", PLAN_LIMIT)
        receipts = select_receipts(
            plan, previous, previous_receipts(client, repo, run_id, previous)
        )
    except (
        ManualCIError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OSError,
        zipfile.BadZipFile,
        subprocess.TimeoutExpired,
    ):
        # Cache availability must never weaken qualification or block execution.
        print("::notice::Prior CI receipts unavailable; executing automatic checks.")
        return
    if not receipts:
        return
    output.mkdir(parents=True, exist_ok=True)
    for receipt in receipts:
        # Contracts, not archive paths, own these filenames.
        (output / f"{receipt['id']}.json").write_text(json.dumps(receipt) + "\n")
    plan["reused_verification_ids"] = sorted(receipt["id"] for receipt in receipts)
    scheduled = scheduled_verifications(plan)
    plan["component_batches"] = component_batches(scheduled)
    plan["platform_batches"] = platform_batches(scheduled)
    plan["e2e_batches"] = e2e_batches(scheduled)
    plan["expected_dispatch_jobs"] = expected_dispatch_jobs(plan)
    plan["plan_sha256"] = digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    print(f"Reusing {len(receipts)} validated automatic CI receipts from this PR run.")
