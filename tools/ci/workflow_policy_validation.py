"""Repository-specific GitHub Actions lifecycle and safety policy checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import yaml
from classify_pr_changes import NIGHTLY_IMAGES, PRODUCTION_RELEASE_IMAGES
from domain_registry import job_records, load_domain_registry
from execution_batches import ALL_DISPATCH_JOBS, dispatch_job
from image_artifacts import publication_tags
from verification_catalog import catalog_errors

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
LOCAL_WORKFLOW_PREFIX = "./.github/workflows/"
REQUIRED_APPROVAL_CONDITION = "#approved-reviews-by >= 2"
RELEASE_IMAGES = set(PRODUCTION_RELEASE_IMAGES)


class WorkflowLike(Protocol):
    path: Path
    relative_path: str
    data: dict[str, Any]
    events: dict[str, Any]
    jobs: dict[str, Any]


def local_target(job: dict[str, Any]) -> str | None:
    uses = job.get("uses")
    if isinstance(uses, str) and uses.startswith(LOCAL_WORKFLOW_PREFIX):
        return uses.removeprefix(LOCAL_WORKFLOW_PREFIX)
    return None


def needs(job: dict[str, Any]) -> set[str]:
    value = job.get("needs", [])
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def validate_gate_transport(gate: dict[str, Any], errors: list[str]) -> None:
    steps = gate.get("steps", [])
    reconciliations = [
        step for step in steps if "check_ci_gate.py" in step.get("run", "")
    ]
    expected = {
        name: {"result": "${{ needs." + name + ".result }}"}
        for name in ALL_DISPATCH_JOBS
    }
    try:
        actual = json.loads(
            reconciliations[0].get("env", {}).get("EXECUTOR_RESULTS", "")
        )
    except (ValueError, TypeError, IndexError):
        actual = None
    if len(reconciliations) != 1 or actual != expected:
        errors.append(
            "ci.yml gate must pass only every prerequisite's result, without job outputs"
        )
    if not any(
        step.get("uses", "").startswith("actions/download-artifact@")
        and step.get("with", {}).get("name") == "ci-plan"
        and step.get("with", {}).get("path") == ".agent-harness/ci"
        for step in steps
    ) or not any(
        "--plan .agent-harness/ci/plan.json" in step.get("run", "")
        for step in reconciliations
    ):
        errors.append(
            "ci.yml gate must read the independently downloaded ci-plan artifact"
        )


def validate_pr_contract(workflows: dict[str, WorkflowLike], errors: list[str]) -> None:
    dispatcher = workflows.get("pr.yml")
    if not dispatcher or "pull_request" not in dispatcher.events:
        errors.append("missing pull-request dispatcher")
        return
    gate = dispatcher.jobs.get("pr-gate", {})
    if (
        gate.get("name") != "PR Gate"
        or gate.get("if") != "always()"
        or needs(gate) != {"ci"}
    ):
        errors.append(
            "pr.yml must retain the always-running stable PR Gate behind shared CI"
        )
    for filename, profile in (
        ("pr.yml", "pr"),
        ("main.yml", "main"),
        ("nightly-build.yml", "nightly"),
        ("release.yml", "release"),
    ):
        workflow = workflows.get(filename)
        call = workflow.jobs.get("ci", {}) if workflow else {}
        if (
            local_target(call) != "ci.yml"
            or call.get("with", {}).get("profile") != profile
        ):
            errors.append(
                f"{filename}: must use the single CI planner/executor with profile {profile}"
            )
        for job in workflow.jobs.values() if workflow else ():
            if local_target(job) in {
                record["workflow"].split("/")[-1] for record in job_records().values()
            }:
                errors.append(
                    f"{filename}: duplicated verification dispatch outside ci.yml"
                )
    shared = workflows.get("ci.yml")
    if not shared:
        errors.append("missing shared ci.yml")
        return
    expected = set(ALL_DISPATCH_JOBS)
    gate = shared.jobs.get("gate", {})
    if needs(gate) != expected or gate.get("if") != "always()":
        errors.append(
            "ci.yml gate must aggregate every executor and build prerequisite"
        )
    validate_gate_transport(gate, errors)
    text = shared.path.read_text()
    if (
        "check_ci_gate.py" not in text
        or "--plan" not in text
        or "ci-result-*" not in text
    ):
        errors.append(
            "ci.yml must reconcile execution artifacts against the pre-execution plan"
        )
    for record in job_records().values():
        call = shared.jobs.get(dispatch_job(record), {})
        if local_target(call) != Path(record["workflow"]).name:
            errors.append(f"ci.yml: no executor for {record['workflow']}")
    errors.extend(catalog_errors(load_domain_registry()))
    planner = workflows.get("ci-changes.yml")
    if not planner or "tools/ci/ci_plan.py" not in planner.path.read_text():
        errors.append("ci-changes.yml must call the tested plan generator")
    builder = workflows.get("build-artifacts.yml")
    if not builder:
        errors.append("missing shared read-only image producer")
    elif any(
        word in builder.path.read_text()
        for word in ("docker/login-action", "push: true")
    ):
        errors.append("shared image producer must not authenticate or publish")
    community = workflows.get("community.yml")
    trigger = community.events.get("pull_request", {}) if community else {}
    if "synchronize" in (trigger or {}).get("types", []):
        errors.append("community metadata must not repeat on every synchronization")


def validate_release_contract(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    release = workflows.get("release.yml")
    if not release:
        errors.append("missing release orchestrator")
        return
    validate_release_publishers(release, errors)
    validate_release_images(release, errors)
    validate_stable_tag_owner(workflows, errors)
    validate_nightly_docker_owner(workflows, errors)
    validate_fixture_tag_policy(workflows, errors)


def validate_release_publishers(release: WorkflowLike, errors: list[str]) -> None:
    expected_publishers = {
        "docker": "docker-publish.yml",
        "helm": "helm-publish.yml",
        "pypi": "pypi-publish.yml",
        "crate": "publish-crate.yml",
    }
    for job_id, target in expected_publishers.items():
        job = release.jobs.get(job_id)
        if not isinstance(job, dict):
            errors.append(f".github/workflows/release.yml: missing '{job_id}' job")
            continue
        if local_target(job) != target or "validate" not in needs(job):
            errors.append(
                f".github/workflows/release.yml: '{job_id}' must call '{target}' "
                "after validate"
            )


def validate_release_images(release: WorkflowLike, errors: list[str]) -> None:
    docker_job = release.jobs.get("docker", {})
    images = (
        docker_job.get("with", {}).get("images")
        if isinstance(docker_job, dict)
        else None
    )
    if images != "${{ needs.ci.outputs.publish_images }}":
        errors.append("release images must consume the planner publication inventory")
    release_text = release.path.read_text(encoding="utf-8")
    fixture_bullets = {
        "- `provider-mocker`",
        "- `vllm-sr-sim`",
    }
    if any(bullet in release_text for bullet in fixture_bullets):
        errors.append(
            ".github/workflows/release.yml: release notes include a test fixture "
            "or developer companion image"
        )


def validate_stable_tag_owner(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    stable_tag_workflows = []
    for workflow in workflows.values():
        push = workflow.events.get("push", {})
        tags = push.get("tags", []) if isinstance(push, dict) else []
        if any(isinstance(tag, str) and tag.startswith("v[0-9]") for tag in tags):
            stable_tag_workflows.append(workflow.path.name)
    if stable_tag_workflows != ["release.yml"]:
        errors.append(
            "Stable release tags must trigger only release.yml; found: "
            + ", ".join(sorted(stable_tag_workflows))
        )


def validate_nightly_docker_owner(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    nightly_docker_calls = []
    for workflow in workflows.values():
        for job_id, job in workflow.jobs.items():
            if not isinstance(job, dict) or local_target(job) != "docker-publish.yml":
                continue
            if job.get("with", {}).get("mode") == "nightly":
                nightly_docker_calls.append(f"{workflow.path.name}:{job_id}")
    if nightly_docker_calls != ["nightly-build.yml:docker"]:
        errors.append(
            "Exactly one nightly Docker publisher is required; found: "
            + ", ".join(sorted(nightly_docker_calls))
        )
    nightly = workflows.get("nightly-build.yml")
    docker = nightly.jobs.get("docker", {}) if nightly else {}
    if docker.get("with", {}).get("images") != "${{ needs.ci.outputs.publish_images }}":
        errors.append("nightly images must consume the planner publication inventory")


def validate_fixture_tag_policy(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    if "provider-mocker" in set(NIGHTLY_IMAGES) | RELEASE_IMAGES:
        errors.append("provider-mocker cannot follow product publication schedules")
    for mode in ("pr", "nightly", "release"):
        try:
            publication_tags("provider-mocker", mode, "", False, "20260101")
        except ValueError:
            continue
        errors.append(f"provider-mocker cannot be published in {mode} mode")


def validate_security_boundary(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    ast_owners = {
        workflow.path.name
        for workflow in workflows.values()
        if "ast_security_scanner.py" in workflow.path.read_text(encoding="utf-8")
    }
    if ast_owners != {"security-scan.yml"}:
        errors.append(
            "AST scanning must have one workflow owner; found: "
            + ", ".join(sorted(ast_owners))
        )
    for workflow in workflows.values():
        if "pull_request_target" in workflow.events:
            errors.append(
                f"{workflow.relative_path}: pull_request_target is not allowed"
            )
        text = workflow.path.read_text(encoding="utf-8")
        if "SPAM_DETECTION_SCRIPT" in text or "eval(" in text:
            errors.append(
                f"{workflow.relative_path}: hidden moderation code execution remains"
            )


def validate_removed_workflows(errors: list[str]) -> None:
    removed = {
        "cleanup-existing-spam.yml",
        "bindings-test.yml",
        "check-linked-issue.yml",
        "docker-release.yml",
        "issue-manager.yml",
        "owner-notification.yml",
        "anti-spam-filter.yml",
        "performance-nightly.yml",
        "skill-review.yml",
        "paper-build.yml",
        "router-learning-eval.yml",
        "recipe-distribution.yml",
        "docker-validate.yml",
    }
    existing = sorted(name for name in removed if (WORKFLOW_DIR / name).exists())
    if existing:
        errors.append("Obsolete workflows still exist: " + ", ".join(existing))


def validate_mergify_contract(errors: list[str]) -> None:
    path = REPO_ROOT / ".mergify.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sections = ("queue_conditions", "merge_conditions")
    queue_rules = data.get("queue_rules", [])
    default_rule = queue_rules[0] if queue_rules else {}
    for section in sections:
        serialized = json.dumps(default_rule.get(section, []))
        if "check-success = PR Gate" not in serialized:
            errors.append(f".mergify.yml: {section} must require PR Gate")
        obsolete = (
            "test-and-build",
            "Lint",
            "Unit Tests",
            "Verify Manifests",
            "Validate OLM Bundle",
        )
        if any(f"check-success = {context}" in serialized for context in obsolete):
            errors.append(f".mergify.yml: {section} retains compatibility contexts")
    pull_request_rules = data.get("pull_request_rules", [])
    queue_rules = [
        rule
        for rule in pull_request_rules
        if isinstance(rule, dict)
        and isinstance(rule.get("actions"), dict)
        and "queue" in rule["actions"]
    ]
    if not any(
        REQUIRED_APPROVAL_CONDITION in rule.get("conditions", [])
        for rule in queue_rules
    ):
        errors.append(
            ".mergify.yml: pull request queue must require at least two "
            "approving reviews"
        )


def validate_workflow_policies(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    validate_pr_contract(workflows, errors)
    validate_release_contract(workflows, errors)
    validate_security_boundary(workflows, errors)
    validate_removed_workflows(errors)
    validate_mergify_contract(errors)
