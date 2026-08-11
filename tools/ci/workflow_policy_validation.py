"""Repository-specific GitHub Actions lifecycle and safety policy checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import yaml
from classify_pr_changes import NIGHTLY_IMAGES, PRODUCTION_RELEASE_IMAGES
from test_domain_registry import (
    domain_records,
    load_test_domain_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
LOCAL_WORKFLOW_PREFIX = "./.github/workflows/"
REQUIRED_COMPATIBILITY_CHECKS = {
    "Run pre-commit hooks check file lint",
    "test-and-build",
    "Lint",
    "Unit Tests",
    "Verify Manifests",
    "Validate OLM Bundle",
}
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


def validate_pr_images_contract(
    dispatcher: WorkflowLike,
    workflows: dict[str, WorkflowLike],
    errors: list[str],
) -> None:
    images = dispatcher.jobs.get("images", {})
    if not isinstance(images, dict) or local_target(images) != "docker-validate.yml":
        errors.append(
            ".github/workflows/pr.yml: images must call the read-only "
            "docker-validate.yml workflow"
        )
    validation = workflows.get("docker-validate.yml")
    if validation is None:
        errors.append(
            ".github/workflows/docker-validate.yml: missing PR image validator"
        )
        return
    permissions = validation.data.get("permissions")
    if permissions != {"contents": "read"}:
        errors.append(
            ".github/workflows/docker-validate.yml: permissions must be "
            "exactly contents: read"
        )
    validation_text = validation.path.read_text(encoding="utf-8")
    if "docker/login-action" in validation_text or "push: true" in validation_text:
        errors.append(
            ".github/workflows/docker-validate.yml: PR image validation "
            "must not authenticate or publish"
        )


def validate_pr_selection_contract(
    dispatcher: WorkflowLike,
    workflows: dict[str, WorkflowLike],
    errors: list[str],
) -> None:
    validate_pr_images_contract(dispatcher, workflows, errors)
    if "bindings" in dispatcher.jobs:
        errors.append(
            ".github/workflows/pr.yml: duplicate standalone bindings job remains"
        )
    performance = dispatcher.jobs.get("performance", {})
    if isinstance(performance, dict) and "outputs.full" in str(
        performance.get("if", "")
    ):
        errors.append(
            ".github/workflows/pr.yml: ci/full must not enable performance tests"
        )
    validate_classifier_contract(workflows, errors)


def validate_pr_contract(workflows: dict[str, WorkflowLike], errors: list[str]) -> None:
    dispatcher = workflows.get("pr.yml")
    if dispatcher is None:
        errors.append(".github/workflows/pr.yml: missing PR dispatcher")
        return
    if "pull_request" not in dispatcher.events:
        errors.append(".github/workflows/pr.yml: missing pull_request trigger")

    compatibility = dispatcher.jobs.get("compatibility", {})
    matrix = (
        compatibility.get("strategy", {}).get("matrix", {}).get("check", [])
        if isinstance(compatibility, dict)
        else []
    )
    if set(matrix) != REQUIRED_COMPATIBILITY_CHECKS:
        errors.append(
            ".github/workflows/pr.yml: compatibility matrix does not preserve "
            "the required branch-protection and Mergify contexts"
        )
    registry_contexts = set(
        load_test_domain_registry().get("compatibility_contexts", [])
    )
    if registry_contexts != REQUIRED_COMPATIBILITY_CHECKS:
        errors.append(
            "tools/agent/test-domain-registry.yaml compatibility contexts "
            "do not match workflow policy"
        )
    gate = dispatcher.jobs.get("pr-gate", {})
    if not isinstance(gate, dict) or gate.get("name") != "PR Gate":
        errors.append(".github/workflows/pr.yml: missing stable 'PR Gate' aggregate")

    validate_pr_selection_contract(dispatcher, workflows, errors)
    validate_registry_dispatch_contract(dispatcher, workflows, errors)

    classifier_callers = []
    for workflow in workflows.values():
        if "pull_request" not in workflow.events:
            continue
        for job in workflow.jobs.values():
            if isinstance(job, dict) and local_target(job) == "ci-changes.yml":
                classifier_callers.append(workflow.path.name)
    if classifier_callers != ["pr.yml"]:
        errors.append(
            "PR change classification must run only in pr.yml; callers: "
            + ", ".join(sorted(classifier_callers))
        )

    community = workflows.get("community.yml")
    pull_request = community.events.get("pull_request", {}) if community else {}
    types = pull_request.get("types", []) if isinstance(pull_request, dict) else []
    if "synchronize" in types:
        errors.append(
            ".github/workflows/community.yml: metadata checks must not run on synchronize"
        )


def validate_registry_dispatch_contract(
    dispatcher: WorkflowLike,
    workflows: dict[str, WorkflowLike],
    errors: list[str],
) -> None:
    for domain_name, domain in domain_records().items():
        if "pr" not in domain.get("cadence", []):
            continue
        job_id = domain["pr_job"]
        workflow_name = Path(domain["workflow"]).name
        job = dispatcher.jobs.get(job_id)
        if not isinstance(job, dict):
            errors.append(
                f".github/workflows/pr.yml: missing registry domain job {job_id!r}"
            )
            continue
        if local_target(job) != workflow_name:
            errors.append(
                f".github/workflows/pr.yml: registry domain {domain_name!r} "
                f"must call {workflow_name!r}"
            )
        workflow = workflows.get(workflow_name)
        if workflow is None or "workflow_call" not in workflow.events:
            errors.append(
                f"{domain['workflow']}: registry domain workflow must be reusable"
            )


def validate_release_contract(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    release = workflows.get("release.yml")
    if release is None:
        errors.append(".github/workflows/release.yml: missing release orchestrator")
        return

    validate_release_publishers(release, errors)
    validate_release_images(release, errors)
    validate_stable_tag_owner(workflows, errors)
    validate_nightly_docker_owner(workflows, errors)
    validate_fixture_tag_policy(workflows, errors)


def validate_classifier_contract(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    classifier = workflows.get("ci-changes.yml")
    if classifier is None:
        errors.append(".github/workflows/ci-changes.yml: missing classifier")
        return
    text = classifier.path.read_text(encoding="utf-8")
    if "tools/ci/classify_pr_changes.py" not in text:
        errors.append(
            ".github/workflows/ci-changes.yml: must use the tested classifier contract"
        )
    if "dorny/paths-filter" in text:
        errors.append(
            ".github/workflows/ci-changes.yml: hidden dorny product filters remain"
        )


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
    raw_images = (
        docker_job.get("with", {}).get("images")
        if isinstance(docker_job, dict)
        else None
    )
    try:
        images = set(json.loads(raw_images))
    except (TypeError, json.JSONDecodeError):
        images = set()
    if images != RELEASE_IMAGES:
        errors.append(
            ".github/workflows/release.yml: release image inventory differs "
            "from production deliverables"
        )
    release_text = release.path.read_text(encoding="utf-8")
    fixture_bullets = {
        "- `anthropic-shim`",
        "- `llm-katan`",
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
    docker_job = nightly.jobs.get("docker", {}) if nightly else {}
    raw_images = (
        docker_job.get("with", {}).get("images")
        if isinstance(docker_job, dict)
        else None
    )
    try:
        images = set(json.loads(raw_images))
    except (TypeError, json.JSONDecodeError):
        images = set()
    if images != set(NIGHTLY_IMAGES):
        errors.append(
            ".github/workflows/nightly-build.yml: nightly image inventory must "
            "include production, companion, and test images"
        )


def validate_fixture_tag_policy(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    publisher = workflows.get("docker-publish.yml")
    text = publisher.path.read_text(encoding="utf-8") if publisher else ""
    if "anthropic-shim|llm-katan" not in text or "${IMAGE}:nightly" not in text:
        errors.append(
            ".github/workflows/docker-publish.yml: nightly test fixtures need "
            "the mutable nightly tag"
        )


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
        "skill-review.yml",
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
    required = REQUIRED_COMPATIBILITY_CHECKS - {"Run pre-commit hooks check file lint"}
    required.add("PR Gate")
    for section in sections:
        serialized = json.dumps(default_rule.get(section, []))
        missing = sorted(
            context
            for context in required
            if f"check-success = {context}" not in serialized
        )
        if missing:
            errors.append(
                f".mergify.yml: {section} is missing contexts: " + ", ".join(missing)
            )


def validate_workflow_policies(
    workflows: dict[str, WorkflowLike], errors: list[str]
) -> None:
    validate_pr_contract(workflows, errors)
    validate_release_contract(workflows, errors)
    validate_security_boundary(workflows, errors)
    validate_removed_workflows(errors)
    validate_mergify_contract(errors)
