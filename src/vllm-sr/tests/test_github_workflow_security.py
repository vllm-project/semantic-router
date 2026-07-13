"""Static regressions for privileged GitHub workflow trust boundaries."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def load_workflow(name: str) -> tuple[dict, str]:
    path = WORKFLOWS_DIR / name
    text = path.read_text(encoding="utf-8")
    return yaml.load(text, Loader=yaml.BaseLoader), text


def workflow_steps(workflow: dict, job_name: str) -> dict[str, dict]:
    steps = workflow["jobs"][job_name]["steps"]
    return {step["name"]: step for step in steps}


def test_security_scan_pr_code_runs_only_with_read_permissions() -> None:
    workflow, text = load_workflow("security-scan.yml")
    triggers = workflow["on"]
    job = workflow["jobs"]["ast-security-scan"]
    steps = workflow_steps(workflow, "ast-security-scan")
    checkout = steps["Check out the repo"]

    assert "pull_request" in triggers
    assert "push" in triggers
    assert "workflow_dispatch" in triggers
    assert "pull_request_target" not in triggers
    assert job["permissions"] == {"contents": "read"}
    assert checkout["with"]["persist-credentials"] == "false"
    assert "ref" not in checkout["with"]
    assert steps["Run AST PR diff scan"]["if"] == (
        "github.event_name == 'pull_request'"
    )
    assert "GITHUB_STEP_SUMMARY" in steps["Write security scan summary"]["run"]
    assert "set -o pipefail" in steps["Run regex fallback scan"]["run"]
    assert "github.event.pull_request.head.sha" not in text
    assert "github.rest.issues" not in text
    assert "pull-requests: write" not in text
    assert "issues: write" not in text


def test_owner_notification_never_interpolates_filenames_into_javascript() -> None:
    workflow, _ = load_workflow("owner-notification.yml")
    steps = workflow_steps(workflow, "notify-owners")
    checkout = steps["Checkout code"]
    collect_script = steps["Get changed files"]["with"]["script"]
    notify_step = steps["Find owners and notify"]
    notify_script = notify_step["with"]["script"]
    output_expression = "${{ steps.changed-files.outputs.filenames_json }}"

    assert checkout["with"]["persist-credentials"] == "false"
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert "JSON.stringify(filenames)" in collect_script
    assert "filenames_json" in collect_script
    assert notify_step["env"]["CHANGED_FILES_JSON"] == output_expression
    assert "JSON.parse(" in notify_script
    assert "process.env.CHANGED_FILES_JSON" in notify_script
    assert output_expression not in notify_script
    assert "${{" not in notify_script
    assert "all_changed_files" not in notify_script
    assert ".split(' ')" not in notify_script


def test_owner_notification_contains_paths_and_escapes_markdown() -> None:
    workflow, _ = load_workflow("owner-notification.yml")
    notify_script = workflow_steps(workflow, "notify-owners")["Find owners and notify"][
        "with"
    ]["script"]

    assert "normalizeRepositoryPath" in notify_script
    assert "isInsideRepository" in notify_script
    assert "part === '..'" in notify_script
    assert "ownerStat.isSymbolicLink()" in notify_script
    assert "escapeHtml" in notify_script
    assert "inlineCode(file)" in notify_script
    assert r"[\p{Cc}\p{Cf}]" in notify_script


def test_skill_review_third_party_code_has_no_write_capability() -> None:
    workflow, text = load_workflow("skill-review.yml")
    job = workflow["jobs"]["review"]
    steps = job["steps"]

    assert job["permissions"] == {"contents": "read"}
    assert steps[0]["with"]["persist-credentials"] == "false"
    assert steps[1]["uses"] == (
        "tesslio/skill-review@c9357d3848c862bfe750e89c1c42599f3f45d483"
    )
    assert steps[1]["with"]["comment"] == "false"
    assert "tesslio/skill-review@main" not in text
    assert "pull-requests: write" not in text


def test_docker_pr_build_never_receives_registry_write_credentials() -> None:
    workflow, _ = load_workflow("docker-publish.yml")
    job = workflow["jobs"]["build_pr"]
    steps = workflow_steps(workflow, "build_pr")
    checkout = steps["Check out the repo"]

    assert job["permissions"] == {"contents": "read"}
    assert checkout["with"]["persist-credentials"] == "false"
    assert all(step.get("uses") != "docker/login-action@v3" for step in job["steps"])
    assert "${{ secrets.GITHUB_TOKEN }}" not in yaml.safe_dump(job)
    assert steps["Build ${{ matrix.image }} (amd64 only)"]["with"]["push"] == "false"

    publish_job = workflow["jobs"]["build_platform"]
    publish_steps = workflow_steps(workflow, "build_platform")
    assert publish_job["permissions"]["packages"] == "write"
    assert (
        publish_steps["Log in to GitHub Container Registry"]["uses"]
        == "docker/login-action@v3"
    )
