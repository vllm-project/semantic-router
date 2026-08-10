#!/usr/bin/env python3
"""Validate local GitHub Actions YAML and reusable-workflow contracts."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from workflow_policy_validation import validate_workflow_policies

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
LOCAL_WORKFLOW_PREFIX = "./.github/workflows/"
NEEDS_OUTPUT_RE = re.compile(r"needs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)")
CALL_OUTPUT_RE = re.compile(r"jobs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class Workflow:
    path: Path
    data: dict[str, Any]

    @property
    def relative_path(self) -> str:
        return str(self.path.relative_to(REPO_ROOT))

    @property
    def events(self) -> dict[str, Any]:
        value = self.data.get("on", self.data.get(True, {}))
        return value if isinstance(value, dict) else {}

    @property
    def jobs(self) -> dict[str, Any]:
        value = self.data.get("jobs", {})
        return value if isinstance(value, dict) else {}

    @property
    def call_contract(self) -> dict[str, Any] | None:
        if "workflow_call" not in self.events:
            return None
        value = self.events.get("workflow_call")
        return value if isinstance(value, dict) else {}


def load_workflows(errors: list[str]) -> dict[str, Workflow]:
    workflows: dict[str, Workflow] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            errors.append(f"{path.relative_to(REPO_ROOT)}: invalid YAML: {error}")
            continue
        if not isinstance(loaded, dict):
            errors.append(f"{path.relative_to(REPO_ROOT)}: workflow root is not a map")
            continue
        workflows[path.name] = Workflow(path=path, data=loaded)
    return workflows


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


def available_outputs(job: dict[str, Any], workflows: dict[str, Workflow]) -> set[str]:
    target_name = local_target(job)
    if target_name:
        target = workflows.get(Path(target_name).name)
        contract = target.call_contract if target else None
        outputs = contract.get("outputs", {}) if contract is not None else {}
        return set(outputs) if isinstance(outputs, dict) else set()
    outputs = job.get("outputs", {})
    return set(outputs) if isinstance(outputs, dict) else set()


def validate_permissions(workflow: Workflow, errors: list[str]) -> None:
    if "permissions" not in workflow.data:
        errors.append(f"{workflow.relative_path}: missing default permissions")
        return
    permissions = workflow.data["permissions"]
    if not isinstance(permissions, dict):
        return
    writes = [key for key, value in permissions.items() if value == "write"]
    if writes:
        errors.append(
            f"{workflow.relative_path}: workflow-level write permissions: "
            + ", ".join(sorted(writes))
        )


def effective_permissions(
    workflow: Workflow, job: dict[str, Any] | None = None
) -> dict[str, str]:
    permissions = (
        job.get("permissions")
        if job is not None and "permissions" in job
        else workflow.data.get("permissions")
    )
    return permissions if isinstance(permissions, dict) else {}


def requested_write_permissions(workflow: Workflow) -> set[str]:
    writes: set[str] = set()
    for raw_job in workflow.jobs.values():
        if not isinstance(raw_job, dict):
            continue
        writes.update(
            permission
            for permission, level in effective_permissions(workflow, raw_job).items()
            if level == "write"
        )
    return writes


def validate_local_calls(
    workflow: Workflow, workflows: dict[str, Workflow], errors: list[str]
) -> None:
    for job_id, raw_job in workflow.jobs.items():
        if not isinstance(raw_job, dict):
            errors.append(f"{workflow.relative_path}: job '{job_id}' is not a map")
            continue
        declared_needs = needs(raw_job)
        validate_needs(workflow, job_id, declared_needs, errors)
        validate_local_call(workflow, job_id, raw_job, workflows, errors)
        validate_needs_outputs(
            workflow, job_id, raw_job, declared_needs, workflows, errors
        )


def validate_needs(
    workflow: Workflow, job_id: str, declared_needs: set[str], errors: list[str]
) -> None:
    unknown_needs = declared_needs - set(workflow.jobs)
    if unknown_needs:
        errors.append(
            f"{workflow.relative_path}: job '{job_id}' has unknown needs: "
            + ", ".join(sorted(unknown_needs))
        )


def validate_local_call(
    workflow: Workflow,
    job_id: str,
    job: dict[str, Any],
    workflows: dict[str, Workflow],
    errors: list[str],
) -> None:
    target_name = local_target(job)
    if not target_name:
        return
    target = workflows.get(Path(target_name).name)
    if target is None:
        errors.append(
            f"{workflow.relative_path}: job '{job_id}' references "
            f"missing workflow '{target_name}'"
        )
        return
    if target.call_contract is None:
        errors.append(
            f"{workflow.relative_path}: job '{job_id}' references "
            f"non-reusable workflow '{target_name}'"
        )
        return
    caller_permissions = effective_permissions(workflow, job)
    unavailable_writes = {
        permission
        for permission in requested_write_permissions(target)
        if caller_permissions.get(permission) != "write"
    }
    if unavailable_writes:
        errors.append(
            f"{workflow.relative_path}: job '{job_id}' cannot call '{target_name}'; "
            "callee requests write permissions above the caller ceiling: "
            + ", ".join(sorted(unavailable_writes))
        )
    validate_call_inputs(workflow, job_id, job, target_name, target, errors)


def validate_call_inputs(
    workflow: Workflow,
    job_id: str,
    job: dict[str, Any],
    target_name: str,
    target: Workflow,
    errors: list[str],
) -> None:
    contract = target.call_contract or {}
    inputs = contract.get("inputs", {})
    inputs = inputs if isinstance(inputs, dict) else {}
    supplied = job.get("with", {})
    supplied = supplied if isinstance(supplied, dict) else {}
    unknown_inputs = set(supplied) - set(inputs)
    missing_inputs = {
        name
        for name, spec in inputs.items()
        if isinstance(spec, dict)
        and spec.get("required") is True
        and name not in supplied
    }
    if unknown_inputs:
        errors.append(
            f"{workflow.relative_path}: job '{job_id}' passes unknown inputs to "
            f"'{target_name}': " + ", ".join(sorted(unknown_inputs))
        )
    if missing_inputs:
        errors.append(
            f"{workflow.relative_path}: job '{job_id}' omits required inputs for "
            f"'{target_name}': " + ", ".join(sorted(missing_inputs))
        )


def validate_needs_outputs(
    workflow: Workflow,
    job_id: str,
    job: dict[str, Any],
    declared_needs: set[str],
    workflows: dict[str, Workflow],
    errors: list[str],
) -> None:
    for needed_job, output in NEEDS_OUTPUT_RE.findall(json.dumps(job)):
        if needed_job not in declared_needs:
            errors.append(
                f"{workflow.relative_path}: job '{job_id}' reads "
                f"needs.{needed_job}.outputs.{output} without declaring the need"
            )
            continue
        source = workflow.jobs.get(needed_job)
        if isinstance(source, dict) and output not in available_outputs(
            source, workflows
        ):
            errors.append(
                f"{workflow.relative_path}: job '{job_id}' reads unknown "
                f"output '{output}' from '{needed_job}'"
            )


def validate_call_outputs(workflow: Workflow, errors: list[str]) -> None:
    contract = workflow.call_contract
    if contract is None:
        return
    outputs = contract.get("outputs", {})
    if not isinstance(outputs, dict):
        errors.append(f"{workflow.relative_path}: workflow_call.outputs is not a map")
        return
    for output_name, specification in outputs.items():
        validate_call_output(workflow, output_name, specification, errors)


def validate_call_output(
    workflow: Workflow,
    output_name: str,
    specification: Any,
    errors: list[str],
) -> None:
    references = CALL_OUTPUT_RE.findall(json.dumps(specification))
    if not references:
        errors.append(
            f"{workflow.relative_path}: workflow output '{output_name}' "
            "does not reference a job output"
        )
        return
    for job_id, job_output in references:
        job = workflow.jobs.get(job_id)
        if not isinstance(job, dict):
            errors.append(
                f"{workflow.relative_path}: workflow output '{output_name}' "
                f"references missing job '{job_id}'"
            )
        elif job_output not in available_outputs(job, {}):
            errors.append(
                f"{workflow.relative_path}: workflow output '{output_name}' "
                f"references missing output '{job_output}' on '{job_id}'"
            )


def main() -> int:
    errors: list[str] = []
    workflows = load_workflows(errors)
    for workflow in workflows.values():
        validate_permissions(workflow, errors)
        validate_local_calls(workflow, workflows, errors)
        validate_call_outputs(workflow, errors)
    validate_workflow_policies(workflows, errors)

    if errors:
        print("Workflow contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Workflow contract validation passed ({len(workflows)} workflows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
