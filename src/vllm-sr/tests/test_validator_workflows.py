"""Tests for workflows-specific CLI validation."""

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.algorithms import WorkflowRoleConfig, WorkflowsAlgorithmConfig  # noqa: E402
from cli.validator_workflows import validate_static_workflow_roles  # noqa: E402


def _decision_with_models(*models: str):
    return SimpleNamespace(
        name="flow-test",
        modelRefs=[SimpleNamespace(model=model) for model in models],
    )


def test_static_workflow_access_list_allows_previous_agent_id():
    workflows = WorkflowsAlgorithmConfig(
        mode="static",
        roles=[
            WorkflowRoleConfig(name="solver", models=["worker-a", "worker-b"]),
            WorkflowRoleConfig(
                name="reviewer",
                models=["worker-c"],
                access_list=["solver:1:worker-b"],
            ),
        ],
    )

    errors = validate_static_workflow_roles(
        _decision_with_models("worker-a", "worker-b", "worker-c"),
        workflows,
    )

    assert errors == []


def test_static_workflow_access_list_rejects_future_agent_id():
    workflows = WorkflowsAlgorithmConfig(
        mode="static",
        roles=[
            WorkflowRoleConfig(
                name="solver",
                models=["worker-a"],
                access_list=["reviewer:0:worker-b"],
            ),
            WorkflowRoleConfig(name="reviewer", models=["worker-b"]),
        ],
    )

    errors = validate_static_workflow_roles(
        _decision_with_models("worker-a", "worker-b"),
        workflows,
    )

    assert len(errors) == 1
    assert "unknown or future role/agent" in errors[0].message
