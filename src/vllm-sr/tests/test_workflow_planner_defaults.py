"""The dynamic planner default does not weaken worker admission contracts."""

import pytest
from cli.algorithms import WorkflowPlannerConfig
from cli.models import UserConfig
from cli.validator import validate_algorithm_configurations
from pydantic import ValidationError


@pytest.mark.parametrize("planner", [None, {}, {"model": "coordinator"}])
def test_dynamic_planner_default_preserves_distinct_worker_minimum(planner):
    workflows = {"mode": "dynamic", "min_successful_responses": 2, "max_parallel": 2}
    if planner is not None:
        workflows["planner"] = planner
    cfg = UserConfig.model_validate(
        {
            "version": "v0.3",
            "routing": {
                "decisions": [
                    {
                        "name": "flow",
                        "priority": 1,
                        "modelRefs": [{"model": "worker-a"}, {"model": "worker-b"}],
                        "algorithm": {
                            "type": "workflows",
                            "minimum_candidates": 2,
                            "workflows": workflows,
                        },
                    }
                ]
            },
        }
    )
    assert validate_algorithm_configurations(cfg) == []
    decision = cfg.routing.decisions[0]
    actual = decision.algorithm.workflows.planner
    assert (actual.model if actual else None) == (planner or {}).get("model")
    decision.modelRefs[1].model = "worker-a"
    errors = validate_algorithm_configurations(cfg)
    assert any("worker pool size 1" in error.message for error in errors)


def test_dynamic_planner_default_keeps_positive_output_validation():
    with pytest.raises(ValidationError):
        WorkflowPlannerConfig(max_completion_tokens=-1)
