"""CLI contract tests for the agentic_facts signal family."""

from pathlib import Path

import pytest
import yaml
from cli.config_schema.validation import validate_config_structure
from cli.config_yaml import safe_load_router_config
from cli.main import main
from cli.models import AgenticFactsRule, Signals
from click.testing import CliRunner
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_equals_and_in_predicates_validate():
    role = AgenticFactsRule(
        name="reviewer_role", field="delegated_role", predicate={"equals": "reviewer"}
    )
    phase = AgenticFactsRule(
        name="execution_phase",
        field="task_phase",
        predicate={"in": ["execute", "verify"]},
    )
    assert role.predicate.equals == "reviewer"
    assert phase.predicate.in_ == ["execute", "verify"]


@pytest.mark.parametrize(
    "fields",
    [
        {"name": " padded ", "field": "task_phase", "predicate": {"equals": "plan"}},
        {"name": "r", "field": "budget", "predicate": {"equals": "low"}},
        {"name": "r", "field": "task_phase", "predicate": {}},
        {"name": "r", "field": "task_phase", "predicate": {"in": []}},
        {
            "name": "r",
            "field": "task_phase",
            "predicate": {"equals": "plan", "in": ["plan"]},
        },
        {"name": "r", "field": "task_phase", "predicate": {"exists": True}},
        {"name": "r", "field": "task_phase", "predicate": {"equals": "plan"}, "x": 1},
    ],
)
def test_invalid_rules_are_rejected(fields):
    with pytest.raises(ValidationError):
        AgenticFactsRule(**fields)


def test_rule_names_are_unique_ignoring_case():
    rule = {"field": "task_phase", "predicate": {"equals": "plan"}}
    with pytest.raises(ValidationError):
        Signals(agentic_facts=[{"name": "Plan", **rule}, {"name": "plan", **rule}])


def test_maintained_fragment_matches_generated_structure():
    fragment = (
        REPO_ROOT / "config/fragments/signal/agentic-facts/delegated-role.yaml"
    ).read_text(encoding="utf-8")
    config = {"version": "v0.3", **safe_load_router_config(fragment)}

    assert validate_config_structure(config) == []


def _write_agentic_facts_config(tmp_path: Path, field: str) -> Path:
    path = tmp_path / "config.yaml"
    created = CliRunner().invoke(main, ["config", "init", "--output", str(path)])
    assert created.exit_code == 0, created.output

    config = safe_load_router_config(path.read_text(encoding="utf-8"))
    config["routing"]["signals"] = {
        "agentic_facts": [
            {
                "name": "reviewer_role",
                "field": field,
                "predicate": {"equals": "reviewer"},
            }
        ]
    }
    review = dict(config["routing"]["decisions"][0])
    review.update(
        name="review-route",
        priority=200,
        rules={
            "operator": "AND",
            "conditions": [{"type": "agentic_facts", "name": "reviewer_role"}],
        },
    )
    config["routing"]["decisions"].insert(0, review)
    config["global"] = {
        "router": {
            "agentic_facts": {
                "enabled": True,
                "carrier_header": "x-vsr-agentic-facts",
                "trust": {
                    "marker_header": "x-vsr-agentic-facts-trusted",
                    "marker_value": "1",
                },
            }
        }
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_config_validate_accepts_agentic_facts_config(tmp_path: Path):
    path = _write_agentic_facts_config(tmp_path, field="delegated_role")

    result = CliRunner().invoke(
        main, ["config", "validate", "--offline", "--config", str(path)]
    )

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output


def test_config_validate_rejects_unknown_agentic_facts_field(tmp_path: Path):
    path = _write_agentic_facts_config(tmp_path, field="trust_boundary")

    result = CliRunner().invoke(
        main, ["config", "validate", "--offline", "--config", str(path)]
    )

    assert result.exit_code != 0
    assert "agentic_facts" in result.output
