"""The CLI and Go loader consume the same rule-tree budget corpus."""

import json
from pathlib import Path

import pytest
from cli.config_schema.validation import validate_config_structure
from cli.decision_rule_limits import validate_decision_rule_limits
from cli.models import UserConfig

CORPUS = json.loads(
    (
        Path(__file__).parents[2]
        / "semantic-router/pkg/config/testdata/decision_rule_limits.json"
    ).read_text()
)


def document_for(case):
    rule = {"type": "keyword", "name": "urgent"}
    if case["kind"] == "depth":
        for _ in range(case["size"] - 1):
            rule = {"operator": "AND", "conditions": [rule]}
    elif case["kind"] == "nodes":
        rule = {"operator": "AND", "conditions": [rule] * (case["size"] - 1)}
    else:
        rule = {}
    decision = {"name": "bounded", "rules": rule, "priority": 1, "modelRefs": []}
    if case["kind"] == "missing":
        del decision["rules"]
    return {
        "version": "v0.3",
        "global": {"router": {"decision_rule_limits": case.get("limits", {})}},
        "routing": {"decisions": [decision]},
    }


@pytest.mark.parametrize("case", CORPUS, ids=lambda case: case["name"])
def test_shared_rule_budget_corpus(case):
    document = document_for(case)
    expected = case.get("error")
    if expected:
        with pytest.raises(ValueError) as error:
            validate_decision_rule_limits(document)
        assert expected in str(error.value)
        assert expected in validate_config_structure(document)[0]
        with pytest.raises(ValueError) as error:
            UserConfig.model_validate(document)
        assert expected in str(error.value)
    else:
        validate_decision_rule_limits(document)
        parsed = UserConfig.model_validate(document)
        output = parsed.model_dump(by_alias=True, exclude_none=True)
        validate_decision_rule_limits(output)
        UserConfig.model_validate(output)
        assert not validate_config_structure(document)


def test_bare_leaf_round_trip_does_not_add_a_node():
    document = document_for(
        {"kind": "depth", "size": 1, "limits": {"max_nodes": 1, "max_depth": 1}}
    )
    parsed = UserConfig.model_validate(document)
    output = parsed.model_dump(by_alias=True, exclude_none=True)
    assert output["routing"]["decisions"][0]["rules"] == {
        "type": "keyword",
        "name": "urgent",
    }
    validate_decision_rule_limits(parsed)
    UserConfig.model_validate(output)


def test_budget_resets_for_each_decision_and_recipe():
    document = document_for({"kind": "depth", "size": 1, "limits": {"max_nodes": 1}})
    document["routing"]["decisions"] *= 2
    document["recipes"] = [{"name": "support", "routing": document["routing"]}]
    validate_decision_rule_limits(document)
    document["recipes"] = [
        {
            "name": "support",
            "routing": document_for({"kind": "nodes", "size": 2})["routing"],
        }
    ]
    with pytest.raises(ValueError, match=r'routing recipe "support".*node count 2'):
        validate_decision_rule_limits(document)


def test_deep_tree_fails_before_recursive_copy_or_model_construction():
    document = document_for({"kind": "depth", "size": 2000})
    assert "depth 17 exceeds max_depth=16" in validate_config_structure(document)[0]
    with pytest.raises(ValueError, match="depth 17 exceeds max_depth=16"):
        UserConfig.model_validate(document)
