"""The CLI refuses the removed application signal forms with the Router's own
lines, and the Router's migration leaves none of them behind."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_application_forms import removed_application_forms  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402

GOLDEN = REPO_ROOT / "src/semantic-router/pkg/config/testdata/migrate"
CASES = (
    "removed-application-fields",
    "removed-domain-autogen",
    "removed-embedding-image-anchors",
    "removed-guard-modules",
    "removed-kb-signal",
    "removed-learned-signal-types",
    "removed-model-modules",
    "removed-safety-hazard",
)
# The cases without a mechanical rewrite have no after.yaml.
MIGRATED = tuple(case for case in CASES if (GOLDEN / case / "after.yaml").exists())


def _golden(case: str, name: str) -> str:
    return (GOLDEN / case / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("case", CASES)
def test_the_cli_refuses_a_removed_application_form_with_the_routers_line(case):
    before = yaml.safe_load(_golden(case, "before.yaml"))
    assert _golden(case, "hint.txt").strip() in removed_application_forms(before)


@pytest.mark.parametrize("case", MIGRATED)
def test_the_routers_migration_leaves_no_removed_application_form(case):
    after = yaml.safe_load(_golden(case, "after.yaml"))
    assert removed_application_forms(after) == []


@pytest.mark.parametrize(
    "collection",
    ["reasks", "topic_continuity", "classifiers", "preferences", "embeddings", "kb"],
)
def test_a_removed_signal_type_is_refused_in_a_recipe(collection):
    rule = {"name": "rule", "candidates": ["a question about tax law"]}
    config = {"recipes": [{"name": "r", "signals": {collection: [rule]}}]}
    assert removed_application_forms(config)[0].startswith(
        f"recipes[r].signals.{collection}: removed_signal_type: "
    )


def test_a_removed_condition_type_is_refused_where_it_is_nested():
    leaf = {"type": "classifier", "name": "risk", "label": "RISKY"}
    rules = {
        "operator": "AND",
        "conditions": [{"operator": "OR", "conditions": [leaf]}],
    }
    config = {
        "recipes": [{"name": "default", "decisions": [{"name": "d", "rules": rules}]}]
    }
    assert removed_application_forms(config) == [
        "recipes[default].decisions[d].rules.conditions[0].conditions[risk]: "
        "removed_signal_type: condition type classifier is systemone; run "
        "`vllm-sr config migrate` (type: systemone)"
    ]


def test_the_maintained_reference_config_has_no_removed_application_form():
    config = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text("utf-8"))
    assert removed_application_forms(config) == []


def test_a_decision_v1_binding_of_a_removed_module_is_named_for_its_signal_type():
    config = {
        "global": {
            "model_catalog": {
                "bindings": {
                    "prompt_guard": {"deployment": "guard", "contract": "decision.v1"},
                    "classifier.domain": {"deployment": "domains"},
                }
            }
        }
    }
    assert removed_application_forms(config) == [
        "global.model_catalog.bindings.prompt_guard: removed_field: binding "
        "prompt_guard is named for the signal type it serves; run `vllm-sr config "
        "migrate` (jailbreak)",
        "global.model_catalog.bindings.classifier.domain: removed_field: binding "
        "classifier.domain served a removed signal or a removed classifier model; "
        "run `vllm-sr config migrate` (drop it; the decision model answers the "
        "signal)",
    ]


def test_parsing_refuses_a_removed_application_form(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(_golden("removed-kb-signal", "before.yaml"), encoding="utf-8")
    with pytest.raises(
        ConfigParseError,
        match=re.escape(
            "recipes[default].signals.kb: removed_signal_type: the kb signal is removed"
        ),
    ):
        parse_user_config(str(path), log_summary=False)
