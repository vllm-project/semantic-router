"""The heuristic signals as the CLI reads them: keywords rules that count, and
the removed events, structure and actions signals and conversation
image_content feature, refused with the Router's own lines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_heuristic_forms import removed_heuristic_forms  # noqa: E402
from cli.models import KeywordSignal  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config_data  # noqa: E402

GOLDEN = REPO_ROOT / "src/semantic-router/pkg/config/testdata/migrate"
MAINTAINED_GOLDEN = (
    REPO_ROOT
    / "src/semantic-router/pkg/config/testdata/model_resolution/maintained.golden.json"
)
RELEASE_SNAPSHOT = "config/recipes/built-in/v0.4/mom-v1/config.yaml"
HEURISTIC_WHATS = (
    "the events signal is removed",
    "condition type event is removed",
    "a structure rule on text_bytes is removed",
    "the structure signal is removed",
    "condition type structure is removed",
    "the actions signal is removed",
    "condition type action is removed",
    "projection input type action is removed",
    "the conversation feature image_content is removed",
)
CASES = (
    "removed-events-signal",
    "removed-structure-signal",
    "removed-structure-text-bytes",
    "removed-actions-signal",
    "removed-conversation-image-content",
)


def _golden(case: str, name: str) -> dict:
    return yaml.safe_load((GOLDEN / case / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES)
def test_parser_refuses_a_removed_heuristic_with_the_router_line(case: str) -> None:
    hint = (GOLDEN / case / "hint.txt").read_text(encoding="utf-8").strip()

    with pytest.raises(ConfigParseError) as refused:
        parse_user_config_data(
            _golden(case, "before.yaml"), "before.yaml", log_summary=False
        )

    assert hint in str(refused.value).splitlines()


@pytest.mark.parametrize(
    "case", [case for case in CASES if (GOLDEN / case / "after.yaml").exists()]
)
def test_migrated_heuristics_have_no_removed_forms(case: str) -> None:
    assert removed_heuristic_forms(_golden(case, "after.yaml")) == []


def test_removed_forms_and_their_references_are_named_in_every_routing_scope() -> None:
    lines = removed_heuristic_forms(_golden("removed-events-signal", "before.yaml"))

    assert [line.split(": ", 1)[0] for line in lines] == [
        "routing.signals.events",
        "recipes[ops].routing.signals.events",
        "routing.decisions[incident].rules.conditions[critical_payment_event]",
        "routing.decisions[incident].rules.conditions[audit_event]",
        "recipes[ops].routing.decisions[outage].rules",
        "routing.projections.scores[incident_pressure].inputs[critical_payment_event]",
    ]


def test_cli_lines_are_the_routers_on_a_release_snapshot() -> None:
    router = json.loads(MAINTAINED_GOLDEN.read_text(encoding="utf-8"))
    router_lines = router[RELEASE_SNAPSHOT]["error"].split("\n")
    snapshot = yaml.safe_load(
        (REPO_ROOT / RELEASE_SNAPSHOT).read_text(encoding="utf-8")
    )

    lines = removed_heuristic_forms(snapshot)

    assert lines
    assert lines == [
        line for line in router_lines if any(what in line for what in HEURISTIC_WHATS)
    ]


def test_keyword_rule_counts_with_one_predicate_on_an_or_rule() -> None:
    rule = KeywordSignal(
        name="many_questions", operator="OR", keywords=["?"], count={"gte": 3}
    )
    assert rule.count is not None and rule.count.gte == 3

    with pytest.raises(ValidationError, match="set count or density, not both"):
        KeywordSignal(
            name="both",
            operator="OR",
            keywords=["?"],
            count={"gte": 3},
            density={"gt": 1},
        )
    with pytest.raises(ValidationError, match="counts the keywords an OR rule finds"):
        KeywordSignal(
            name="all", operator="AND", keywords=["a", "b"], density={"gt": 1}
        )
