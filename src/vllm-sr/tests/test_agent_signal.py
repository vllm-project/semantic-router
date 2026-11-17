"""The agent signal and the blocking fast_response as the CLI reads them, with
the Router's contract: a rule sets at least one of role, harness and types,
types name a subagent's type, and a decision can refuse what it matches."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.models import AgentRule, FastResponsePluginConfig, UserConfig  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402


def _config(signal: dict) -> UserConfig:
    return UserConfig.model_validate(
        {
            "version": "v0.5",
            "providers": {
                "models": [
                    {
                        "name": "small",
                        "backend_refs": [
                            {"name": "local", "endpoint": "127.0.0.1:8000"}
                        ],
                    }
                ]
            },
            "recipes": [
                {
                    "name": "default",
                    "signals": {"agent": [signal]},
                    "decisions": [
                        {
                            "name": "explore",
                            "priority": 100,
                            "rules": {"type": "agent", "name": signal["name"]},
                            "modelRefs": [
                                {
                                    "model": "small",
                                    "use_reasoning": True,
                                    "reasoning_effort": "low",
                                }
                            ],
                        },
                        {
                            "name": "block",
                            "priority": 200,
                            "rules": {"type": "agent", "name": signal["name"]},
                            "plugins": [
                                {
                                    "type": "fast_response",
                                    "configuration": {
                                        "message": "Blocked by policy.",
                                        "block": True,
                                    },
                                }
                            ],
                        },
                    ],
                }
            ],
            "entrypoints": [
                {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True}
            ],
        }
    )


def test_a_decision_routes_and_blocks_by_agent_rule():
    config = _config(
        {"name": "explore", "role": "subagent", "types": ["Explore", "collab_spawn"]}
    )

    assert config.recipes[0].signals.agent[0].types == ["Explore", "collab_spawn"]
    messages = [
        error.message for error in validate_user_config(config, log_summary=False)
    ]
    assert not [m for m in messages if "agent" in m or "explore" in m], messages


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        (
            {"name": "any"},
            "set at least one of role, harness, types or request_classes",
        ),
        ({"name": "x", "request_classes": ["title"]}, "request_classes"),
        (
            {"name": "x", "harness": "codex", "request_classes": ["compaction"]},
            "request_classes cannot go with harness codex",
        ),
        (
            {
                "name": "x",
                "harness": "deepseek_harness",
                "request_classes": ["auxiliary"],
            },
            "request_classes cannot go with harness deepseek_harness",
        ),
        ({"name": "x", "role": "child"}, "role"),
        ({"name": "x", "harness": "cursor"}, "harness"),
        (
            {"name": "x", "role": "main", "types": ["Explore"]},
            "types name a subagent's type",
        ),
        ({"name": "x", "types": [" "]}, "types must be nonempty and trimmed"),
        ({"name": " x ", "role": "main"}, "nonempty and trimmed"),
    ],
)
def test_agent_rule_refuses_what_the_router_refuses(rule, message):
    with pytest.raises(ValidationError, match=message):
        AgentRule.model_validate(rule)


@pytest.mark.parametrize(
    "harness", ["claude_code", "codex", "opencode", "deepseek_harness"]
)
def test_agent_rule_accepts_every_harness_the_router_names(harness):
    assert (
        AgentRule.model_validate({"name": "x", "harness": harness}).harness == harness
    )


def test_agent_rule_matches_request_classes():
    rule = AgentRule.model_validate(
        {"name": "housekeeping", "request_classes": ["compaction", "auxiliary"]}
    )
    assert rule.request_classes == ["compaction", "auxiliary"]


def test_agent_rule_names_are_unique_ignoring_case():
    with pytest.raises(ValidationError, match="agent signal names must be unique"):
        UserConfig.model_validate(
            {
                "version": "v0.5",
                "recipes": [
                    {
                        "name": "default",
                        "signals": {
                            "agent": [
                                {"name": "sub", "role": "subagent"},
                                {"name": "Sub", "role": "main"},
                            ]
                        },
                    }
                ],
            }
        )


def test_fast_response_block_defaults_to_answering():
    assert FastResponsePluginConfig(message="busy").block is False
    assert FastResponsePluginConfig(message="no", block=True).block is True
