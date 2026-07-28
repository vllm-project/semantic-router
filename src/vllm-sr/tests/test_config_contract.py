import json
from pathlib import Path

import pytest
import yaml
from cli.config_contract import (
    CANONICAL_VERSION,
    LEGACY_SIGNAL_KEY_TO_CANONICAL,
    build_projection_reference_index,
    build_signal_reference_index,
    signal_reference_exists,
)
from cli.config_migration import migrate_config_data
from cli.models import Decision, Projections, Signals, UserConfig
from cli.parser import ConfigParseError, parse_user_config

CONTRACT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "semantic-router"
    / "pkg"
    / "config"
    / "testdata"
    / "canonical_contract_cases.json"
)


def load_contract_corpus() -> dict:
    return json.loads(CONTRACT_CORPUS_PATH.read_text(encoding="utf-8"))


def test_cli_executes_canonical_contract_golden_corpus(tmp_path: Path):
    corpus = load_contract_corpus()
    assert corpus["supported_version"] == CANONICAL_VERSION

    for case in corpus["steady_state"]:
        config_path = tmp_path / f"{case['name']}.yaml"
        config_path.write_text(case["input"], encoding="utf-8")
        if not case["valid"]:
            with pytest.raises(ConfigParseError) as exc:
                parse_user_config(str(config_path))
            assert case["error"] in str(exc.value)
            continue

        parsed = parse_user_config(str(config_path))
        assert parsed.version == case["normalized_version"]


def test_cli_executes_canonical_migration_golden_corpus():
    corpus = load_contract_corpus()

    for case in corpus["migrations"]:
        assert migrate_config_data(case["input"]) == case["normalized"], case["name"]


def test_cli_accepts_exhaustive_router_reference_config():
    config_path = Path(__file__).resolve().parents[3] / "config" / "config.yaml"

    parsed = parse_user_config(str(config_path))

    assert parsed.version == CANONICAL_VERSION
    assert parsed.routing.decisions


def test_legacy_signal_inventory_covers_flat_authz_and_context_blocks():
    assert LEGACY_SIGNAL_KEY_TO_CANONICAL["role_bindings"] == "role_bindings"
    assert LEGACY_SIGNAL_KEY_TO_CANONICAL["context_rules"] == "context"
    assert LEGACY_SIGNAL_KEY_TO_CANONICAL["events"] == "events"
    assert "session_metrics" not in LEGACY_SIGNAL_KEY_TO_CANONICAL


def test_build_signal_reference_index_expands_complexity_levels_and_authz_names():
    signals = Signals(
        complexity=[
            {
                "name": "difficulty",
                "easy": {"candidates": ["simple"]},
                "hard": {"candidates": ["complex"]},
            }
        ],
        role_bindings=[
            {
                "name": "admin-access",
                "role": "admin",
                "subjects": [{"kind": "User", "name": "alice"}],
            }
        ],
        events=[
            {
                "name": "critical_event",
                "event_types": ["payment_failed"],
                "severities": ["critical"],
            }
        ],
    )

    signal_names = build_signal_reference_index(signals)

    assert "difficulty:easy" in signal_names
    assert "difficulty:medium" in signal_names
    assert "difficulty:hard" in signal_names
    assert "admin-access" in signal_names
    assert "critical_event" in signal_names


def test_signal_reference_exists_strips_suffixes_for_non_complexity_signals():
    signal_names = {"security", "admin-access"}

    assert signal_reference_exists(signal_names, "keyword", "security:match")
    assert signal_reference_exists(signal_names, "authz", "admin-access")
    assert not signal_reference_exists(signal_names, "complexity", "security:match")


def test_projection_reference_index_collects_mapping_outputs():
    projections = Projections(
        mappings=[
            {
                "name": "difficulty_band",
                "source": "difficulty_score",
                "method": "threshold_bands",
                "outputs": [
                    {"name": "balance_simple", "lt": 0.15},
                    {"name": "balance_medium", "gte": 0.15, "lt": 0.45},
                ],
            }
        ]
    )

    assert build_projection_reference_index(projections) == {
        "balance_simple",
        "balance_medium",
    }


def test_decision_accepts_typed_output_contract_spec():
    decision = Decision(
        name="gpqa",
        description="strict choice route",
        priority=100,
        rules={"operator": "AND", "conditions": []},
        output_contract="Return exactly one answer letter: A, B, C, or D.",
        output_contract_spec={
            "type": "choice",
            "choice_set": {"values": ["A", "B", "C", "D"]},
            "render": {"mode": "value"},
            "extract": {"mode": "exact", "sources": ["content"]},
        },
        modelRefs=[{"model": "model-a", "use_reasoning": False}],
    )

    assert decision.output_contract_spec is not None
    assert decision.output_contract_spec.choice_set is not None
    assert decision.output_contract_spec.choice_set.values == ["A", "B", "C", "D"]


def test_decision_accepts_terminal_action_output_contract_spec():
    decision = Decision(
        name="terminal",
        description="terminal action route",
        priority=100,
        rules={"operator": "AND", "conditions": []},
        output_contract="Return one terminal action JSON object.",
        output_contract_spec={
            "type": "structured_json",
            "json_schema": {"schema_ref": "terminal_action_v1"},
            "extract": {
                "mode": "json_object",
                "sources": ["content", "candidate_responses"],
            },
        },
        modelRefs=[{"model": "model-a", "use_reasoning": False}],
    )

    assert decision.output_contract_spec is not None
    assert decision.output_contract_spec.json_schema is not None
    assert decision.output_contract_spec.json_schema.schema_ref == "terminal_action_v1"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (None, "version: required"),
        ("", "version: must not be empty"),
        (3, "version: must be a string"),
        ("v0.1", "unsupported config version"),
        ("v99.0", "unsupported config version"),
    ],
)
def test_parser_enforces_canonical_version_before_interpretation(
    tmp_path: Path, version, expected: str
):
    config = {"routing": {"unknown_before_interpretation": True}}
    if version is not None:
        config["version"] = version
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigParseError, match=expected):
        parse_user_config(str(path))


def test_nested_unknown_field_has_stable_indexed_path(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": CANONICAL_VERSION,
                "routing": {
                    "modelCards": [
                        {"name": "demo", "descriptino": "silently-dropped-before"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigParseError) as exc_info:
        parse_user_config(str(path))

    assert "routing.modelCards[0].descriptino" in str(exc_info.value)
    assert 'did you mean "description"' in str(exc_info.value)


def test_plugin_owner_rejects_unknown_backend_field():
    with pytest.raises(ValueError):
        UserConfig.model_validate(
            {
                "version": CANONICAL_VERSION,
                "routing": {
                    "decisions": [
                        {
                            "name": "docs",
                            "description": "docs",
                            "priority": 1,
                            "rules": {"operator": "AND", "conditions": []},
                            "modelRefs": [],
                            "plugins": [
                                {
                                    "type": "rag",
                                    "configuration": {
                                        "enabled": True,
                                        "backend": "mcp",
                                        "backend_config": {
                                            "server_nam": "docs",
                                            "tool_name": "search",
                                        },
                                    },
                                }
                            ],
                        }
                    ]
                },
            }
        )


def test_plugin_named_extension_allows_arbitrary_tool_arguments():
    config = UserConfig.model_validate(
        {
            "version": CANONICAL_VERSION,
            "routing": {
                "decisions": [
                    {
                        "name": "docs",
                        "description": "docs",
                        "priority": 1,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [],
                        "plugins": [
                            {
                                "type": "rag",
                                "configuration": {
                                    "enabled": True,
                                    "backend": "mcp",
                                    "backend_config": {
                                        "server_name": "docs",
                                        "tool_name": "search",
                                        "tool_arguments": {
                                            "custom_filter": {"nested": True}
                                        },
                                    },
                                },
                            }
                        ],
                    }
                ]
            },
        }
    )

    assert (
        config.decisions[0]
        .plugins[0]
        .configuration["backend_config"]["tool_arguments"]["custom_filter"]["nested"]
    )
