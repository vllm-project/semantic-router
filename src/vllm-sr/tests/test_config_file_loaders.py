import importlib
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

config_translator = importlib.import_module("cli.config_translator")
models = importlib.import_module("cli.models")
parser = importlib.import_module("cli.parser")
utils = importlib.import_module("cli.utils")

load_profile_values = config_translator.load_profile_values
EmbeddingModelsConfig = models.EmbeddingModelsConfig
ConfigParseError = parser.ConfigParseError
load_config_file = parser.load_config_file
parse_user_config = parser.parse_user_config
find_config_file = utils.find_config_file


def write_minimal_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "providers": {
                    "defaults": {"default_model": "demo-model"},
                    "models": [
                        {
                            "name": "demo-model",
                            "backend_refs": [{"endpoint": "127.0.0.1:8000"}],
                        }
                    ],
                },
                "routing": {
                    "modelCards": [{"name": "demo-model"}],
                    "decisions": [
                        {
                            "name": "default-route",
                            "description": "fallback",
                            "priority": 100,
                            "rules": {"operator": "AND", "conditions": []},
                            "modelRefs": [{"model": "demo-model"}],
                        }
                    ],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_parse_user_config_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigParseError, match="Configuration file not found"):
        parse_user_config(str(missing))


def test_parse_user_config_rejects_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\nrouting: [broken\n", encoding="utf-8")

    with pytest.raises(ConfigParseError, match="Invalid YAML syntax"):
        parse_user_config(str(config_path))


def test_parse_user_config_rejects_empty_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigParseError, match="Configuration file is empty"):
        parse_user_config(str(config_path))


def test_parse_user_config_preserves_cached_input_pricing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["providers"]["models"][0]["pricing"] = {
        "currency": "USD",
        "prompt_per_1m": 2.0,
        "cached_input_per_1m": 0.25,
        "completion_per_1m": 6.0,
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    parsed = parse_user_config(str(config_path))

    pricing = parsed.providers.models[0].pricing
    assert pricing is not None
    assert pricing.cached_input_per_1m == 0.25
    assert pricing.model_dump()["cached_input_per_1m"] == 0.25


def test_embedding_models_config_accepts_remote_endpoint() -> None:
    config = EmbeddingModelsConfig(
        embedding_config={
            "backend": "openai_compatible",
            "model_type": "remote",
            "target_dimension": 1024,
        },
        endpoint={
            "base_url": "http://embedding-service:8000/v1",
            "model": "BAAI/bge-m3",
            "api_key_env": "EMBEDDING_API_KEY",
            "timeout_seconds": 5,
            "max_retries": 2,
            "dimensions": 1024,
        },
    )

    dumped = config.model_dump(exclude_none=True)
    assert dumped["embedding_config"]["backend"] == "openai_compatible"
    assert dumped["embedding_config"]["model_type"] == "remote"
    assert dumped["endpoint"]["base_url"] == "http://embedding-service:8000/v1"
    assert dumped["endpoint"]["api_key_env"] == "EMBEDDING_API_KEY"


def test_parse_user_config_accepts_decision_learning_controls(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.setdefault("global", {}).setdefault("router", {})["learning"] = {
        "enabled": True,
        "adaptation": {
            "enabled": True,
            "strategy": "routing_sampling",
            "candidate_set": "decision",
        },
        "protection": {
            "enabled": True,
            "scope": "conversation",
        },
    }
    data["routing"]["decisions"][0]["adaptations"] = {
        "adaptation": {
            "mode": "observe",
            "candidate_set": "tier",
        },
        "protection": {
            "mode": "apply",
            "stability_weight": 1.5,
            "switch_margin": 0.11,
        },
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    parsed = parse_user_config(str(config_path))

    adaptations = parsed.decisions[0].adaptations
    assert adaptations is not None
    assert adaptations.adaptation is not None
    assert adaptations.adaptation.mode == "observe"
    assert adaptations.adaptation.candidate_set == "tier"
    assert adaptations.protection is not None
    assert adaptations.protection.mode == "apply"
    assert adaptations.protection.stability_weight == 1.5
    assert adaptations.protection.switch_margin == 0.11


def test_parse_user_config_rejects_unknown_decision_adaptation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["routing"]["decisions"][0]["adaptations"] = {
        "unknown_learning": {"mode": "apply"}
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError, match="unknown_learning"):
        parse_user_config(str(config_path))


def test_parse_user_config_rejects_removed_decision_coordination(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["routing"]["decisions"][0]["adaptations"] = {
        "coordination": {"protection_weight": 1.0}
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError, match="coordination"):
        parse_user_config(str(config_path))


def test_parse_user_config_rejects_protection_candidate_set(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["routing"]["decisions"][0]["adaptations"] = {
        "protection": {
            "mode": "apply",
            "candidate_set": "tier",
        }
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError, match="candidate_set"):
        parse_user_config(str(config_path))


def test_parse_user_config_rejects_decision_observe_component_apply(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["routing"]["decisions"][0]["adaptations"] = {
        "mode": "observe",
        "adaptation": {"mode": "apply"},
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError, match="cannot be apply"):
        parse_user_config(str(config_path))


def test_parse_user_config_rejects_decision_bypass_component_observe(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["routing"]["decisions"][0]["adaptations"] = {
        "mode": "bypass",
        "protection": {"mode": "observe"},
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError, match="cannot be observe"):
        parse_user_config(str(config_path))


def test_parse_user_config_rejects_removed_decision_protection_weight(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["routing"]["decisions"][0]["adaptations"] = {
        "protection": {
            "weight": 1.0,
        }
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError, match="weight"):
        parse_user_config(str(config_path))


@pytest.mark.parametrize(
    "learning_patch, expected_path",
    [
        (
            {"adaptations": {"session_aware": {"enabled": True}}},
            "global.router.learning.adaptations",
        ),
        (
            {"adaptation": {"routing_sampling": {"enabled": True}}},
            "global.router.learning.adaptation.routing_sampling",
        ),
        (
            {"protection": {"privacy_affinity": True}},
            "global.router.learning.protection.privacy_affinity",
        ),
        (
            {"protection": {"identity": {"headers": {"run": "x-run-id"}}}},
            "global.router.learning.protection.identity.headers.run",
        ),
        (
            {"protection": {"tuning": {"prefix_cache_weight": 0.2}}},
            "global.router.learning.protection.tuning.prefix_cache_weight",
        ),
        (
            {"protection": {"tuning": {"weight": 1.0}}},
            "global.router.learning.protection.tuning.weight",
        ),
        (
            {"protection": {"tuning": {"protection_weight": 1.0}}},
            "global.router.learning.protection.tuning.protection_weight",
        ),
        (
            {"protection": {"tuning": {"handoff_penalty_weight": 1.0}}},
            "global.router.learning.protection.tuning.handoff_penalty_weight",
        ),
        (
            {"protection": {"tuning": {"cache_weight": 0.2}}},
            "global.router.learning.protection.tuning.cache_weight",
        ),
        (
            {"protection": {"tuning": {"handoff_penalty": 0.05}}},
            "global.router.learning.protection.tuning.handoff_penalty",
        ),
        (
            {"protection": {"tuning": {"switch_history_weight": 0.04}}},
            "global.router.learning.protection.tuning.switch_history_weight",
        ),
        (
            {"protection": {"tuning": {"max_cache_cost_multiplier": 2.5}}},
            "global.router.learning.protection.tuning.max_cache_cost_multiplier",
        ),
    ],
)
def test_parse_user_config_rejects_unknown_global_learning_fields(
    tmp_path: Path,
    learning_patch: dict,
    expected_path: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.setdefault("global", {}).setdefault("router", {})["learning"] = {
        "enabled": True,
        **learning_patch,
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError) as exc:
        parse_user_config(str(config_path))

    assert "Unsupported Router Learning config fields" in str(exc.value)
    assert expected_path in str(exc.value)


@pytest.mark.parametrize(
    "learning_patch, expected_text",
    [
        (
            {"adaptation": {"candidate_set": "cluster"}},
            "global.router.learning.adaptation.candidate_set",
        ),
        (
            {"adaptation": {"strategy": "linucb"}},
            "global.router.learning.adaptation.strategy",
        ),
        (
            {"protection": {"scope": "run"}},
            "global.router.learning.protection.scope",
        ),
        (
            {"protection": {"identity": {"headers": {"session": ""}}}},
            "global.router.learning.protection.identity.headers.session",
        ),
        (
            {"protection": {"tuning": {"switch_margin": -0.1}}},
            "global.router.learning.protection.tuning.switch_margin",
        ),
        (
            {"protection": {"tuning": {"min_turns_before_switch": 1.5}}},
            "global.router.learning.protection.tuning.min_turns_before_switch",
        ),
        (
            {"enabled": "yes"},
            "global.router.learning.enabled",
        ),
        (
            {"adaptation": []},
            "global.router.learning.adaptation",
        ),
    ],
)
def test_parse_user_config_rejects_invalid_global_learning_values(
    tmp_path: Path,
    learning_patch: dict,
    expected_text: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.setdefault("global", {}).setdefault("router", {})["learning"] = {
        "enabled": True,
        **learning_patch,
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError) as exc:
        parse_user_config(str(config_path))

    assert "Invalid Router Learning config values" in str(exc.value)
    assert expected_text in str(exc.value)


def test_parse_user_config_rejects_unknown_pricing_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["providers"]["models"][0]["pricing"] = {
        "currency": "USD",
        "prompt_per_1m": 2.0,
        "cached_input": 0.25,
        "completion_per_1m": 6.0,
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError) as exc:
        parse_user_config(str(config_path))

    assert "cached_input" in str(exc.value)
    assert "Extra inputs are not permitted" in str(exc.value)


def test_parse_user_config_rejects_removed_session_aware_algorithm(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["routing"]["decisions"][0]["algorithm"] = {
        "type": "session_aware",
        "session_aware": {"fallback_method": "static"},
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError) as exc:
        parse_user_config(str(config_path))

    assert "Removed Router Learning config fields" in str(exc.value)
    assert "global.router.learning.protection" in str(exc.value)


@pytest.mark.parametrize(
    "algorithm_type",
    ["elo", "rl_driven", "gmtrouter", "bandit", "personalization"],
)
def test_parse_user_config_rejects_migrated_learning_algorithms(
    tmp_path: Path, algorithm_type: str
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["routing"]["decisions"][0]["algorithm"] = {"type": algorithm_type}
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError) as exc:
        parse_user_config(str(config_path))

    assert "Removed Router Learning config fields" in str(exc.value)
    assert f"algorithm.type={algorithm_type}" in str(exc.value)


@pytest.mark.parametrize(
    "method",
    [
        "session_aware",
        "lookup_tables",
        "elo",
        "rl_driven",
        "gmtrouter",
        "bandit",
        "personalization",
    ],
)
def test_parse_user_config_rejects_removed_global_learning_selector_methods(
    tmp_path: Path, method: str
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.setdefault("global", {}).setdefault("router", {})["model_selection"] = {
        "method": method
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigParseError) as exc:
        parse_user_config(str(config_path))

    assert "Removed Router Learning config fields" in str(exc.value)
    assert f"global.router.model_selection.method={method}" in str(exc.value)


def test_load_config_file_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ConfigParseError, match="Configuration file not found"):
        load_config_file(str(missing))


def test_load_config_file_rejects_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\nproviders: [broken\n", encoding="utf-8")

    with pytest.raises(ConfigParseError, match="Invalid YAML syntax"):
        load_config_file(str(config_path))


def test_load_config_file_returns_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)

    loaded = load_config_file(str(config_path))

    assert loaded["version"] == "v0.3"
    assert loaded["providers"]["defaults"]["default_model"] == "demo-model"


def test_find_config_file_returns_explicit_file_path(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("version: v0.3\n", encoding="utf-8")

    found = find_config_file(path=str(tmp_path), file=str(config_path))

    assert found == str(config_path.resolve())


def test_find_config_file_finds_root_config_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\n", encoding="utf-8")

    found = find_config_file(path=str(tmp_path))

    assert found == str(config_path.resolve())


def test_find_config_file_finds_nested_config_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("version: v0.3\n", encoding="utf-8")

    found = find_config_file(path=str(tmp_path))

    assert found == str(config_path.resolve())


def test_find_config_file_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        find_config_file(path=str(tmp_path))


def test_load_profile_values_returns_none_without_profile(tmp_path: Path) -> None:
    assert load_profile_values(None, str(tmp_path)) is None


def test_load_profile_values_returns_none_when_profile_file_missing(
    tmp_path: Path,
) -> None:
    assert load_profile_values("dev", str(tmp_path)) is None


def test_load_profile_values_loads_named_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "values-dev.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {"image": {"tag": "dev"}, "dependencies": {"observability": {}}}
        ),
        encoding="utf-8",
    )

    loaded = load_profile_values("dev", str(tmp_path))

    assert loaded == {"image": {"tag": "dev"}, "dependencies": {"observability": {}}}
