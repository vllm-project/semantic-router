import importlib
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

config_translator = importlib.import_module("cli.config_translator")
parser = importlib.import_module("cli.parser")
utils = importlib.import_module("cli.utils")

load_profile_values = config_translator.load_profile_values
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


def test_parse_user_config_accepts_decision_session_aware_overrides(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.setdefault("global", {}).setdefault("router", {})["learning"] = {
        "enabled": True,
        "adaptations": {
            "session_aware": {
                "enabled": True,
                "scope": "conversation",
            }
        },
    }
    data["routing"]["decisions"][0]["adaptations"] = {
        "session_aware": {
            "mode": "observe",
            "scope": "session",
            "tuning": {
                "switch_margin": 0.11,
                "cache_weight": 0.25,
            },
        }
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    parsed = parse_user_config(str(config_path))

    adaptation = parsed.decisions[0].adaptations.session_aware
    assert adaptation is not None
    assert adaptation.mode == "observe"
    assert adaptation.scope == "session"
    assert adaptation.tuning is not None
    assert adaptation.tuning.switch_margin == 0.11
    assert adaptation.tuning.cache_weight == 0.25


def test_parse_user_config_accepts_learning_adaptation_controls(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_minimal_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data.setdefault("global", {}).setdefault("router", {})["learning"] = {
        "enabled": True,
        "adaptations": {
            "bandit": {
                "enabled": True,
                "algorithm": "linucb",
                "scope": "decision",
                "goals": {"quality": 1.0, "cost": 0.25, "latency": 0.1},
                "tuning": {"exploration_budget": 0.05},
            },
            "elo": {"enabled": True, "scope": "decision"},
            "personalization": {"enabled": True, "scope": "session"},
        },
    }
    data["routing"]["decisions"][0]["adaptations"] = {
        "bandit": {
            "mode": "observe",
            "scope": "decision",
            "goals": {"quality": 1.0, "cost": 0.1},
            "tuning": {"exploration_budget": 0.01},
        },
        "elo": {"mode": "apply"},
        "personalization": {"mode": "bypass"},
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    parsed = parse_user_config(str(config_path))

    adaptations = parsed.decisions[0].adaptations
    assert adaptations.bandit is not None
    assert adaptations.bandit.mode == "observe"
    assert adaptations.bandit.scope == "decision"
    assert adaptations.bandit.goals == {"quality": 1.0, "cost": 0.1}
    assert adaptations.bandit.tuning.exploration_budget == 0.01
    assert adaptations.elo is not None
    assert adaptations.elo.mode == "apply"
    assert adaptations.personalization is not None
    assert adaptations.personalization.mode == "bypass"


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
    assert "global.router.learning.adaptations.session_aware" in str(exc.value)


@pytest.mark.parametrize("algorithm_type", ["elo", "rl_driven", "gmtrouter"])
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
