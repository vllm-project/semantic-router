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
