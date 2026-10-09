"""`vllm-sr config migrate`: the retired Looper endpoint."""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_migration import migrate_config_data  # noqa: E402
from cli.config_migration_looper import LOOPER_ENDPOINT_PATH  # noqa: E402
from cli.config_migration_notes import MigrationNotes  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402

ENDPOINT = "http://localhost:8899/v1/chat/completions"


def _migrate(looper):
    notes = MigrationNotes()
    migrated = migrate_config_data(
        {
            "version": "v0.3",
            "listeners": [],
            "providers": {"defaults": {"model": "general"}},
            "routing": {"modelCards": [{"name": "general"}]},
            "global": {"integrations": {"looper": looper}},
        },
        notes,
    )
    looper_notes = [note.path for note in notes if note.path == LOOPER_ENDPOINT_PATH]
    return migrated, looper_notes


def test_migrate_drops_the_looper_endpoint_and_keeps_its_other_settings():
    migrated, notes = _migrate({"endpoint": ENDPOINT, "timeout_seconds": 300})

    assert migrated["global"]["integrations"]["looper"] == {"timeout_seconds": 300}
    assert notes == [LOOPER_ENDPOINT_PATH]


def test_migrate_drops_a_looper_block_that_only_held_the_endpoint():
    migrated, notes = _migrate({"endpoint": ENDPOINT})

    assert "integrations" not in migrated["global"]
    assert notes == [LOOPER_ENDPOINT_PATH]


def test_migrate_leaves_a_looper_block_without_the_endpoint_alone():
    migrated, notes = _migrate({"timeout_seconds": 300})

    assert migrated["global"]["integrations"]["looper"] == {"timeout_seconds": 300}
    assert notes == []


def _reference_config(tmp_path, looper_endpoint=None):
    config = yaml.safe_load(
        (PROJECT_ROOT.parents[1] / "config" / "config.yaml").read_text(encoding="utf-8")
    )
    looper = config["global"]["integrations"]["looper"]
    looper.pop("endpoint", None)
    if looper_endpoint is not None:
        looper["endpoint"] = looper_endpoint
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _parse_warnings(monkeypatch, path):
    warnings = []
    monkeypatch.setattr("cli.parser.log.warning", warnings.append)
    parse_user_config(str(path), log_summary=False)
    return [message for message in warnings if LOOPER_ENDPOINT_PATH in message]


def test_parse_accepts_the_looper_endpoint_with_a_deprecation_warning(
    tmp_path, monkeypatch
):
    path = _reference_config(tmp_path, ENDPOINT)

    [warning] = _parse_warnings(monkeypatch, path)

    assert "deprecated and ignored" in warning
    assert f"vllm-sr config migrate --config {path}" in warning


def test_parse_says_nothing_without_the_looper_endpoint(tmp_path, monkeypatch):
    assert _parse_warnings(monkeypatch, _reference_config(tmp_path)) == []
