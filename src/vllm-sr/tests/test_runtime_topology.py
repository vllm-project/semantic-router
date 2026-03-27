import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

consts = importlib.import_module("cli.consts")
runtime_topology = importlib.import_module("cli.runtime_topology")

RUNTIME_TOPOLOGY_ENV = consts.RUNTIME_TOPOLOGY_ENV
RUNTIME_TOPOLOGY_LEGACY = consts.RUNTIME_TOPOLOGY_LEGACY
RUNTIME_TOPOLOGY_SPLIT = consts.RUNTIME_TOPOLOGY_SPLIT
resolve_runtime_topology = runtime_topology.resolve_runtime_topology
split_runtime_enabled = runtime_topology.split_runtime_enabled


def test_runtime_topology_defaults_to_split(monkeypatch):
    monkeypatch.delenv(RUNTIME_TOPOLOGY_ENV, raising=False)

    assert resolve_runtime_topology() == RUNTIME_TOPOLOGY_SPLIT
    assert split_runtime_enabled() is True


def test_runtime_topology_accepts_split_env(monkeypatch):
    monkeypatch.setenv(RUNTIME_TOPOLOGY_ENV, RUNTIME_TOPOLOGY_SPLIT)

    assert resolve_runtime_topology() == RUNTIME_TOPOLOGY_SPLIT
    assert split_runtime_enabled() is True


def test_runtime_topology_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv(RUNTIME_TOPOLOGY_ENV, RUNTIME_TOPOLOGY_LEGACY)

    assert resolve_runtime_topology(RUNTIME_TOPOLOGY_SPLIT) == RUNTIME_TOPOLOGY_SPLIT
    assert split_runtime_enabled(RUNTIME_TOPOLOGY_SPLIT) is True


def test_runtime_topology_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv(RUNTIME_TOPOLOGY_ENV, "sidecar")

    with pytest.raises(ValueError, match="Invalid runtime topology"):
        resolve_runtime_topology()
