"""Pytest fixtures for fleet-sim API tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch) -> None:
    """Redirect all storage I/O to a throw-away temp directory."""
    import fleet_sim.api.storage as stor

    base = tmp_path / "api_store"
    traces_dir = base / "traces"
    jobs_dir = base / "jobs"
    traces_dir.mkdir(parents=True)
    jobs_dir.mkdir(parents=True)
    trace_meta = base / "traces_meta.json"
    fleets_file = base / "fleets.json"
    trace_meta.write_text("{}")
    fleets_file.write_text("{}")

    monkeypatch.setattr(stor, "_TRACES_DIR", traces_dir)
    monkeypatch.setattr(stor, "_TRACE_META", trace_meta)
    monkeypatch.setattr(stor, "_FLEETS_FILE", fleets_file)
    monkeypatch.setattr(stor, "_JOBS_DIR", jobs_dir)


@pytest.fixture()
def client(isolated_storage) -> Generator[TestClient, None, None]:
    from fleet_sim.api.app import app

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
