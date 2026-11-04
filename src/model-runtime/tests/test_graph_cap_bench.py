"""graph_cap_bench's input handling (``tools/graph_cap_bench.py``)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "graph_cap_bench.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("graph_cap_bench", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_prompts(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_stream_returns_no_requests_for_an_empty_file(tool, tmp_path):
    prompts = _write_prompts(tmp_path / "empty.jsonl", [])
    assert tool.stream([f"s:{prompts}:10"], [1]) == []


def test_main_rejects_an_empty_stream_before_loading_the_model(
    tool, tmp_path, monkeypatch
):
    prompts = _write_prompts(tmp_path / "empty.jsonl", [])
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "graph_cap_bench.py",
            "--package",
            str(tmp_path / "package"),
            "--stream",
            f"s:{prompts}:10",
            "--cap",
            "4096",
            "--output",
            str(out),
        ],
    )
    with pytest.raises(SystemExit, match="contains no requests"):
        tool.main()
    assert not out.exists()
