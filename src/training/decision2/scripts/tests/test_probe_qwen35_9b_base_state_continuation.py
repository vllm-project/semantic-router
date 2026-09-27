"""CPU-only trajectory and ledger checks for the 9B state replay."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _probe():
    scripts = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "probe_qwen35_9b_base_state_continuation",
        scripts / "probe_qwen35_9b_base_state_continuation.py",
    )
    assert spec and spec.loader
    import sys

    sys.path.insert(0, str(scripts))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_comparison_rejects_token_and_numeric_drift() -> None:
    probe = _probe()
    expected = {"tokens": 100, "loss": 0.5, "gradient_norm": 2.0}
    assert probe.compare_step(65, expected, expected) == {
        "loss": 0.0,
        "gradient_norm": 0.0,
    }
    with pytest.raises(ValueError, match="token schedule"):
        probe.compare_step(65, {**expected, "tokens": 101}, expected)
    with pytest.raises(ValueError, match="loss trajectory"):
        probe.compare_step(65, {**expected, "loss": 0.51}, expected)
    with pytest.raises(ValueError, match="gradient_norm trajectory"):
        probe.compare_step(65, {**expected, "gradient_norm": 2.11}, expected)


def test_original_step_roster_is_complete(tmp_path: Path) -> None:
    probe = _probe()
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"event": "train", "step": step, "tokens": step}
        for step in range(probe.FIRST_UPDATE, probe.LAST_UPDATE)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert len(probe.original_steps(path)) == 42
    path.write_text("\n".join(json.dumps(row) for row in rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="roster differs"):
        probe.original_steps(path)
