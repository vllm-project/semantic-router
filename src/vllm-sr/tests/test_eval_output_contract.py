"""Stream and width contracts for eval terminal output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import requests
from cli.commands.eval import eval as eval_command
from click.testing import CliRunner

CLI_ROOT = Path(__file__).resolve().parents[1]


def _run_cli_subprocess(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CLI_ROOT)
    environment["PYDANTIC_DISABLE_PLUGINS"] = "__all__"
    return subprocess.run(
        [sys.executable, "-m", "cli.main", *args],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _learning_record() -> dict:
    return {
        "id": "replay-1",
        "request_id": "req-1",
        "decision": "simple_general",
        "decision_tier": 1,
        "original_model": "frontier",
        "selected_model": "small",
        "prompt_tokens": 1000,
        "cached_prompt_tokens": 250,
        "total_tokens": 1100,
        "actual_cost": 0.2,
        "baseline_cost": 0.5,
        "cost_savings": 0.3,
        "route_diagnostics": {
            "decision": "simple_general",
            "selected_model": "small",
            "original_model": "frontier",
        },
        "learning": {
            "adaptation": {"action": "propose_switch"},
            "protection": {"action": "hold_current"},
        },
        "outcomes": [
            {
                "source": "eval",
                "target": "model",
                "target_ref": "small",
                "verdict": "overprovisioned",
                "score": 1,
            }
        ],
    }


def test_recipe_learning_json_is_pure_in_subprocess(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps({"object": "router_replay.list", "data": [_learning_record()]}),
        encoding="utf-8",
    )

    result = _run_cli_subprocess(
        tmp_path,
        "eval",
        "recipe-learning",
        "--replay-file",
        str(replay_path),
        "--json",
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    artifact = json.loads(result.stdout)
    assert artifact["object"] == "router_learning.recipe_learning"


def test_recipe_learning_network_error_is_clean_and_redacted(tmp_path: Path) -> None:
    result = _run_cli_subprocess(
        tmp_path,
        "eval",
        "recipe-learning",
        "--endpoint",
        "http://user:TOPSECRET@127.0.0.1:1",
        "--timeout",
        "1",
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("Error: Router replay endpoint is not reachable")
    assert "Traceback" not in result.stderr
    assert "TOPSECRET" not in result.stderr
    assert "***@127.0.0.1:1" in result.stderr


def test_eval_network_error_is_clean_and_redacted(tmp_path: Path) -> None:
    result = _run_cli_subprocess(
        tmp_path,
        "eval",
        "--prompt",
        "hello",
        "--endpoint",
        "http://user:EVALSECRET@127.0.0.1:1",
        "--timeout",
        "1",
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("Error: Router is not running at")
    assert "Traceback" not in result.stderr
    assert "EVALSECRET" not in result.stderr
    assert "***@127.0.0.1:1" in result.stderr


def test_eval_readable_output_wraps_long_values(monkeypatch) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "requested_model": "vllm-sr/a-very-long-mixture-of-model-entrypoint-name",
        "decision_result": {
            "decision_name": "long-output",
            "algorithm": "multi_factor",
            "plugins": [
                "response_cache",
                "request_parameters",
                "system_prompt_rewriting",
            ],
            "matched_signals": {},
            "unmatched_signals": {},
            "used_signals": [],
        },
        "signal_confidences": {},
    }
    monkeypatch.setattr(requests, "post", MagicMock(return_value=mock_resp))
    monkeypatch.setattr("cli.terminal.output_width", lambda: 48)

    result = CliRunner().invoke(eval_command, ["--prompt", "hello"])

    assert result.exit_code == 0
    assert result.stderr == ""
    assert max(map(len, result.stdout.splitlines())) <= 48


def test_eval_unknown_schema_fallback_wraps_on_narrow_terminal(monkeypatch) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"unknown": "x" * 100}
    monkeypatch.setattr(requests, "post", MagicMock(return_value=mock_resp))
    monkeypatch.setattr("cli.terminal.output_width", lambda: 40)
    monkeypatch.setattr("cli.commands.eval_rendering.output_width", lambda: 40)

    result = CliRunner().invoke(eval_command, ["--prompt", "hello"])

    assert result.exit_code == 0
    assert result.stderr == ""
    assert max(map(len, result.stdout.splitlines())) <= 40
