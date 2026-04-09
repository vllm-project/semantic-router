"""Tests for vllm-sr chat and chat_client helpers."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

chat_client = importlib.import_module("cli.chat_client")
chat_command = importlib.import_module("cli.commands.chat")
main = importlib.import_module("cli.main").main


def _patch_stack_layout(monkeypatch: pytest.MonkeyPatch, port_offset: int) -> None:
    class _Layout:
        pass

    _layout = _Layout()
    _layout.port_offset = port_offset

    def _fake_resolve():
        return _layout

    monkeypatch.setattr(chat_client, "resolve_runtime_stack", _fake_resolve)


def test_resolve_chat_base_url_k8s_not_supported():
    with pytest.raises(ValueError, match="Non-Docker"):
        chat_client.resolve_chat_base_url(config_path="config.yaml", target="k8s")


def test_resolve_listener_host_port_with_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _patch_stack_layout(monkeypatch, 10)
    host, port = chat_client.resolve_listener_host_port(str(cfg))
    assert host == "localhost"
    assert port == 8899 + 10


def test_build_chat_payload_and_extract():
    payload = chat_client.build_chat_payload(
        model="MoM",
        user_text="hi",
        system_text="be brief",
        temperature=0.2,
    )
    assert payload["model"] == "MoM"
    assert payload["temperature"] == 0.2
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"

    text = chat_client.extract_assistant_text(
        {"choices": [{"message": {"content": "hello there"}}]}
    )
    assert text == "hello there"


def test_extract_assistant_text_api_error():
    with pytest.raises(ValueError, match="API error"):
        chat_client.extract_assistant_text(
            {"error": {"message": "bad request", "type": "invalid"}}
        )


def test_chat_completions_url():
    u = chat_client.chat_completions_url("http://localhost:8899")
    assert u.endswith("/v1/chat/completions")
    assert u.startswith("http://localhost:8899")


def test_cli_chat_invokes_post(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _patch_stack_layout(monkeypatch, 0)

    monkeypatch.setattr(chat_command.DockerBackend, "is_running", lambda self: True)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "assistant says hi"}}]
    }
    mock_post = MagicMock(return_value=mock_resp)
    monkeypatch.setattr(chat_command.requests, "post", mock_post)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "hello", "--config", str(cfg)],
    )

    assert result.exit_code == 0
    assert result.output.strip() == "assistant says hi"
    mock_post.assert_called_once()
    call_kw = mock_post.call_args.kwargs
    assert "json" in call_kw
    assert call_kw["json"]["model"] == "MoM"
    assert call_kw["json"]["messages"][-1]["content"] == "hello"


def test_cli_chat_json_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _patch_stack_layout(monkeypatch, 0)
    monkeypatch.setattr(chat_command.DockerBackend, "is_running", lambda self: True)

    body = {"choices": [{"message": {"content": "x"}}]}
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = body
    monkeypatch.setattr(
        chat_command.requests, "post", MagicMock(return_value=mock_resp)
    )

    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--json", "hi", "--config", str(cfg)])

    assert result.exit_code == 0
    assert json.loads(result.output) == body


def test_cli_chat_not_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _patch_stack_layout(monkeypatch, 0)
    monkeypatch.setattr(chat_command.DockerBackend, "is_running", lambda self: False)

    with caplog.at_level("ERROR", logger="cli.commands.chat"):
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "hello", "--config", str(cfg)])

    assert result.exit_code != 0
    assert "does not appear to be running" in caplog.text


def test_cli_chat_connection_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _patch_stack_layout(monkeypatch, 0)
    monkeypatch.setattr(chat_command.DockerBackend, "is_running", lambda self: True)

    monkeypatch.setattr(
        chat_command.requests,
        "post",
        MagicMock(side_effect=requests.exceptions.ConnectionError("refused")),
    )

    with caplog.at_level("ERROR", logger="cli.commands.chat"):
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "hello", "--config", str(cfg)])

    assert result.exit_code != 0
    assert "Could not reach" in caplog.text
