"""End-user CLI presentation and stream contracts."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from cli.commands import runtime
from cli.main import main
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]


def _run_cli(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYDANTIC_DISABLE_PLUGINS"] = "__all__"
    return subprocess.run(
        [sys.executable, "-m", "cli.main", *args],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_no_log_decoration(value: str) -> None:
    assert " - INFO - " not in value
    assert " - ERROR - " not in value
    assert not re.search(r"^\d{4}-\d{2}-\d{2} ", value, re.MULTILINE)


def test_validate_result_uses_clean_stdout() -> None:
    config_path = REPO_ROOT / "config/recipes/built-in/latest/mom-v1/config.yaml"

    result = _run_cli("validate", "--config", str(config_path))

    assert result.returncode == 0, result.stderr
    assert "✓ Configuration is valid" in result.stdout
    assert "Configuration summary" in result.stdout
    assert result.stderr == ""
    _assert_no_log_decoration(result.stdout)


def test_validate_failure_uses_clean_stderr(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("version: [\n", encoding="utf-8")

    result = _run_cli("validate", "--config", str(config_path), cwd=tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Error: Configuration parsing failed" in result.stderr
    _assert_no_log_decoration(result.stderr)


def test_root_help_redirect_has_no_ansi_or_log_decoration() -> None:
    result = CliRunner().invoke(main)

    assert result.exit_code == 0
    assert "vLLM Semantic Router" in result.stdout
    assert "\x1b[" not in result.stdout
    assert result.stderr == ""
    _assert_no_log_decoration(result.stdout)


class _DashboardBackend:
    def __init__(self, url: str | None) -> None:
        self.url = url

    def is_running(self) -> bool:
        return True

    def get_dashboard_url(self) -> str | None:
        return self.url


def test_dashboard_missing_url_is_a_clean_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime, "_build_backend", lambda *args, **kwargs: _DashboardBackend(None)
    )

    result = CliRunner().invoke(runtime.dashboard, ["--no-open"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: Dashboard URL could not be determined\n"


def test_dashboard_browser_failure_does_not_report_success(monkeypatch) -> None:
    url = "http://localhost:8700"
    monkeypatch.setattr(
        runtime, "_build_backend", lambda *args, **kwargs: _DashboardBackend(url)
    )
    monkeypatch.setattr(runtime.webbrowser, "open", lambda _url: False)

    result = CliRunner().invoke(runtime.dashboard)

    assert result.exit_code == 1
    assert "✓ Dashboard opened" not in result.stdout
    assert "Could not open a browser" in result.stderr
