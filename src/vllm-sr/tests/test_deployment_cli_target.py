"""Tests for CLI deployment-target routing."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.commands import runtime as rt  # noqa: E402
from cli.deployment_backend import resolve_target  # noqa: E402
from cli.main import main  # noqa: E402
from click.testing import CliRunner  # noqa: E402

# CLI integration — serve --target routes to correct backend


class TestCLITargetRouting:
    def test_serve_default_target_builds_docker_backend(self, monkeypatch):
        built = []

        class _FakeDocker:
            def deploy(self, **kw):
                pass

        def _fake_build(target, **kw):
            built.append(resolve_target(target))
            return _FakeDocker()

        monkeypatch.setattr(rt, "_build_backend", _fake_build)
        monkeypatch.setattr(
            rt,
            "ensure_bootstrap_workspace",
            lambda _: MagicMock(config_path=Path("/dev/null"), setup_mode=False),
        )
        monkeypatch.setattr(
            rt,
            "resolve_effective_config_path",
            lambda *a: Path("/dev/null"),
        )

        runner = CliRunner()
        runner.invoke(
            main,
            ["serve", "--config", "/dev/null", "--image-pull-policy", "never"],
        )

        assert built and built[0] == "docker"

    def test_stop_target_k8s_builds_k8s_backend(self, monkeypatch):
        built = []

        class _FakeK8s:
            def teardown(self):
                pass

        def _fake_build(target, **kw):
            built.append(resolve_target(target))
            return _FakeK8s()

        monkeypatch.setattr(rt, "_build_backend", _fake_build)

        runner = CliRunner()
        runner.invoke(main, ["stop", "--target", "k8s"])

        assert built and built[0] == "k8s"

    def test_stop_target_k8s_cleanup_failure_exits_nonzero(self, monkeypatch, caplog):
        class _FailingK8s:
            def teardown(self):
                raise RuntimeError("Runtime credential Secret cleanup was incomplete")

        monkeypatch.setattr(rt, "_build_backend", lambda *args, **kwargs: _FailingK8s())

        result = CliRunner().invoke(main, ["stop", "--target", "k8s"])

        assert result.exit_code == 1
        assert "cleanup was incomplete" in caplog.text
