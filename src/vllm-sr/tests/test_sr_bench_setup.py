"""Pinned harness setup must provide runtime data before any paid call."""

import subprocess
from types import SimpleNamespace

import pytest
from cli.sr_bench import external, setup


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_explicit_setup_hydrates_missing_sparse_assets_without_changing_pin(tmp_path):
    root = tmp_path / "tau3"
    root.mkdir()
    _git(root, "init")
    tracked = [
        "src/tau2/__init__.py",
        "data/tau2/domains/airline/tasks.json",
        *setup.TAU3_TEXT_ASSETS,
    ]
    for name in tracked:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen fixture\n")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=sr-bench test",
        "-c",
        "user.email=sr-bench@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Fixture",
    )
    revision = _git(root, "rev-parse", "HEAD")
    _git(root, "sparse-checkout", "set", "src", "data/tau2/domains/airline")
    assert setup.missing_tau3_text_assets(root) == list(setup.TAU3_TEXT_ASSETS)
    environment = root / ".venv" / "unchanged"
    environment.parent.mkdir()
    environment.write_text("existing environment")
    spec = {**setup.PACKAGES["tau3"], "revision": revision}

    setup._checkout(spec, root)

    setup.validate_tau3_text_assets(root)
    assert _git(root, "rev-parse", "HEAD") == revision
    assert _git(root, "diff", "HEAD") == ""
    assert environment.read_text() == "existing environment"
    assert all((root / name).read_text() == "frozen fixture\n" for name in tracked)

    (root / tracked[0]).write_text("operator change\n")
    with pytest.raises(ValueError, match="differs from its pinned version"):
        setup._checkout(spec, root)
    assert (root / tracked[0]).read_text() == "operator change\n"


@pytest.mark.parametrize("missing_asset", setup.TAU3_TEXT_ASSETS)
@pytest.mark.parametrize("empty", [False, True])
def test_tau3_preflight_rejects_missing_simulator_data_before_runtime(
    tmp_path, monkeypatch, missing_asset, empty
):
    for name in setup.TAU3_TEXT_ASSETS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("simulator prompt\n")
    missing = tmp_path / missing_asset
    if empty:
        missing.write_text("")
    else:
        missing.unlink()
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setattr(
        external, "harness_paths", lambda _name: (tmp_path, interpreter)
    )
    revision = "a" * 40
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=revision + "\n")

    monkeypatch.setattr(external.subprocess, "run", run)
    with pytest.raises(ValueError, match="tau3 simulator data is missing or empty"):
        external._preflight_harness("tau3", {"source_revision": revision})
    assert all(command[0] == "git" for command in commands)
    assert not any(str(interpreter) in command for command in commands)


def test_setup_inspection_reports_missing_data_without_installing(
    tmp_path, monkeypatch
):
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setattr(setup, "harness_paths", lambda _name: (tmp_path, interpreter))
    monkeypatch.setenv("SR_BENCH_HOME", str(tmp_path))

    receipt = setup.setup(benchmark="tau3", install=False)["benchmarks"][0]

    assert receipt["installed"] is True
    assert receipt["runtime_assets_available"] is False
    assert receipt["missing_runtime_assets"] == list(setup.TAU3_TEXT_ASSETS)
    assert not (tmp_path / "data").exists()
