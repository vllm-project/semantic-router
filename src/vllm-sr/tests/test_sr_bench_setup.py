"""Pinned harness setup must provide runtime data before any paid call."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from cli.sr_bench import external, setup


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "numpy",
        "scipy.linalg",
        "sympy",
        "h5py",
        "datasets",
        "tqdm",
        "mpl_toolkits.mplot3d",
    ],
)
def test_grading_image_checks_dependencies_before_accepting_build(
    tmp_path, monkeypatch, missing
):
    monkeypatch.setenv("SR_BENCH_HOME", str(tmp_path))
    base = "python@sha256:" + "a" * 64
    image = "sha256:" + "b" * 64
    directory = tmp_path / "sandbox"
    commands, imports = [], []

    def inspect(args, **_kwargs):
        assert args == ["docker", "image", "inspect", "python:3.12-slim-bookworm"]
        return json.dumps([{"RepoDigests": [base]}])

    def import_dependency(name, *_args, **_kwargs):
        imports.append(name)
        if name == missing:
            raise ImportError(f"Unavailable grading dependency: {name}")
        return SimpleNamespace(Axes3D=object())

    def run(args, **kwargs):
        assert kwargs["check"] is True
        commands.append(args)
        if args[1] == "build":
            lines = (directory / "Dockerfile").read_text().splitlines()
            assert lines[0] == f"FROM {base}"
            assert "matplotlib==3.9.4" in lines[1].split()
            check = json.loads(lines[2].removeprefix("RUN "))
            assert check[:2] == ["python", "-c"]
            # Run only the builder's fixed import check; no packages or cases
            # are loaded, and Docker is entirely mocked for this contract test.
            try:
                exec(check[2], {"__builtins__": {"__import__": import_dependency}})
            except ImportError as exc:
                raise subprocess.CalledProcessError(1, args) from exc
            Path(args[args.index("--iidfile") + 1]).write_text(image + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(setup.subprocess, "check_output", inspect)
    monkeypatch.setattr(setup.subprocess, "run", run)
    if missing:
        with pytest.raises(ValueError, match="Pinned dependency setup failed"):
            setup.build_grading_image()
        assert missing in imports
        assert not (directory / "manifest.json").exists()
        assert not (directory / "image-id").exists()
    else:
        receipt = setup.build_grading_image()
        assert imports == [
            "numpy",
            "scipy.linalg",
            "sympy",
            "h5py",
            "datasets",
            "tqdm",
            "mpl_toolkits.mplot3d",
        ]
        assert receipt["sandbox_image"] == image
        assert receipt["base_image"] == base
        assert (
            receipt["dockerfile_sha256"]
            == hashlib.sha256((directory / "Dockerfile").read_bytes()).hexdigest()
        )
        assert json.loads((directory / "manifest.json").read_text()) == receipt
    assert [command[:2] for command in commands] == [
        ["docker", "pull"],
        ["docker", "build"],
    ]


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
