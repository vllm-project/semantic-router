"""Exercise the local tag helper in an isolated Git repository."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_SCRIPT = REPO_ROOT / "src" / "vllm-sr" / "scripts" / "release.sh"


def run(root: Path, *command: str) -> str:
    result = subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.mark.parametrize("current_version", ["0.4.0", "0.3.9"])
def test_release_tags_the_release_version_and_starts_the_next_cycle(
    tmp_path: Path, current_version: str
) -> None:
    script = tmp_path / "src" / "vllm-sr" / "scripts" / "release.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(RELEASE_SCRIPT, script)

    pyproject = tmp_path / "src" / "vllm-sr" / "pyproject.toml"
    pyproject.write_text(
        f'[project]\nversion = "{current_version}"\n'
        "[project.optional-dependencies]\n"
        f'runtime = ["vllm-srun[multimodal]=={current_version}"]\n',
        encoding="utf-8",
    )
    runtime = tmp_path / "src" / "model-runtime" / "pyproject.toml"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(
        f'[project]\nversion = "{current_version}"\n'
        '[tool.mypy]\npython_version = "3.10"\n',
        encoding="utf-8",
    )
    checker = tmp_path / "tools" / "release" / "check_version_contract.py"
    checker.parent.mkdir(parents=True)
    checker.write_text(
        'import sys\nassert sys.argv[-2:] == ["--version", "0.4.0"]\n',
        encoding="utf-8",
    )

    run(tmp_path, "git", "init", "-q")
    run(tmp_path, "git", "config", "user.name", "Release Test")
    run(tmp_path, "git", "config", "user.email", "release-test@example.invalid")
    run(tmp_path, "git", "config", "commit.gpgsign", "false")
    run(tmp_path, "git", "add", ".")
    run(tmp_path, "git", "commit", "-qm", "fixture")

    run(tmp_path, "bash", str(script), "0.4.0", "0.5.0")

    for ref, version in (("v0.4.0", "0.4.0"), ("HEAD", "0.5.0")):
        cli = run(tmp_path, "git", "show", f"{ref}:src/vllm-sr/pyproject.toml")
        assert f'version = "{version}"' in cli
        assert f'runtime = ["vllm-srun[multimodal]=={version}"]' in cli
        assert (
            run(tmp_path, "git", "show", f"{ref}:src/model-runtime/pyproject.toml")
            == f'[project]\nversion = "{version}"\n[tool.mypy]\npython_version = "3.10"'
        )
    assert run(tmp_path, "git", "status", "--porcelain") == ""
