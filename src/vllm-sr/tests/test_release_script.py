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


@pytest.mark.parametrize(
    ("current_candle", "tagged_candle"),
    [("0.4.1", "0.4.1"), ("0.3.7", "0.4.0")],
)
def test_release_tag_preserves_independent_candle_patch(
    tmp_path: Path, current_candle: str, tagged_candle: str
) -> None:
    script = tmp_path / "src" / "vllm-sr" / "scripts" / "release.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(RELEASE_SCRIPT, script)

    pyproject = tmp_path / "src" / "vllm-sr" / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.4.0"\n', encoding="utf-8")
    cargo = tmp_path / "candle-binding" / "Cargo.toml"
    cargo.parent.mkdir()
    cargo.write_text(
        f'[package]\nname = "candle-semantic-router"\nversion = "{current_candle}"\n',
        encoding="utf-8",
    )
    lock = tmp_path / "candle-binding" / "Cargo.lock"
    lock.write_text(
        f'[[package]]\nname = "candle-semantic-router"\nversion = "{current_candle}"\n',
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

    assert f'version = "{tagged_candle}"' in run(
        tmp_path, "git", "show", "v0.4.0:candle-binding/Cargo.toml"
    )
    assert f'version = "{tagged_candle}"' in run(
        tmp_path, "git", "show", "v0.4.0:candle-binding/Cargo.lock"
    )
    assert 'version = "0.4.0"' in run(
        tmp_path, "git", "show", "v0.4.0:src/vllm-sr/pyproject.toml"
    )
    assert 'version = "0.5.0"' in run(
        tmp_path, "git", "show", "HEAD:src/vllm-sr/pyproject.toml"
    )
    assert run(tmp_path, "git", "status", "--porcelain") == ""
