"""Exercise the local tag helper in an isolated Git repository."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_SCRIPT = REPO_ROOT / "src" / "vllm-sr" / "scripts" / "release.sh"
PYPROJECT = "src/vllm-sr/pyproject.toml"
CHART = "deploy/helm/semantic-router/Chart.yaml"

# Records each contract check with the tree it ran on; fails when told to.
FAKE_CHECKER = """\
import json, os, sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
record = {
    "argv": sys.argv[1:],
    "pyproject": (root / "src/vllm-sr/pyproject.toml").read_text(),
    "chart": (root / "deploy/helm/semantic-router/Chart.yaml").read_text(),
}
with open(os.environ["RELEASE_TEST_CHECKS"], "a") as log:
    log.write(json.dumps(record) + "\\n")
mode = "release" if sys.argv[1:2] == ["--version"] else "development"
sys.exit(1 if os.environ.get("RELEASE_TEST_FAIL") == mode else 0)
"""


def run(root: Path, *command: str) -> str:
    result = subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def release_fixture(tmp_path: Path, current_version: str) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    script = root / "src" / "vllm-sr" / "scripts" / "release.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(RELEASE_SCRIPT, script)
    (root / PYPROJECT).write_text(
        f'[project]\nversion = "{current_version}"\n', encoding="utf-8"
    )
    chart = root / CHART
    chart.parent.mkdir(parents=True)
    chart.write_text(
        'apiVersion: v2\nversion: 0.2.0\n# appVersion is the image tag.\nappVersion: "latest"\n',
        encoding="utf-8",
    )
    checker = root / "tools" / "release" / "check_version_contract.py"
    checker.parent.mkdir(parents=True)
    checker.write_text(FAKE_CHECKER, encoding="utf-8")

    run(root, "git", "init", "-q")
    run(root, "git", "config", "user.name", "Release Test")
    run(root, "git", "config", "user.email", "release-test@example.invalid")
    run(root, "git", "config", "commit.gpgsign", "false")
    run(root, "git", "add", ".")
    run(root, "git", "commit", "-qm", "fixture")
    return root, tmp_path / "checks.jsonl"


def release(root: Path, checks: Path, *versions: str, fail: str = ""):
    env = {**os.environ, "RELEASE_TEST_CHECKS": str(checks), "RELEASE_TEST_FAIL": fail}
    return subprocess.run(
        ["bash", str(root / "src/vllm-sr/scripts/release.sh"), *versions],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def checks_of(checks: Path) -> list[dict]:
    if not checks.exists():
        return []
    return [json.loads(line) for line in checks.read_text().splitlines()]


@pytest.mark.parametrize("current_version", ["0.5.0", "0.4.9"])
def test_release_pins_the_chart_on_the_tag_and_restores_latest_for_the_next_cycle(
    tmp_path: Path, current_version: str
) -> None:
    root, checks = release_fixture(tmp_path, current_version)
    start = run(root, "git", "rev-parse", "HEAD")

    result = release(root, checks, "0.5.0", "0.6.0")
    assert result.returncode == 0, result.stderr

    assert 'version = "0.5.0"' in run(root, "git", "show", f"v0.5.0:{PYPROJECT}")
    assert 'appVersion: "v0.5.0"' in run(root, "git", "show", f"v0.5.0:{CHART}")
    assert 'version = "0.6.0"' in run(root, "git", "show", f"HEAD:{PYPROJECT}")
    head_chart = run(root, "git", "show", f"HEAD:{CHART}")
    assert 'appVersion: "latest"' in head_chart
    assert "# appVersion is the image tag." in head_chart
    assert run(root, "git", "rev-list", "--count", f"{start}..v0.5.0") == "1"
    assert run(root, "git", "rev-parse", "HEAD~1") == run(
        root, "git", "rev-parse", "v0.5.0^{commit}"
    )
    assert run(root, "git", "status", "--porcelain") == ""

    release_check, cycle_check = checks_of(checks)
    assert release_check["argv"] == ["--version", "0.5.0"]
    assert 'version = "0.5.0"' in release_check["pyproject"]
    assert 'appVersion: "v0.5.0"' in release_check["chart"]
    assert cycle_check["argv"] == []
    assert 'version = "0.6.0"' in cycle_check["pyproject"]
    assert 'appVersion: "latest"' in cycle_check["chart"]


def test_a_failed_release_check_creates_no_tag(tmp_path: Path) -> None:
    root, checks = release_fixture(tmp_path, "0.5.0")

    result = release(root, checks, "0.5.0", "0.6.0", fail="release")

    assert result.returncode != 0
    assert "nothing is tagged" in result.stderr
    assert run(root, "git", "tag", "--list") == ""
    assert [check["argv"] for check in checks_of(checks)] == [["--version", "0.5.0"]]


def test_a_failed_development_cycle_check_prints_a_working_undo(
    tmp_path: Path,
) -> None:
    root, checks = release_fixture(tmp_path, "0.5.0")
    start = run(root, "git", "rev-parse", "HEAD")

    result = release(root, checks, "0.5.0", "0.6.0", fail="development")

    assert result.returncode != 0
    assert run(root, "git", "tag", "--list") == "v0.5.0"
    undo = result.stderr.split("Undo with: ", 1)[1].strip()
    assert undo.startswith("git tag -d v0.5.0 && git reset --hard ")
    subprocess.run(["bash", "-c", undo], cwd=root, check=True, capture_output=True)
    assert run(root, "git", "rev-parse", "HEAD") == start
    assert run(root, "git", "tag", "--list") == ""


@pytest.mark.parametrize("next_version", ["0.4.9", "0.5.0"])
def test_the_next_cycle_must_sort_after_the_release(
    tmp_path: Path, next_version: str
) -> None:
    root, checks = release_fixture(tmp_path, "0.5.0")
    start = run(root, "git", "rev-parse", "HEAD")

    result = release(root, checks, "0.5.0", next_version)

    assert result.returncode != 0
    assert run(root, "git", "rev-parse", "HEAD") == start
    assert run(root, "git", "tag", "--list") == ""
    assert checks_of(checks) == []
