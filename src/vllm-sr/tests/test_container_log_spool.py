import os
import shutil
import stat
import subprocess

import pytest
from cli.container_log_spool import (
    BOUNDED_LOG_SPOOL_SCRIPT,
    LOG_SPOOL_COMPONENTS,
    LOG_SPOOL_MAX_BYTES,
    LOG_SPOOL_MAX_LINE_BYTES,
    LOG_SPOOL_PRODUCER_FILE,
    LOG_SPOOL_REQUIRED_TOOLS,
    bounded_log_spool_entrypoint,
    prepare_runtime_log_spool,
)
from cli.container_run_command import append_supplemental_gids


def test_prepare_runtime_log_spool_creates_fixed_private_regular_files(tmp_path):
    (tmp_path / ".vllm-sr").mkdir()
    spool = prepare_runtime_log_spool(str(tmp_path / ".vllm-sr"), "team-a")

    assert spool.root == tmp_path / ".vllm-sr" / "logs" / "team-a"
    assert stat.S_IMODE(spool.root.stat().st_mode) == 0o750
    assert {path.name for path in spool.root.iterdir()} == {
        f"{component}.log" for component in LOG_SPOOL_COMPONENTS
    }
    for component in LOG_SPOOL_COMPONENTS:
        path = spool.host_file(component)
        assert path.is_file()
        assert not path.is_symlink()
        expected_mode = 0o660 if component == "envoy" else 0o640
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode


def test_prepare_runtime_log_spool_rejects_symlinked_component(tmp_path):
    stack_root = tmp_path / ".vllm-sr" / "logs" / "team-a"
    stack_root.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_text("must remain outside\n", encoding="utf-8")
    (stack_root / "router.log").symlink_to(outside)

    with pytest.raises(ValueError, match="unavailable or unsafe"):
        prepare_runtime_log_spool(str(tmp_path / ".vllm-sr"), "team-a")

    assert outside.read_text(encoding="utf-8") == "must remain outside\n"


def test_bounded_log_spool_entrypoint_uses_fixed_limits_and_path():
    entrypoint, args = bounded_log_spool_entrypoint("/service", ["--flag"])

    assert entrypoint == "/bin/sh"
    assert args[0:4] == [
        "-c",
        BOUNDED_LOG_SPOOL_SCRIPT,
        "vllm-sr-log-spool",
        LOG_SPOOL_PRODUCER_FILE,
    ]
    assert args[4:6] == [str(LOG_SPOOL_MAX_BYTES), str(LOG_SPOOL_MAX_LINE_BYTES)]
    assert args[-3:] == ["--", "/service", "--flag"]


def test_bounded_log_spool_relay_preserves_output_and_bounds_file(tmp_path):
    spool_file = tmp_path / "current.log"
    spool_file.touch(mode=0o600)
    # The deployed wrapper accepts only its fixed producer mount. Substitute
    # that case pattern only for this isolated shell-level behavior test.
    script = BOUNDED_LOG_SPOOL_SCRIPT.replace(
        "/var/log/vllm-sr-producer/*.log",
        f"{tmp_path}/*.log",
        1,
    )
    service_script = (
        "i=0; while [ $i -lt 40 ]; do "
        "printf 'entry-%03d-%0200d\\n' $i 0; i=$((i + 1)); done"
    )

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            script,
            "vllm-sr-log-spool-test",
            str(spool_file),
            "1024",
            "64",
            "--",
            "/bin/sh",
            "-c",
            service_script,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "entry-000" in result.stdout
    assert "entry-039" in result.stdout
    assert os.path.getsize(spool_file) <= 1024
    retained = spool_file.read_text(encoding="utf-8")
    assert "entry-039" in retained
    assert "[log record truncated]" in retained


def test_rootless_podman_retains_host_spool_group(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    command = ["podman", "run"]

    append_supplemental_gids(command, [1000], "podman")

    assert command[-2:] == ["--group-add", "keep-groups"]


def test_docker_uses_fixed_numeric_spool_group():
    command = ["docker", "run"]

    append_supplemental_gids(command, [1000, 1000], "docker")

    assert command[-2:] == ["--group-add", "1000"]


def test_bounded_log_spool_fails_before_service_when_tool_is_missing(tmp_path):
    spool_file = tmp_path / "current.log"
    spool_file.touch(mode=0o600)
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    for tool in LOG_SPOOL_REQUIRED_TOOLS:
        if tool == "awk":
            continue
        resolved = shutil.which(tool)
        assert resolved is not None
        (tool_dir / tool).symlink_to(resolved)
    script = BOUNDED_LOG_SPOOL_SCRIPT.replace(
        "/var/log/vllm-sr-producer/*.log",
        f"{tmp_path}/*.log",
        1,
    )
    service_marker = tmp_path / "service-started"

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            script,
            "vllm-sr-log-spool-test",
            str(spool_file),
            "1024",
            "64",
            "--",
            "/bin/sh",
            "-c",
            f"touch {service_marker}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PATH": str(tool_dir)},
    )

    assert result.returncode == 69
    assert "required runtime tool is unavailable: awk" in result.stderr
    assert not service_marker.exists()
