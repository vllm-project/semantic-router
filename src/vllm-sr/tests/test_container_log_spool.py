import json
import os
import select
import shutil
import signal
import stat
import subprocess
import sys
from pathlib import Path

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


def _spool_command(tmp_path, *service_command):
    spool_file = tmp_path / "current.log"
    spool_file.touch(mode=0o600)
    # Substitute only the fixed mount and temporary path roots; execute the
    # deployed shell unchanged within this test's private filesystem scope.
    script = BOUNDED_LOG_SPOOL_SCRIPT.replace(
        "/var/log/vllm-sr-producer/*.log", f"{tmp_path}/*.log", 1
    ).replace("/tmp/vllm-sr-log-spool", f"{tmp_path}/vllm-sr-log-spool")
    return [
        "/bin/sh",
        "-c",
        script,
        "vllm-sr-log-spool-test",
        str(spool_file),
        "1024",
        "64",
        "--",
        *service_command,
    ]


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
    service_script = (
        "i=0; while [ $i -lt 40 ]; do "
        "printf 'entry-%03d-%0200d\\n' $i 0; i=$((i + 1)); done"
    )

    result = subprocess.run(
        _spool_command(tmp_path, "/bin/sh", "-c", service_script),
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


def test_bounded_log_spool_starts_with_same_pid_fifo_left_by_previous_run(tmp_path):
    previous_directory = tmp_path / "vllm-sr-log-spool.previous"
    previous_directory.mkdir(mode=0o700)
    sentinel = previous_directory / "sentinel"
    sentinel.write_text("another launch owns this directory\n")
    # exec preserves the PID, reproducing a restarted container's old $$ names
    # without requiring a container or relying on OS PID reuse in the test.
    bootstrap = r"""
mkfifo "$1/vllm-sr-log-spool.$$.fifo"
printf 'old trim sentinel\n' > "$1/vllm-sr-log-spool-trim.$$"
printf '%s\n' "$$" > "$1/previous-pid"
shift
exec "$@"
"""
    result = subprocess.run(
        [
            "/bin/sh",
            "-ec",
            bootstrap,
            "previous-spool-test",
            str(tmp_path),
            *_spool_command(tmp_path, "/bin/sh", "-c", "printf 'service-resumed\\n'"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "service-resumed\n"
    assert (tmp_path / "current.log").read_text() == "service-resumed\n"
    previous_pid = (tmp_path / "previous-pid").read_text().strip()
    old_fifo = tmp_path / f"vllm-sr-log-spool.{previous_pid}.fifo"
    assert stat.S_ISFIFO(old_fifo.stat().st_mode)
    old_trim = tmp_path / f"vllm-sr-log-spool-trim.{previous_pid}"
    assert old_trim.read_text() == "old trim sentinel\n"
    assert sentinel.read_text() == "another launch owns this directory\n"
    assert set(tmp_path.glob("vllm-sr-log-spool.*")) == {
        old_fifo,
        previous_directory,
    }


@pytest.mark.parametrize("service_status", [0, 17])
def test_bounded_log_spool_cleans_its_private_directory(tmp_path, service_status):
    metadata_file = tmp_path / "service-metadata.json"
    service_script = """
import json, stat, sys
from pathlib import Path
directory = next(path for path in Path(sys.argv[1]).glob("vllm-sr-log-spool.*")
                 if path.is_dir())
metadata = {"directory": str(directory),
            "mode": stat.S_IMODE(directory.stat().st_mode),
            "fifos": sum(stat.S_ISFIFO(path.stat().st_mode)
                         for path in directory.iterdir())}
Path(sys.argv[2]).write_text(json.dumps(metadata))
print("service-stdout")
print("service-stderr", file=sys.stderr)
sys.exit(int(sys.argv[3]))
"""
    result = subprocess.run(
        _spool_command(
            tmp_path,
            sys.executable,
            "-c",
            service_script,
            str(tmp_path),
            str(metadata_file),
            str(service_status),
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == service_status, result.stderr
    assert set(result.stdout.splitlines()) == {"service-stdout", "service-stderr"}
    assert (tmp_path / "current.log").read_text() == result.stdout
    metadata = json.loads(metadata_file.read_text())
    assert metadata["mode"] == 0o700
    assert metadata["fifos"] == 1
    assert not Path(metadata["directory"]).exists()


@pytest.mark.parametrize("service_umask", [0o022, 0o002, 0o027])
def test_bounded_log_spool_preserves_service_umask(tmp_path, service_umask):
    service_file = tmp_path / "service-created"
    service_script = """
import json, os, stat, sys
from pathlib import Path
inherited = os.umask(0)
os.umask(inherited)
Path(sys.argv[2]).touch(mode=0o666)
directories = [stat.S_IMODE(path.stat().st_mode)
               for path in Path(sys.argv[1]).glob("vllm-sr-log-spool.*")
               if path.is_dir()]
print(json.dumps({"umask": inherited, "directory_modes": directories}))
"""
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            'umask "$1"; shift; exec "$@"',
            "service-umask-test",
            f"{service_umask:03o}",
            *_spool_command(
                tmp_path,
                sys.executable,
                "-c",
                service_script,
                str(tmp_path),
                str(service_file),
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    actual_mode = stat.S_IMODE(service_file.stat().st_mode)
    expected_mode = 0o666 & ~service_umask
    assert actual_mode == expected_mode, f"{actual_mode:o} != {expected_mode:o}"
    metadata = json.loads(result.stdout)
    assert metadata["umask"] == service_umask
    assert metadata["directory_modes"] == [0o700]
    assert not list(tmp_path.glob("vllm-sr-log-spool.*"))


def test_bounded_log_spool_forwards_term_and_cleans_private_directory(tmp_path):
    service_script = """
import signal, sys
def stopped(signum, frame):
    print("service-stopped", flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, stopped)
print("service-ready", flush=True)
signal.pause()
"""
    process = subprocess.Popen(
        _spool_command(tmp_path, sys.executable, "-c", service_script),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert select.select([process.stdout], [], [], 5)[0], "service did not start"
        assert process.stdout.readline() == "service-ready\n"
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 128 + signal.SIGTERM, stderr
        assert stdout == "service-stopped\n"
        assert not list(tmp_path.glob("vllm-sr-log-spool.*"))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


def test_rootless_podman_retains_host_spool_group(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(sys, "platform", "linux")
    command = ["podman", "run"]

    append_supplemental_gids(command, [1000], "podman")

    assert command[-2:] == ["--group-add", "keep-groups"]


def test_rootless_podman_on_macos_uses_numeric_group_add(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(sys, "platform", "darwin")
    command = ["podman", "run"]

    append_supplemental_gids(command, [1000], "podman")

    assert command[-2:] == ["--group-add", "1000"]


def test_docker_uses_fixed_numeric_spool_group():
    command = ["docker", "run"]

    append_supplemental_gids(command, [1000, 1000], "docker")

    assert command[-2:] == ["--group-add", "1000"]


@pytest.mark.parametrize("missing_tool", ["awk", "mktemp"])
def test_bounded_log_spool_fails_before_service_when_tool_is_missing(
    tmp_path, missing_tool
):
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    for tool in LOG_SPOOL_REQUIRED_TOOLS:
        if tool == missing_tool:
            continue
        resolved = shutil.which(tool)
        assert resolved is not None
        (tool_dir / tool).symlink_to(resolved)
    service_marker = tmp_path / "service-started"

    result = subprocess.run(
        _spool_command(
            tmp_path,
            "/bin/sh",
            "-c",
            ': > "$1"',
            "service",
            str(service_marker),
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PATH": str(tool_dir)},
    )

    assert result.returncode == 69
    assert f"required runtime tool is unavailable: {missing_tool}" in result.stderr
    assert not service_marker.exists()
