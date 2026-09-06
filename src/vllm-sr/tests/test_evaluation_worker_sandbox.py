from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_probe(
    tmp_path: Path,
    body: str,
) -> subprocess.CompletedProcess[str]:
    work = tmp_path / "work"
    suites = tmp_path / "suites"
    work.mkdir(mode=0o700)
    suites.mkdir(mode=0o700)
    suite_object = suites / "private-suite-object"
    suite_object.write_text("private normalized evidence", encoding="utf-8")
    suite_object.chmod(0o600)
    package_root = Path(__file__).resolve().parents[1]
    script = f"""
from pathlib import Path
import ctypes
import errno
import os
import socket
import struct
import sys
from cli.evaluation.sandbox import WorkerSandboxPolicy, apply_worker_sandbox
root = Path(sys.argv[1]).resolve()
raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
raw_descriptor = raw_socket.detach()
pipe_read, pipe_write = os.pipe()
apply_worker_sandbox(WorkerSandboxPolicy(
    writable_root=root / "work",
    suite_store=root / "suites",
    readable_roots=(Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(), Path(sys.argv[2]).resolve()),
    cpu_seconds=30,
))
{body}
"""
    environment = {
        "PYTHONPATH": str(package_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(package_root)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )


def test_worker_sandbox_allows_only_private_output_tree(tmp_path: Path) -> None:
    secret = tmp_path.parent / f"{tmp_path.name}-dashboard-secret"
    secret.write_text("must-not-reach-worker", encoding="utf-8")
    result = _run_probe(
        tmp_path,
        f"""
(root / "work" / "evidence.txt").write_text("ok", encoding="utf-8")
try:
    Path({str(secret)!r}).read_text(encoding="utf-8")
except PermissionError:
    pass
else:
    raise AssertionError("worker escaped its filesystem policy")
""",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "work" / "evidence.txt").read_text() == "ok"


def test_worker_sandbox_denies_network_process_and_unix_socket_escape(
    tmp_path: Path,
) -> None:
    result = _run_probe(
        tmp_path,
        """
for operation in (
    lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM),
    lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM),
    lambda: os.execve(sys.executable, [sys.executable, "-c", "pass"], {}),
):
    try:
        operation()
    except PermissionError:
        continue
    raise AssertionError("worker escaped its syscall or TCP policy")

address = struct.pack("=H", socket.AF_INET) + struct.pack("!H", 1) + socket.inet_aton("127.0.0.1") + bytes(8)
buffer = ctypes.create_string_buffer(address)
libc = ctypes.CDLL(None, use_errno=True)
if libc.connect(raw_descriptor, ctypes.byref(buffer), len(address)) != -1 or ctypes.get_errno() != errno.EPERM:
    raise AssertionError("raw libc connect escaped the worker seccomp policy")
os.close(raw_descriptor)
""",
    )
    assert result.returncode == 0, result.stderr


def test_worker_sandbox_preserves_inherited_broker_pipes_and_threads(
    tmp_path: Path,
) -> None:
    result = _run_probe(
        tmp_path,
        """
from concurrent.futures import ThreadPoolExecutor
os.write(pipe_write, b"broker")
if os.read(pipe_read, 6) != b"broker":
    raise AssertionError("inherited broker pipe stopped working")
with ThreadPoolExecutor(max_workers=2) as pool:
    if tuple(pool.map(lambda value: value * 2, (2, 3))) != (4, 6):
        raise AssertionError("worker thread execution stopped working")
""",
    )
    assert result.returncode == 0, result.stderr


def test_worker_sandbox_cannot_mutate_read_only_suite_metadata(tmp_path: Path) -> None:
    suite_object = tmp_path / "suites" / "private-suite-object"
    result = _run_probe(
        tmp_path,
        """
suite_object = root / "suites" / "private-suite-object"
descriptor = os.open(suite_object, os.O_RDONLY)
try:
    operations = (
        lambda: os.fchmod(descriptor, 0o644),
        lambda: os.utime(suite_object, ns=(1, 1)),
        lambda: os.setxattr(suite_object, b"user.evaluation-escape", b"set"),
    )
    for operation in operations:
        try:
            operation()
        except PermissionError:
            continue
        raise AssertionError("worker mutated read-only suite metadata")
finally:
    os.close(descriptor)
""",
    )
    assert result.returncode == 0, result.stderr
    assert suite_object.stat().st_mode & 0o777 == 0o600
    assert suite_object.stat().st_mtime_ns != 1
    assert "user.evaluation-escape" not in os.listxattr(suite_object)
