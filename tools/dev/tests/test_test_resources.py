"""Exercise concurrent processes and cleanup against fake container commands."""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools/dev/with_test_resources.py"
SPEC = importlib.util.spec_from_file_location("test_resources", SCRIPT)
assert SPEC and SPEC.loader
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


class ResourceProcessTests(unittest.TestCase):
    def test_cancelled_lock_wait_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            name = f"test-{Path(directory).name}"
            locks = resources.ResourceLocks()
            locks.acquire(name, 0)
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--resource",
                    name,
                    "--",
                    sys.executable,
                    "-c",
                    "print('entered')",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                # The parent already owns the lease; the child cannot enter.
                self.assertIn("Waiting for test resource:", child.stderr.readline())
                child.send_signal(signal.SIGINT)
                stdout, _ = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 130)
                self.assertNotIn("entered", stdout)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                locks.close()

    def test_make_dry_run_does_not_allocate_a_stack(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--isolate-stack",
                "--",
                sys.executable,
                "-c",
                "print('dry-run')",
            ],
            env={**os.environ, "MAKEFLAGS": "n", "CONTAINER_RUNTIME": "/nonexistent"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "dry-run")

    def test_podman_only_host_preserves_cli_runtime_fallback(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(resources.shutil, "which", return_value=None),
        ):
            self.assertEqual(resources.container_runtime(), "podman")

    def test_nested_process_reuses_lease_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            resource = f"test-{Path(directory).name}"
            command = [
                sys.executable,
                str(SCRIPT),
                "--resource",
                resource,
                "--timeout",
                "0",
                "--",
            ]
            result = subprocess.run(
                command + command + [sys.executable, "-c", "raise SystemExit(7)"],
                check=False,
            )
            self.assertEqual(result.returncode, 7)
            result = subprocess.run(
                [*command, sys.executable, "-c", "pass"], check=False
            )
            self.assertEqual(result.returncode, 0)

    def test_concurrent_process_cannot_enter_another_runs_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            resource = f"test-{Path(directory).name}"
            command = [
                sys.executable,
                str(SCRIPT),
                "--resource",
                resource,
                "--timeout",
                "0",
                "--",
            ]
            holder = subprocess.Popen(
                [
                    *command,
                    sys.executable,
                    "-c",
                    "import sys; print('ready', flush=True); sys.stdin.read()",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(holder.stdout.readline().strip(), "ready")
                blocked = subprocess.run(
                    [*command, sys.executable, "-c", "print('entered')"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(blocked.returncode, 2)
                self.assertNotIn("entered", blocked.stdout)
            finally:
                holder.communicate("", timeout=10)

    def test_busy_listener_releases_partially_acquired_port_locks(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            locks = resources.ResourceLocks(Path(directory))
            with (
                mock.patch.object(resources, "STACK_PORTS", (port,)),
                self.assertRaises(RuntimeError),
            ):
                resources.reserve_stack_ports(locks, "0")
            self.assertEqual(locks.owned, set())

    def test_existing_stack_is_refused_without_destructive_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            log = Path(directory) / "calls"
            runtime.write_text(
                f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{log}'\necho occupied-vllm-sr-router-container\n"
            )
            runtime.chmod(0o755)
            with (
                mock.patch.dict(os.environ, {"CONTAINER_RUNTIME": str(runtime)}),
                self.assertRaisesRegex(RuntimeError, "existing test stack"),
            ):
                resources.require_empty_stack("occupied")
            self.assertEqual(log.read_text().strip(), "ps -a --format {{.Names}}")

    def test_milvus_cleanup_refuses_foreign_run_and_preserves_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            log = root / "calls"
            data = root / "data"
            data.mkdir()
            (data / "owned-by-a").write_text("keep")
            runtime.write_text(
                f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> \'{log}\'\necho \'{{"com.vllm.semantic-router.managed":"true","com.vllm.semantic-router.stack":"a","com.vllm.semantic-router.run":"run-a"}}\'\n'
            )
            runtime.chmod(0o755)
            result = subprocess.run(
                [
                    "make",
                    "-f",
                    "tools/make/milvus.mk",
                    "stop-milvus-unlocked",
                    "LOG_TARGET=true",
                    f"CONTAINER_RUNTIME={runtime}",
                    f"MILVUS_DATA_DIR={data}",
                    "MILVUS_STACK_NAME=b",
                    "MILVUS_RUN_ID=run-b",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ownership labels do not match", result.stderr)
            self.assertEqual((data / "owned-by-a").read_text(), "keep")
            self.assertNotIn("stop ", log.read_text())
            self.assertNotIn("rm ", log.read_text())

    def test_milvus_failed_removal_preserves_mounted_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            data = root / "data"
            data.mkdir()
            (data / "keep").write_text("still mounted")
            runtime.write_text(
                '#!/bin/sh\nif [ "$1" = rm ]; then exit 1; fi\n'
                'echo \'{"com.vllm.semantic-router.managed":"true",'
                '"com.vllm.semantic-router.stack":"owned",'
                '"com.vllm.semantic-router.run":"run"}\'\n'
            )
            runtime.chmod(0o755)
            result = subprocess.run(
                [
                    "make",
                    "-f",
                    "tools/make/milvus.mk",
                    "stop-milvus-unlocked",
                    "LOG_TARGET=true",
                    f"CONTAINER_RUNTIME={runtime}",
                    f"MILVUS_DATA_DIR={data}",
                    "MILVUS_STACK_NAME=owned",
                    "MILVUS_RUN_ID=run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((data / "keep").read_text(), "still mounted")


if __name__ == "__main__":
    unittest.main()
