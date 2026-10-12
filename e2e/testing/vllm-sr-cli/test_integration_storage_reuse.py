#!/usr/bin/env python3
"""
test_integration_storage_reuse.py - the stores' data across serve directories.

`vllm-sr stop` removes the managed stores and keeps their named data volumes.
A stack served again from another directory has no storage state there, so it
finds those volumes by asking the container runtime about the removed
containers. The unit suite fakes that answer; this module takes it from a real
daemon: the second directory must start, and its stores must mount the volumes
the first one left.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli.storage_secrets import POSTGRES_DATA_MOUNT_PATH, REDIS_DATA_MOUNT_PATH
from cli_test_base import CLITestBase
from serve_session import ServeSessionMixin

integration_only = unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)


class TestStorageAcrossServeDirectories(ServeSessionMixin, CLITestBase):
    """Real-daemon coverage for serving the stack again from a new directory."""

    SERVE_GATEWAY = "standalone"

    # Provisioning two stores, re-keying Postgres, and starting the runtime
    # takes longer than a serve that has no storage to bring up.
    STORAGE_STARTUP_TIMEOUT = 300

    def setUp(self):
        super().setUp()
        self.HEALTH_CHECK_TIMEOUT = self.STORAGE_STARTUP_TIMEOUT

    def tearDown(self):
        self.run_cli(["stop"], timeout=60)
        self._cleanup_container()
        super().tearDown()

    @integration_only
    def test_a_new_directory_after_stop_starts_on_the_volumes_stop_kept(self):
        """Serve, stop, then serve from a directory with no stack state."""
        self.print_test_header(
            "serve from a new directory after stop",
            "the stack starts again and its stores mount the volumes stop kept",
        )
        with self._running_serve(managed_storage=True):
            volumes = self._store_volumes()

        code, _stdout, stderr = self.run_cli(["stop"], timeout=120)
        self.assertEqual(code, 0, f"vllm-sr stop failed: {stderr}")
        for volume in volumes.values():
            self.assertTrue(
                self._volume_exists(volume), f"stop removed the data volume {volume}"
            )
        print("  ✓ stop removed the stores and kept their data volumes")

        self._serve_from_a_new_directory()
        with self._running_serve(managed_storage=True):
            self.assertEqual(self._store_volumes(), volumes)
        print("  ✓ the stores of the new directory mount the same volumes")
        self.print_test_result(True, "a new directory starts on the kept stores")

    def _store_volumes(self) -> dict[str, str]:
        return {
            "redis": self._data_volume(
                self.REDIS_CONTAINER_NAME, REDIS_DATA_MOUNT_PATH
            ),
            "postgres": self._data_volume(
                self.POSTGRES_CONTAINER_NAME, POSTGRES_DATA_MOUNT_PATH
            ),
        }

    def _data_volume(self, container_name: str, destination: str) -> str:
        code, mounts, stderr = self.inspect_container(
            "{{json .Mounts}}", container_name=container_name
        )
        self.assertEqual(code, 0, f"Could not inspect {container_name}: {stderr}")
        names = [
            mount.get("Name")
            for mount in json.loads(mounts) or []
            if mount.get("Type") == "volume" and mount.get("Destination") == destination
        ]
        self.assertEqual(
            len(names), 1, f"{container_name} has no data volume at {destination}"
        )
        return names[0]

    def _volume_exists(self, volume: str) -> bool:
        result = self._run_subprocess(
            [self.container_runtime, "volume", "inspect", volume], timeout=10
        )
        return result.returncode == 0

    def _serve_from_a_new_directory(self) -> None:
        """Continue the test in a directory with no stack state, as a second checkout."""
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)
        self.test_dir = str(
            Path(tempfile.mkdtemp(prefix="vllm-sr-cli-test-")).resolve()
        )
        environment = patch.dict(os.environ, {"VLLM_SR_STATE_ROOT_DIR": self.test_dir})
        environment.start()
        self.addCleanup(environment.stop)
        os.chdir(self.test_dir)
        print(f"  Serving again from: {self.test_dir}")


if __name__ == "__main__":
    unittest.main()
