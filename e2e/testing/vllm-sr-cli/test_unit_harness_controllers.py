"""A test stack's instance controller, and the state serve keeps for it, never
outlive its tests.

`serve` starts the controller in a session of its own, so removing a stack's
containers leaves it running; the harness stops it as `vllm-sr stop` does.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cli.runtime_stack import DEFAULT_STACK_NAME
from cli_test_base import CLITestBase


class TestStackControllers(CLITestBase):
    def _stand_in_controller(self, stack_name: str):
        """A process that lists in ps as the controller of *stack_name*."""
        directory = Path(tempfile.mkdtemp(prefix="vllm-sr-cli-test-controller-"))
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        manifest = {"layout": {"stack_name": stack_name}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(120)",
                "-m",
                "cli.instance_controller",
                "--directory",
                str(directory),
            ],
            start_new_session=True,
        )
        self.addCleanup(process.wait, timeout=30)
        self.addCleanup(process.kill)
        return process, directory

    def test_cleanup_stops_this_stacks_controllers_only(self):
        ours, our_directory = self._stand_in_controller(self.runtime_stack.stack_name)
        other, other_directory = self._stand_in_controller(
            f"{self.runtime_stack.stack_name}-another"
        )
        self.assertEqual([pid for pid, _ in self._stack_controllers()], [ours.pid])

        self._cleanup_container()

        self.assertIsNotNone(ours.wait(timeout=30))
        self.assertFalse(our_directory.exists())
        self.assertIsNone(other.poll())
        self.assertTrue(other_directory.exists())

    def test_cleanup_removes_the_state_serve_kept_for_this_stack_only(self):
        state_home = Path(tempfile.mkdtemp(prefix="vllm-sr-cli-test-state-"))
        self.addCleanup(shutil.rmtree, state_home, ignore_errors=True)
        environment = mock.patch.dict(os.environ, {"XDG_STATE_HOME": str(state_home)})
        environment.start()
        self.addCleanup(environment.stop)
        instances = state_home / "vllm-sr" / "instances"
        for key, stack_name in (
            ("ours", self.runtime_stack.stack_name),
            ("theirs", f"{self.runtime_stack.stack_name}-another"),
        ):
            (instances / key).mkdir(parents=True)
            manifest = {"layout": {"stack_name": stack_name}}
            (instances / key / "manifest.json").write_text(json.dumps(manifest))

        # `vllm-sr stop` stopped the controller and kept its state.
        self._cleanup_container()

        self.assertEqual(sorted(path.name for path in instances.iterdir()), ["theirs"])

    def test_a_class_without_its_own_test_stack_touches_no_controller(self):
        default, directory = self._stand_in_controller(DEFAULT_STACK_NAME)

        self.assertEqual(CLITestBase._stack_controllers(), [])
        CLITestBase._stop_stack_controllers()

        self.assertIsNone(default.poll())
        self.assertTrue(directory.exists())

    def test_a_controller_that_outlives_the_tests_fails_the_class(self):
        leaked, _directory = self._stand_in_controller(self.runtime_stack.stack_name)

        with self.assertRaisesRegex(AssertionError, "outlived the tests of stack"):
            type(self).tearDownClass()

        self.assertIsNotNone(leaked.wait(timeout=30))
        self.assertEqual(self._stack_controllers(), [])


if __name__ == "__main__":
    unittest.main()
