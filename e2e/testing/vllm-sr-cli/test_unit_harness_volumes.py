"""A test stack's data volumes never outlive its tests.

`vllm-sr stop` keeps a stack's named data volumes for the next serve; the
harness removes them, because a test stack never returns.
"""

import subprocess
import unittest
from unittest import mock

from cli.runtime_stack import DEFAULT_STACK_NAME, resolve_runtime_stack
from cli_test_base import CLITestBase


class TestStackVolumes(CLITestBase):
    def _runtime(self, volumes: list[str], *, removable: bool = True):
        """Stand in for a container runtime that holds *volumes*.

        Returns the volumes it still holds and every command it ran, both
        kept current. It removes a volume only if *removable*, as a runtime
        refuses to remove one a container still mounts.
        """
        present = list(volumes)
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            if command[1:] == ["volume", "ls", "-q"]:
                listing = "".join(f"{volume}\n" for volume in present)
                return subprocess.CompletedProcess(command, 0, listing, "")
            if command[1:3] == ["volume", "rm"]:
                if not removable:
                    return subprocess.CompletedProcess(command, 1, "", "in use")
                present.remove(command[3])
            return subprocess.CompletedProcess(command, 0, "", "")

        patcher = mock.patch.object(CLITestBase, "_run_subprocess", side_effect=run)
        patcher.start()
        self.addCleanup(patcher.stop)
        return present, commands

    def test_cleanup_removes_this_stacks_data_volumes_only(self):
        ours = [
            f"{self.runtime_stack.redis_container_name}-data",
            f"{self.runtime_stack.grafana_container_name}-data",
        ]
        theirs = [
            f"{self.runtime_stack.stack_name}-another-vllm-sr-redis-data",
            f"{DEFAULT_STACK_NAME}-redis-data",
            "0" * 64,
        ]
        present, commands = self._runtime(ours + theirs)

        self._cleanup_container()

        self.assertEqual(present, theirs)
        volume_removals = [
            index
            for index, command in enumerate(commands)
            if command[1:3] == ["volume", "rm"]
        ]
        self.assertEqual([commands[index][3] for index in volume_removals], ours)
        container_commands = [
            index
            for index, command in enumerate(commands)
            if command[1] in ("stop", "rm")
        ]
        self.assertLess(max(container_commands), min(volume_removals))

    def test_a_class_without_its_own_test_stack_touches_no_volume(self):
        present, commands = self._runtime([f"{DEFAULT_STACK_NAME}-redis-data"])

        with mock.patch.object(CLITestBase, "container_runtime", "docker", create=True):
            CLITestBase._cleanup_container()

        self.assertEqual(present, [f"{DEFAULT_STACK_NAME}-redis-data"])
        self.assertFalse([command for command in commands if command[1] == "volume"])

    def test_the_default_stack_keeps_its_volumes(self):
        self._runtime([f"{DEFAULT_STACK_NAME}-redis-data"])
        default_stack = resolve_runtime_stack(stack_name=DEFAULT_STACK_NAME)

        with mock.patch.object(type(self), "runtime_stack", default_stack):
            self.assertEqual(self._stack_volumes(), [])

    def test_a_data_volume_that_outlives_the_tests_fails_the_class(self):
        held = f"{self.runtime_stack.redis_container_name}-data"
        present, _commands = self._runtime([held], removable=False)

        with self.assertRaisesRegex(
            AssertionError, "data volumes outlived the tests of stack"
        ):
            type(self).tearDownClass()

        self.assertEqual(present, [held])


if __name__ == "__main__":
    unittest.main()
