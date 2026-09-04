#!/usr/bin/env python3
"""
test_integration_storage_isolation.py - the storage data-network boundary.

The unit suite in `src/vllm-sr/tests/test_storage_network_isolation.py` mocks
the container runtime, so it can only assert the command sequence the CLI
emits. Whether a workload on the application network can actually open a
connection to a store is a property of a real daemon, and that is what these
tests measure: one serve session provisions Redis and Postgres, then the same
probe runs from each of the stack's two networks.

"""

import os
import unittest
from contextlib import contextmanager

from cli_test_base import CLITestBase
from serve_session import ServeSessionMixin

REDIS_PROBE_IMAGE = "docker.io/library/redis:7-alpine"
POSTGRES_PROBE_IMAGE = "docker.io/library/postgres:16-alpine"

# A store that answers says so in its protocol, not in an exit code. Redis
# reports `NOAUTH` to an unauthenticated PING, which is a nonzero exit and a
# proof of reachability at the same time -- so the probe is classified by what
# it printed. Reaching for the password would only narrow what the test covers.
REDIS_REACHABLE_MARKERS = ("PONG", "NOAUTH")
POSTGRES_REACHABLE_MARKER = "accepting connections"

integration_only = unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)


class TestStorageNetworkIsolation(ServeSessionMixin, CLITestBase):
    """Real-daemon coverage for the east-west boundary around the stores."""

    # Provisioning two stores, re-keying Postgres, and starting the runtime
    # takes longer than a serve that has no storage to bring up.
    STORAGE_STARTUP_TIMEOUT = 300

    CONTROL_CONTAINER_NAME = "vllm-sr-cli-test-control-redis"

    def setUp(self):
        super().setUp()
        self.HEALTH_CHECK_TIMEOUT = self.STORAGE_STARTUP_TIMEOUT

    def tearDown(self):
        self._run_subprocess(
            [self.container_runtime, "rm", "-f", self.CONTROL_CONTAINER_NAME],
            timeout=30,
        )
        # `stop` is what removes both stack networks; the class-level cleanup
        # only removes containers, and a leftover network makes the next run
        # reuse a boundary it never created.
        self.run_cli(["stop"], timeout=60)
        self._cleanup_container()
        super().tearDown()

    @integration_only
    def test_the_stores_answer_only_from_the_data_network(self):
        """Probe both stores from both networks inside one serve session."""
        self.print_test_header(
            "storage data-network boundary",
            "Redis and Postgres answer on the data network and nowhere else",
        )
        with self._running_serve(managed_storage=True):
            self._assert_the_stores_are_on_the_data_network_alone()
            self._assert_the_stores_answer_from_the_data_network()
            self._assert_the_stores_are_unreachable_from_the_application_network()
            self._assert_the_application_network_probe_is_not_vacuous()
            self._assert_the_probe_sees_a_store_moved_back()
        self.print_test_result(True, "the boundary holds and the probe detects it")

    def _assert_the_stores_are_on_the_data_network_alone(self):
        """The attachment itself, which is what a regression would change."""
        for container_name in (
            self.REDIS_CONTAINER_NAME,
            self.POSTGRES_CONTAINER_NAME,
        ):
            self.assertEqual(
                self.container_networks(container_name),
                {self.DATA_NETWORK_NAME},
                f"{container_name} is not on the data network alone",
            )
        print("  ✓ Redis and Postgres are attached to the data network alone")

    def _assert_the_stores_answer_from_the_data_network(self):
        """The positive half, and the only guard against a vacuous probe.

        `pg_isready` reports a host it cannot resolve exactly the way it
        reports one it cannot reach, so a mistyped target would look like a
        clean pass on the application network. It cannot look like a pass
        here, and both probes name the target through the same helper.
        """
        redis_output = self._probe_redis(self.DATA_NETWORK_NAME)
        self.assertTrue(
            self._redis_answered(redis_output),
            f"Redis did not answer from the data network: {redis_output}",
        )
        postgres_output = self._probe_postgres(self.DATA_NETWORK_NAME)
        self.assertIn(
            POSTGRES_REACHABLE_MARKER,
            postgres_output,
            f"Postgres did not answer from the data network: {postgres_output}",
        )
        print("  ✓ Both stores answer from the data network")

    def _assert_the_stores_are_unreachable_from_the_application_network(self):
        """The boundary. Envoy, Dashboard, and OpenClaw workloads live here."""
        redis_output = self._probe_redis(self.NETWORK_NAME)
        self.assertFalse(
            self._redis_answered(redis_output),
            f"Redis answered a workload on the application network: {redis_output}",
        )
        postgres_output = self._probe_postgres(self.NETWORK_NAME)
        self.assertNotIn(
            POSTGRES_REACHABLE_MARKER,
            postgres_output,
            f"Postgres answered a workload on the application network: "
            f"{postgres_output}",
        )
        print("  ✓ Neither store answers from the application network")

    def _assert_the_application_network_probe_is_not_vacuous(self):
        """A control container genuinely on the application network answers."""
        with self._running_control_redis():
            output = self._probe_redis(
                self.NETWORK_NAME, host=self.CONTROL_CONTAINER_NAME
            )
        self.assertTrue(
            self._redis_answered(output),
            f"the control container did not answer, so the probe proves "
            f"nothing about the managed stores: {output}",
        )
        print("  ✓ A control container on the application network answers")

    def _assert_the_probe_sees_a_store_moved_back(self):
        """Move Redis back and require the probe to notice, then restore it.

        This is the regression the boundary exists to prevent, so the suite
        performs it rather than trusting that a probe which has only ever seen
        a passing stack would fail on a broken one.
        """
        connect = self._run_subprocess(
            [
                self.container_runtime,
                "network",
                "connect",
                self.NETWORK_NAME,
                self.REDIS_CONTAINER_NAME,
            ],
            timeout=30,
        )
        self.assertEqual(
            connect.returncode,
            0,
            f"could not move Redis back onto the application network: "
            f"{connect.stderr}",
        )
        try:
            output = self._probe_redis(self.NETWORK_NAME)
            self.assertTrue(
                self._redis_answered(output),
                f"Redis was moved onto the application network and the probe "
                f"still reported it unreachable, so the probe cannot detect a "
                f"regression: {output}",
            )
        finally:
            disconnect = self._run_subprocess(
                [
                    self.container_runtime,
                    "network",
                    "disconnect",
                    self.NETWORK_NAME,
                    self.REDIS_CONTAINER_NAME,
                ],
                timeout=30,
            )
            self.assertEqual(
                disconnect.returncode,
                0,
                f"Redis was left on the application network: " f"{disconnect.stderr}",
            )
        print("  ✓ The probe reports a store moved back onto the application network")

    @contextmanager
    def _running_control_redis(self):
        """Run an unmanaged Redis on the application network.

        It carries no `requirepass`, so its `PONG` also shows that the blocked
        probes above stopped at the connection rather than at authentication.
        """
        self._run_subprocess(
            [self.container_runtime, "rm", "-f", self.CONTROL_CONTAINER_NAME],
            timeout=30,
        )
        result = self._run_subprocess(
            [
                self.container_runtime,
                "run",
                "-d",
                "--name",
                self.CONTROL_CONTAINER_NAME,
                "--network",
                self.NETWORK_NAME,
                REDIS_PROBE_IMAGE,
            ],
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"failed to start the control container: {result.stderr}",
        )
        self.assertTrue(
            self.wait_for_container_running(
                timeout=30,
                container_name=self.CONTROL_CONTAINER_NAME,
            ),
            "the control container did not reach running state",
        )
        try:
            yield
        finally:
            self._run_subprocess(
                [self.container_runtime, "rm", "-f", self.CONTROL_CONTAINER_NAME],
                timeout=30,
            )

    def _probe_redis(self, network_name: str, host: str | None = None) -> str:
        """Ask Redis for a PING from inside *network_name*."""
        target = host or self.REDIS_CONTAINER_NAME
        return self._probe_output(
            network_name=network_name,
            image=REDIS_PROBE_IMAGE,
            # `timeout` bounds a connection a blocked network black-holes
            # instead of refusing, which is what keeps a failing boundary from
            # stalling the job until the suite-level timeout.
            shell_command=f"timeout 10 redis-cli -h {target} -p 6379 ping 2>&1",
        )

    def _probe_postgres(self, network_name: str, host: str | None = None) -> str:
        """Ask Postgres whether it accepts connections from *network_name*."""
        target = host or self.POSTGRES_CONTAINER_NAME
        return self._probe_output(
            network_name=network_name,
            image=POSTGRES_PROBE_IMAGE,
            shell_command=f"pg_isready -h {target} -p 5432 -t 10 2>&1",
        )

    def _probe_output(
        self, *, network_name: str, image: str, shell_command: str
    ) -> str:
        """Run one probe and return what it printed.

        The image is the store's own, which is already local once `serve` has
        started that store, so the probe speaks the real protocol without
        pulling anything.
        """
        result = self.run_network_probe(
            network_name=network_name,
            image=image,
            shell_command=f"{shell_command} || true",
            timeout=90,
        )
        output = f"{result.stdout}{result.stderr}".strip()
        print(f"    probe on {network_name}: {output or '<no output>'}")
        return output

    @staticmethod
    def _redis_answered(output: str) -> bool:
        return any(marker in output for marker in REDIS_REACHABLE_MARKERS)


if __name__ == "__main__":
    unittest.main()
