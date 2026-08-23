"""Fast deployment-contract checks for managed and standalone CLI modes."""

import unittest

from cli.control_plane_deployment import local_managed_store_environment
from cli.control_plane_migration import build_control_plane_migration_command
from cli.runtime_stack import resolve_runtime_stack
from cli.storage_backends import detect_required_backends


def _managed_config():
    return {
        "version": "v0.4",
        "global": {
            "control_plane": {"mode": "managed"},
            "stores": {
                "access": {
                    "type": "postgres",
                    "postgres": {"dsn_env": "ACCESS_DATABASE_URL"},
                },
                "access_runtime": {
                    "type": "redis",
                    "redis": {"url_env": "ACCESS_RUNTIME_URL"},
                },
            },
        },
    }


class TestManagedDeploymentContract(unittest.TestCase):
    def test_standalone_does_not_select_postgres_or_valkey(self):
        config = {"version": "v0.4", "listeners": []}

        self.assertEqual(detect_required_backends(config), set())
        runtime_env = {}
        with local_managed_store_environment(
            config,
            runtime_env,
            resolve_runtime_stack(),
            state_root_dir=".",
        ) as backends:
            self.assertEqual(backends, set())

    def test_managed_migration_inherits_dsn_without_argv_value(self):
        secret = "postgresql://operator-secret@postgres/control"

        command = build_control_plane_migration_command(
            _managed_config(),
            env_vars={"ACCESS_DATABASE_URL": secret},
            network_name="vllm-sr-network",
            router_image="router:test",
            container_runtime="docker",
        )

        self.assertIn("/usr/local/bin/access-migrate", command)
        self.assertIn("ACCESS_DATABASE_URL", command)
        self.assertTrue(all(secret not in value for value in command))


if __name__ == "__main__":
    unittest.main()
