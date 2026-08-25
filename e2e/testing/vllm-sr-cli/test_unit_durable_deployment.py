"""Fast deployment checks for capability-derived file and durable routing."""

import unittest

from cli.control_plane_deployment import runtime_capabilities
from cli.management_migration import build_management_migration_command
from cli.storage_backends import detect_required_backends


def _durable_config():
    return {
        "version": "v0.3",
        "global": {
            "stores": {
                "management": {
                    "postgres": {"dsn_env": "ACCESS_DATABASE_URL"},
                },
                "runtime": {
                    "redis": {"url_env": "ACCESS_RUNTIME_URL"},
                },
            },
        },
    }


class TestDurableDeploymentContract(unittest.TestCase):
    def test_file_routing_does_not_select_postgres_or_valkey(self):
        config = {"version": "v0.3", "listeners": []}

        self.assertEqual(detect_required_backends(config), set())
        capabilities = runtime_capabilities(config)
        self.assertTrue(capabilities.file_routing)
        self.assertFalse(capabilities.durable_management)

    def test_durable_migration_inherits_dsn_without_argv_value(self):
        secret = "postgresql://operator-secret@postgres/control"

        command = build_management_migration_command(
            _durable_config(),
            env_vars={"ACCESS_DATABASE_URL": secret},
            network_name="vllm-sr-network",
            router_image="router:test",
            container_runtime="docker",
        )

        self.assertIn("/usr/local/bin/management-migrate", command)
        self.assertIn("ACCESS_DATABASE_URL", command)
        self.assertTrue(all(secret not in value for value in command))


if __name__ == "__main__":
    unittest.main()
