from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_ENVOY_CONFIG = REPO_ROOT / "deploy" / "local" / "envoy.yaml"


class LocalEnvoyConfigTests(unittest.TestCase):
    def test_local_fixture_has_one_provider_neutral_upstream(self) -> None:
        config = yaml.safe_load(LOCAL_ENVOY_CONFIG.read_text(encoding="utf-8"))
        resources = config["static_resources"]
        connection_manager = resources["listeners"][0]["filter_chains"][0]["filters"][
            0
        ]["typed_config"]
        virtual_hosts = connection_manager["route_config"]["virtual_hosts"]

        self.assertEqual(len(virtual_hosts), 1)
        routes = virtual_hosts[0]["routes"]
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["match"], {"prefix": "/"})
        self.assertEqual(routes[0]["route"]["cluster"], "vllm_backend_cluster")
        self.assertNotIn("host_rewrite_literal", routes[0]["route"])
        self.assertEqual(
            {cluster["name"] for cluster in resources["clusters"]},
            {"extproc_service", "vllm_backend_cluster"},
        )


if __name__ == "__main__":
    unittest.main()
