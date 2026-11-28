#!/usr/bin/env python3
"""Address family of the Envoy model clusters `vllm-sr serve --gateway extproc` renders.

A model whose backend is a hostname gets a LOGICAL_DNS cluster, or a STRICT_DNS
cluster when it has more than one backend. The Envoy image the CLI runs loads
those clusters and resolves them from its /etc/hosts: a dual-stack name must
resolve to its IPv4 address, and an IPv6-only name must still resolve.
"""

import json
import os
import time
import unittest
from pathlib import Path
from typing import NamedTuple
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml
from cli.config_generator import generate_envoy_config_from_user_config
from cli.consts import VLLM_SR_ENVOY_CONTAINER_IMAGE_DEFAULT
from cli.parser import parse_user_config
from cli_test_base import CLITestBase, stack_scoped_test_container_name

# Documentation addresses (RFC 5737, RFC 3849): the test only resolves them.
DUAL_STACK_HOST = "dual-stack.test"
DUAL_STACK_IPV4 = "192.0.2.10"
DUAL_STACK_IPV6 = "2001:db8::10"
IPV6_ONLY_HOST = "ipv6-only.test"
IPV6_ONLY_ADDRESS = "2001:db8::20"
HOSTS_ENTRIES = (
    (DUAL_STACK_IPV4, DUAL_STACK_HOST),
    (DUAL_STACK_IPV6, DUAL_STACK_HOST),
    (IPV6_ONLY_ADDRESS, IPV6_ONLY_HOST),
)
RESOLVE_TIMEOUT_SECONDS = 30


class Case(NamedTuple):
    host: str
    backends: int
    cluster_type: str
    expected_address: str


CASES = {
    "dualstackone": Case(DUAL_STACK_HOST, 1, "LOGICAL_DNS", DUAL_STACK_IPV4),
    "dualstacktwo": Case(DUAL_STACK_HOST, 2, "STRICT_DNS", DUAL_STACK_IPV4),
    "ipv6onlyone": Case(IPV6_ONLY_HOST, 1, "LOGICAL_DNS", IPV6_ONLY_ADDRESS),
    "ipv6onlytwo": Case(IPV6_ONLY_HOST, 2, "STRICT_DNS", IPV6_ONLY_ADDRESS),
}


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestEnvoyModelClusterAddressFamily(CLITestBase):
    """Hostname model clusters prefer IPv4 and resolve IPv6-only names."""

    def test_hostname_model_clusters_prefer_ipv4_and_resolve_ipv6_only_names(self):
        rendered = self._render_envoy_config()
        by_name = {
            cluster["name"]: cluster
            for cluster in rendered["static_resources"]["clusters"]
        }
        clusters = {model: by_name[f"model_{model}_cluster"] for model in CASES}
        for model, case in CASES.items():
            self.assertEqual(
                clusters[model]["type"],
                case.cluster_type,
                f"{model} rendered an unexpected cluster type",
            )

        name = stack_scoped_test_container_name(
            self.runtime_stack.stack_name, "vllm-sr-cli-test-envoy-dns"
        )
        admin_port = self._start_envoy(name, rendered["admin"], list(clusters.values()))
        resolved = self._wait_for_resolved_addresses(admin_port, clusters)
        if any(resolved[m] != {c.expected_address} for m, c in CASES.items()):
            print(f"Envoy log tail:\n{self._envoy_logs(name)}")

        for model, case in CASES.items():
            with self.subTest(cluster_type=case.cluster_type, host=case.host):
                self.assertEqual(
                    resolved[model],
                    {case.expected_address},
                    f"{case.cluster_type} cluster for {case.host} resolved "
                    f"{sorted(resolved[model])}",
                )

    def _render_envoy_config(self) -> dict:
        config_path = Path(
            self.write_minimal_canonical_config(model_name=next(iter(CASES)))
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["providers"]["models"] = [
            {
                "name": model,
                "backend_refs": [
                    {
                        "name": f"replica-{index}",
                        "endpoint": f"http://{case.host}:{8000 + index}/v1",
                        "provider": "vllm",
                        "weight": 1,
                    }
                    for index in range(case.backends)
                ],
            }
            for model, case in CASES.items()
        ]
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        output_path = config_path.with_name("envoy.yaml")
        generate_envoy_config_from_user_config(
            parse_user_config(str(config_path)), str(output_path)
        )
        return yaml.safe_load(output_path.read_text(encoding="utf-8"))

    def _start_envoy(self, name: str, admin: dict, clusters: list[dict]) -> int:
        """Run the CLI's Envoy image with its admin block and the model clusters."""
        runtime = self.container_runtime
        self._run_subprocess([runtime, "rm", "-f", name], timeout=30)
        self.addCleanup(self._run_subprocess, [runtime, "rm", "-f", name], timeout=30)

        admin_port = admin["address"]["socket_address"]["port_value"]
        bootstrap = {"admin": admin, "static_resources": {"clusters": clusters}}
        image = (
            os.environ.get("VLLM_SR_ENVOY_IMAGE", "").strip()
            or VLLM_SR_ENVOY_CONTAINER_IMAGE_DEFAULT
        )
        add_hosts = [
            flag
            for address, host in HOSTS_ENTRIES
            for flag in ("--add-host", f"{host}:{address}")
        ]
        started = self._run_subprocess(
            [
                runtime,
                "run",
                "-d",
                "--name",
                name,
                *add_hosts,
                "-p",
                f"127.0.0.1::{admin_port}",
                "--entrypoint",
                "/usr/local/bin/envoy",
                image,
                "--config-yaml",
                json.dumps(bootstrap),
            ],
            timeout=120,
        )
        self.assertEqual(started.returncode, 0, started.stderr)

        # A runtime drops IPv6 --add-host entries when the container has IPv6
        # disabled; without them the test cannot tell what the template did.
        hosts = self._run_subprocess(
            [runtime, "exec", name, "cat", "/etc/hosts"], timeout=30
        )
        if hosts.returncode != 0:
            self.fail(f"{hosts.stderr}\nEnvoy log tail:\n{self._envoy_logs(name)}")
        entries = {tuple(line.split()[:2]) for line in hosts.stdout.splitlines()}
        for address, host in HOSTS_ENTRIES:
            self.assertIn(
                (address, host),
                entries,
                f"{runtime} left '{address} {host}' out of /etc/hosts; "
                "containers here may have IPv6 disabled",
            )

        published = self._run_subprocess(
            [runtime, "port", name, f"{admin_port}/tcp"], timeout=30
        )
        self.assertEqual(published.returncode, 0, published.stderr)
        return int(published.stdout.strip().splitlines()[0].rsplit(":", 1)[1])

    def _envoy_logs(self, name: str) -> str:
        logs = self._run_subprocess(
            [self.container_runtime, "logs", "--tail", "50", name], timeout=30
        )
        return logs.stdout + logs.stderr

    def _wait_for_resolved_addresses(
        self, admin_port: int, clusters: dict[str, dict]
    ) -> dict[str, set[str]]:
        """Poll the admin API until every cluster has a host, or the timeout."""
        deadline = time.monotonic() + RESOLVE_TIMEOUT_SECONDS
        while True:
            addresses = self._cluster_addresses(admin_port)
            resolved = {
                model: addresses.get(cluster["name"], set())
                for model, cluster in clusters.items()
            }
            if all(resolved.values()) or time.monotonic() >= deadline:
                return resolved
            time.sleep(0.5)

    @staticmethod
    def _cluster_addresses(admin_port: int) -> dict[str, set[str]]:
        url = f"http://127.0.0.1:{admin_port}/clusters?format=json"
        try:
            with urllib_request.urlopen(url, timeout=5) as response:
                body = json.load(response)
        except (urllib_error.URLError, ConnectionError, TimeoutError):
            return {}
        return {
            status["name"]: {
                host["address"]["socket_address"]["address"]
                for host in status.get("host_statuses", [])
            }
            for status in body.get("cluster_statuses", [])
        }


if __name__ == "__main__":
    unittest.main()
