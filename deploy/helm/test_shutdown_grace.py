"""Helm-gate contracts for how the Router pod stops: its grace period and lifecycle hooks (offline)."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy" / "helm" / "semantic-router"


def router_pod(values: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
        yaml.safe_dump(values, handle)
        handle.flush()
        result = subprocess.run(
            ["helm", "template", "grace", str(CHART), "-f", handle.name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode:
        raise AssertionError(result.stderr)
    return next(
        doc["spec"]["template"]["spec"]
        for doc in yaml.safe_load_all(result.stdout)
        if isinstance(doc, dict)
        and doc.get("kind") == "Deployment"
        and doc["metadata"]["labels"].get("app.kubernetes.io/component") == "router"
    )


class ShutdownGraceTest(unittest.TestCase):
    def test_defaults_keep_the_kubernetes_grace_period(self):
        pod = router_pod({})
        self.assertNotIn("terminationGracePeriodSeconds", pod)
        self.assertNotIn("lifecycle", pod["containers"][0])

    def test_grace_period_and_pre_stop_hook_reach_the_router_pod(self):
        pre_stop = {"preStop": {"exec": {"command": ["sleep", "15"]}}}
        pod = router_pod({"terminationGracePeriodSeconds": 220, "lifecycle": pre_stop})
        self.assertEqual(pod["terminationGracePeriodSeconds"], 220)
        self.assertEqual(pod["containers"][0]["lifecycle"], pre_stop)

    def test_an_explicit_zero_grace_period_reaches_the_router_pod(self):
        pod = router_pod({"terminationGracePeriodSeconds": 0})
        self.assertEqual(pod["terminationGracePeriodSeconds"], 0)


if __name__ == "__main__":
    unittest.main()
