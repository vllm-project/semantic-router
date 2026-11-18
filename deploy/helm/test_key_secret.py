"""Helm-gate contracts for the Secret that keeps keys a ConfigMap save moves out (offline)."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy" / "helm" / "semantic-router"
SECRET = "keys-semantic-router-provider-keys"


def render(values: dict) -> list[dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
        yaml.safe_dump(values, handle)
        handle.flush()
        result = subprocess.run(
            ["helm", "template", "keys", str(CHART), "-f", handle.name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode:
        raise AssertionError(result.stderr)
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


def container(docs: list[dict], component: str) -> dict:
    return next(
        doc["spec"]["template"]["spec"]["containers"][0]
        for doc in docs
        if doc.get("kind") == "Deployment"
        and doc["metadata"]["labels"].get("app.kubernetes.io/component") == component
    )


def env_value(target: dict, name: str):
    return next((e.get("value") for e in target["env"] if e["name"] == name), None)


class KeySecretTest(unittest.TestCase):
    def test_the_router_loads_the_key_secret_and_both_writers_name_it(self):
        docs = render(
            {"dashboard": {"enabled": True}, "envFromSecrets": ["provider-env"]}
        )
        secret = next(
            d
            for d in docs
            if d.get("kind") == "Secret" and d["metadata"]["name"] == SECRET
        )
        # Saves add the entries; an upgrade that renders none leaves them.
        self.assertNotIn("data", secret)
        # Both read the keys a saved config names by entry: the Router to serve
        # them, the Dashboard to verify models and send Try it requests.
        self.assertEqual(
            container(docs, "router")["envFrom"],
            [{"secretRef": {"name": SECRET}}, {"secretRef": {"name": "provider-env"}}],
        )
        self.assertEqual(
            container(docs, "dashboard")["envFrom"], [{"secretRef": {"name": SECRET}}]
        )
        for component in ("router", "dashboard"):
            self.assertEqual(
                env_value(container(docs, component), "VLLM_SR_K8S_KEY_SECRET_NAME"),
                SECRET,
            )

    def test_the_config_writers_may_update_only_the_key_secret(self):
        role = next(
            d
            for d in render({})
            if d.get("kind") == "Role"
            and d["metadata"]["name"].endswith("-config-writer")
        )
        rules = [r for r in role["rules"] if "secrets" in r["resources"]]
        self.assertEqual(
            rules,
            [
                {
                    "apiGroups": [""],
                    "resources": ["secrets"],
                    "resourceNames": [SECRET],
                    "verbs": ["get", "update"],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
