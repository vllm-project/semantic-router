"""Helm-gate Grafana Live contracts for charts and static deployments (offline)."""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ORIGINS = "https://dashboard.example.com,https://dashboard.example.net:8443"
ENV_NAME = "GF_LIVE_ALLOWED_ORIGINS"


def run(*command: str, **kwargs) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False, **kwargs
    )
    if result.returncode:
        raise AssertionError(f"{command}:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def grafana_env(rendered: str) -> dict:
    deployment = next(
        doc
        for doc in yaml.safe_load_all(rendered)
        if isinstance(doc, dict)
        and doc.get("kind") == "Deployment"
        and doc["metadata"]["name"] == "grafana"
    )
    container = next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"]
        if item["name"] == "grafana"
    )
    # Kubernetes defaults an omitted EnvVar.value to the empty string.
    return {item["name"]: item.get("value", "") for item in container["env"]}


class HelmGrafanaLiveTests(unittest.TestCase):
    def test_rendered_grafana_config(self):
        for origins in (None, ORIGINS):
            with self.subTest(origins=origins), tempfile.TemporaryDirectory() as tmp:
                values = {
                    "dashboard": {"enabled": True},
                    "dependencies": {"observability": {"grafana": {"enabled": True}}},
                }
                if origins is not None:
                    values["grafana"] = {
                        "grafana.ini": {"live": {"allowed_origins": origins}}
                    }
                path = Path(tmp) / "values.yaml"
                path.write_text(yaml.safe_dump(values))
                documents = list(
                    yaml.safe_load_all(
                        run(
                            "helm",
                            "template",
                            "live-test",
                            "deploy/helm/semantic-router",
                            "-f",
                            str(path),
                        )
                    )
                )
                configmap = next(
                    doc
                    for doc in documents
                    if doc["kind"] == "ConfigMap"
                    and "grafana.ini" in doc.get("data", {})
                )
                config = configparser.ConfigParser(interpolation=None)
                config.read_string(configmap["data"]["grafana.ini"])
                self.assertEqual(config.get("live", "allowed_origins"), origins or "")
                deployment = next(
                    doc
                    for doc in documents
                    if doc["kind"] == "Deployment"
                    and doc["metadata"]["name"] == "live-test-grafana"
                )
                volumes = deployment["spec"]["template"]["spec"]["volumes"]
                self.assertTrue(
                    any(
                        volume.get("configMap", {}).get("name")
                        == configmap["metadata"]["name"]
                        for volume in volumes
                    )
                )


class StaticGrafanaLiveTests(unittest.TestCase):
    def test_defaults_and_kustomize_overrides(self):
        for platform in ("kubernetes", "openshift"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as tmp:
                base = ROOT / "deploy" / platform / "observability"
                # Render the complete maintained Kubernetes package. OpenShift's
                # script owns the stack; its Grafana manifest is also patchable alone.
                if platform == "kubernetes":
                    shutil.copytree(base, Path(tmp) / "base")
                else:
                    (Path(tmp) / "base").mkdir()
                    shutil.copy(
                        base / "grafana/deployment.yaml",
                        Path(tmp) / "base/grafana.yaml",
                    )
                    (Path(tmp) / "base/kustomization.yaml").write_text(
                        "resources:\n  - grafana.yaml\n"
                    )
                default = run("kubectl", "kustomize", str(Path(tmp) / "base"))
                self.assertEqual(grafana_env(default)[ENV_NAME], "")
                patch = {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": "grafana"},
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "grafana",
                                        "env": [{"name": ENV_NAME, "value": ORIGINS}],
                                    }
                                ]
                            }
                        }
                    },
                }
                (Path(tmp) / "kustomization.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "resources": ["base"],
                            "patches": [
                                {
                                    "target": {"kind": "Deployment", "name": "grafana"},
                                    "patch": yaml.safe_dump(patch),
                                }
                            ],
                        }
                    )
                )
                rendered = run("kubectl", "kustomize", tmp)
                expected = grafana_env(default) | {ENV_NAME: ORIGINS}
                self.assertEqual(grafana_env(rendered), expected)


# Only local YAML operations use the real kubectl. All cluster operations are
# replaced, so this test cannot deploy, build images, or contact a cluster.
OFFLINE_OPENSHIFT = r"""
oc() {
    case "$1 ${2:-}" in
        'whoami ') echo test-user ;;
        'get namespace') return 0 ;;
        'get svc')
            case "$3" in
                vllm-model-a) echo 192.0.2.10 ;;
                vllm-model-b) echo 192.0.2.11 ;;
                *) return 1 ;;
            esac ;;
        'get route') echo "$3.apps.example.org" ;;
        'get imagestream'|'get buildconfig') return 0 ;;
        'create namespace'|'create configmap'|'patch buildconfig/dashboard-custom'|\
        'start-build dashboard-custom'|'rollout status') return 0 ;;
        'set env')
            if [[ "${FAIL_LOCAL_RENDER:-}" == true ]]; then return 1; fi
            [[ " $* " == *' --local '* ]] || return 1
            command kubectl "$@" ;;
        apply\ *)
            while [[ $# -gt 0 && "$1" != -f ]]; do shift; done
            shift
            printf '\n---\n' >> "$APPLIED_MANIFESTS"
            cat "$1" >> "$APPLIED_MANIFESTS" ;;
        *) echo "Unexpected oc call: $*" >&2; return 1 ;;
    esac
}
source "$1" --namespace live-test
"""


class OpenShiftGrafanaLiveTests(unittest.TestCase):
    def render(self, origins, *, manifest_origins="", fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # Copy the deployment inputs so patched-manifest cases cannot edit
            # the checkout. Temporary render files stay in this directory too.
            shutil.copytree(ROOT / "deploy/openshift", directory / "openshift")
            manifest = directory / "openshift/observability/grafana/deployment.yaml"
            doc = yaml.safe_load(manifest.read_text())
            for entry in doc["spec"]["template"]["spec"]["containers"][0]["env"]:
                if entry["name"] == ENV_NAME:
                    entry["value"] = manifest_origins
            manifest.write_text(yaml.safe_dump(doc))
            output = directory / "applied.yaml"
            env = {
                **os.environ,
                "TMPDIR": tmp,
                "APPLIED_MANIFESTS": str(output),
                "KUBECONFIG": str(directory / "no-cluster"),
                "PROVIDER_MOCKER_IMAGE": "example.com/provider-mocker@sha256:"
                + "0" * 64,
                "FAIL_LOCAL_RENDER": "true" if fail else "false",
            }
            env.pop(ENV_NAME, None)
            if origins is not None:
                env[ENV_NAME] = origins
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    OFFLINE_OPENSHIFT,
                    "test",
                    str(directory / "openshift/deploy-to-openshift.sh"),
                ],
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if not fail:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
            return output.read_text()

    def test_script_default_override_and_clear(self):
        for origins, patched, expected in (
            (None, "", ""),
            (ORIGINS, "https://old.example.com", ORIGINS),
            (None, ORIGINS, ORIGINS),
            ("", ORIGINS, ""),
        ):
            with self.subTest(origins=origins, patched=patched):
                rendered = self.render(origins, manifest_origins=patched)
                env = grafana_env(rendered)
                self.assertEqual(env[ENV_NAME], expected)
                self.assertEqual(
                    env["GF_SERVER_ROOT_URL"], "https://grafana.apps.example.org"
                )
                self.assertNotIn("DYNAMIC_GRAFANA_ROUTE_URL", rendered)
                deployment = next(
                    doc
                    for doc in yaml.safe_load_all(rendered)
                    if isinstance(doc, dict)
                    and doc.get("kind") == "Deployment"
                    and doc["metadata"]["name"] == "grafana"
                )
                self.assertEqual(deployment["metadata"]["namespace"], "live-test")

    def test_failed_local_render_does_not_apply_grafana(self):
        rendered = self.render(ORIGINS, fail=True)
        self.assertFalse(
            any(
                isinstance(doc, dict)
                and doc.get("kind") == "Deployment"
                and doc["metadata"]["name"] == "grafana"
                for doc in yaml.safe_load_all(rendered)
            )
        )


if __name__ == "__main__":
    unittest.main()
