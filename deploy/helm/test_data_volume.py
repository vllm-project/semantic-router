"""Helm-gate contracts for the Router's writable data directory (offline)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy" / "helm" / "semantic-router"
RELEASE = "data"
DATA_DIR = "/var/lib/vsr"
# The storage pages' Helm values; the Router's docs test loads their Router
# examples (src/semantic-router/pkg/config/local_storage_test.go).
DOC_PAGES = (
    ROOT / "website/docs/installation/storage-overview.md",
    ROOT
    / "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/installation/storage-overview.md",
)
VALUES_BLOCK = re.compile(r'^```yaml title="values\.yaml"\n(.*?)^```', re.M | re.S)
VECTOR_STORE = {
    "config": {
        "global": {
            "stores": {
                "vector_store": {
                    "enabled": True,
                    "backend_type": "memory",
                }
            }
        }
    }
}


def helm(*args: str, values: dict | None = None) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
        yaml.safe_dump(values or {}, handle)
        handle.flush()
        return subprocess.run(
            ["helm", *args, RELEASE, str(CHART), "-f", handle.name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )


def render(values: dict | None = None) -> list[dict]:
    result = helm("template", values=values)
    if result.returncode:
        raise AssertionError(result.stderr)
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


def install_note(values: dict | None = None) -> str:
    """Render the note NOTES.txt prints, which helm template leaves out."""
    with tempfile.TemporaryDirectory() as workspace:
        chart = Path(workspace) / "semantic-router"
        shutil.copytree(CHART, chart)
        (chart / "templates" / "zz-install-note.yaml").write_text(
            "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: install-note\n"
            'data:\n  note: {{ include "semantic-router.dataVolumeNote" . | quote }}\n',
            encoding="utf-8",
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump(values or {}, handle)
            handle.flush()
            result = subprocess.run(
                [
                    "helm",
                    "template",
                    RELEASE,
                    str(chart),
                    "-f",
                    handle.name,
                    "--show-only",
                    "templates/zz-install-note.yaml",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
    if result.returncode:
        raise AssertionError(result.stderr)
    return yaml.safe_load(result.stdout)["data"]["note"]


def router_pod(documents: list[dict]) -> dict:
    return next(
        doc["spec"]["template"]["spec"]
        for doc in documents
        if doc["kind"] == "Deployment"
        and doc["metadata"]["labels"].get("app.kubernetes.io/component") == "router"
    )


def data_mounts(pod: dict) -> list[dict]:
    return [
        mount
        for mount in pod["containers"][0]["volumeMounts"]
        if mount["mountPath"].rstrip("/") == DATA_DIR
    ]


def volume(pod: dict, name: str) -> dict:
    return next(item for item in pod["volumes"] if item["name"] == name)


def claims(documents: list[dict]) -> dict[str, dict]:
    return {
        doc["metadata"]["name"]: doc
        for doc in documents
        if doc["kind"] == "PersistentVolumeClaim"
    }


class DataVolumeTest(unittest.TestCase):
    def test_default_is_a_writable_empty_dir_beside_a_read_only_root(self) -> None:
        documents = render()
        pod = router_pod(documents)
        self.assertTrue(
            pod["containers"][0]["securityContext"]["readOnlyRootFilesystem"]
        )
        [mount] = data_mounts(pod)
        self.assertEqual(
            volume(pod, mount["name"]), {"name": "router-data", "emptyDir": {}}
        )
        self.assertNotIn("data-semantic-router-data", claims(documents))

    def test_persistence_renders_and_mounts_a_claim(self) -> None:
        documents = render(
            {
                "data": {
                    "persistence": {
                        "enabled": True,
                        "size": "5Gi",
                        "storageClassName": "fast",
                        "accessMode": "ReadWriteMany",
                        "annotations": {"helm.sh/resource-policy": "keep"},
                    }
                }
            }
        )
        claim = claims(documents)["data-semantic-router-data"]
        self.assertEqual(claim["spec"]["accessModes"], ["ReadWriteMany"])
        self.assertEqual(claim["spec"]["storageClassName"], "fast")
        self.assertEqual(claim["spec"]["resources"]["requests"]["storage"], "5Gi")
        self.assertEqual(
            claim["metadata"]["annotations"], {"helm.sh/resource-policy": "keep"}
        )
        pod = router_pod(documents)
        [mount] = data_mounts(pod)
        self.assertEqual(
            volume(pod, mount["name"])["persistentVolumeClaim"],
            {"claimName": "data-semantic-router-data"},
        )

    def test_existing_claim_is_mounted_without_a_new_one(self) -> None:
        documents = render(
            {"data": {"persistence": {"enabled": True, "existingClaim": "uploads"}}}
        )
        self.assertNotIn("data-semantic-router-data", claims(documents))
        pod = router_pod(documents)
        [mount] = data_mounts(pod)
        self.assertEqual(
            volume(pod, mount["name"])["persistentVolumeClaim"],
            {"claimName": "uploads"},
        )

    def test_an_extra_mount_replaces_the_chart_volume(self) -> None:
        documents = render(
            {
                "data": {"persistence": {"enabled": True}},
                "extraVolumes": [
                    {"name": "my-data", "persistentVolumeClaim": {"claimName": "mine"}}
                ],
                "extraVolumeMounts": [{"name": "my-data", "mountPath": DATA_DIR + "/"}],
            }
        )
        self.assertNotIn("data-semantic-router-data", claims(documents))
        pod = router_pod(documents)
        self.assertEqual([mount["name"] for mount in data_mounts(pod)], ["my-data"])
        self.assertNotIn("router-data", [item["name"] for item in pod["volumes"]])

    def test_schema_rejects_unknown_keys_and_read_only_access(self) -> None:
        for values, field in (
            ({"data": {"mountPath": "/data"}}, "mountPath"),
            ({"data": {"persistence": {"accessMode": "ReadOnlyMany"}}}, "accessMode"),
            ({"data": {"persistence": {"enabled": "yes"}}}, "enabled"),
        ):
            result = helm("template", values=values)
            self.assertNotEqual(result.returncode, 0, values)
            self.assertIn(field, result.stderr)

    def test_documented_values_put_the_data_directory_on_a_claim(self) -> None:
        for page in DOC_PAGES:
            blocks = VALUES_BLOCK.findall(page.read_text(encoding="utf-8"))
            self.assertTrue(blocks, f"{page} has no values.yaml block")
            for block in blocks:
                values = yaml.safe_load(block)
                documents = render(values)
                pod = router_pod(documents)
                [mount] = data_mounts(pod)
                claim = volume(pod, mount["name"])["persistentVolumeClaim"]
                self.assertEqual(
                    claims(documents)[claim["claimName"]]["spec"]["resources"],
                    {"requests": {"storage": values["data"]["persistence"]["size"]}},
                )

    def test_install_note_warns_only_about_files_in_an_empty_dir(self) -> None:
        self.assertIn("--set data.persistence.enabled=true", install_note(VECTOR_STORE))
        persistent = {**VECTOR_STORE, "data": {"persistence": {"enabled": True}}}
        elsewhere = yaml.safe_load(yaml.safe_dump(VECTOR_STORE))
        elsewhere["config"]["global"]["stores"]["vector_store"][
            "file_storage_dir"
        ] = "/mnt/uploads"
        for values in (None, persistent, elsewhere):
            self.assertEqual(install_note(values), "", values)


if __name__ == "__main__":
    unittest.main()
