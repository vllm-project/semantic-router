"""Render the production profile and simulate its install without a cluster."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy/helm/semantic-router"

FAKE_HELM = """#!/usr/bin/env bash
case "$1" in
  template) exec "$REAL_HELM" "$@" ;;
  list|status) exit 0 ;;
  install) shift; exec "$REAL_HELM" template "$@" > "$INSTALL_MANIFEST" ;;
  *) exit 1 ;;
esac
"""
FAKE_KUBECTL = """#!/usr/bin/env bash
echo "$*" >> "$KUBECTL_CALLS"
"""


class ProductionImageTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        work = Path(directory.name)
        self.tools_bin = work / "bin"
        self.tools_bin.mkdir()
        self.chart = work / "chart"
        # The guard and both Deployments are in the parent chart. This fixture
        # renders them without downloading unrelated dependency charts.
        shutil.copytree(
            CHART, self.chart, ignore=shutil.ignore_patterns("charts", "Chart.lock")
        )
        self.chart_metadata = yaml.safe_load((self.chart / "Chart.yaml").read_text())
        self.chart_metadata.pop("dependencies", None)
        self.set_app_version("latest")

        self.manifest = work / "installed.yaml"
        self.kube_calls = work / "kubectl.log"
        for name, contents in (("helm", FAKE_HELM), ("kubectl", FAKE_KUBECTL)):
            executable = self.tools_bin / name
            executable.write_text(contents)
            executable.chmod(0o755)
        real_helm = shutil.which("helm")
        if real_helm is None:
            self.fail("helm is required for production image rendering")
        self.env = {
            **os.environ,
            "REAL_HELM": real_helm,
            "INSTALL_MANIFEST": str(self.manifest),
            "KUBECTL_CALLS": str(self.kube_calls),
            "MAKEFLAGS": "",
        }

    def set_app_version(self, tag: str) -> None:
        self.chart_metadata["appVersion"] = tag
        (self.chart / "Chart.yaml").write_text(yaml.safe_dump(self.chart_metadata))

    def install(self, overrides: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "make",
                "helm-prod",
                f"TOOLS_BIN_DIR={self.tools_bin}",
                f"HELM_CHART_PATH={self.chart}",
                f"HELM_SET_VALUES={overrides}",
            ],
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def assert_images(self, router_tag: str, dashboard_tag: str) -> None:
        images = {
            container["name"]: container["image"]
            for doc in yaml.safe_load_all(self.manifest.read_text())
            if doc and doc["kind"] == "Deployment"
            for container in doc["spec"]["template"]["spec"]["containers"]
        }
        self.assertTrue(images["semantic-router"].endswith(f":{router_tag}"), images)
        self.assertTrue(images["dashboard"].endswith(f":{dashboard_tag}"), images)

    def test_rejects_before_installing_if_either_tag_is_unpinned(self) -> None:
        for overrides in (
            "",
            "image.tag=v0.4.0",
            "dashboard.image.tag=v0.4.0",
            "image.tag=v0.4.0,safetyGuards.requirePinnedImages=false",
        ):
            with self.subTest(overrides=overrides):
                result = self.install(overrides)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(
                    "production requires pinned Router and Dashboard image tags",
                    result.stderr,
                )
                self.assertFalse(self.kube_calls.exists())
                self.assertFalse(self.manifest.exists())

    def test_installs_with_both_pinned_tags(self) -> None:
        result = self.install("image.tag=v0.4.0,dashboard.image.tag=v0.4.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_images("v0.4.0", "v0.4.1")

    def test_versioned_chart_supplies_both_tags(self) -> None:
        self.set_app_version("v0.4.0")
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_images("v0.4.0", "v0.4.0")

    def test_explicit_latest_overrides_a_versioned_default(self) -> None:
        self.set_app_version("v0.4.0")
        result = self.install("image.tag=latest")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('image.tag resolves to "latest"', result.stderr)
        self.assertFalse(self.manifest.exists())
