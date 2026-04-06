"""Local Kind lifecycle support for the managed k8s dev target."""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import textwrap
from pathlib import Path

DEFAULT_KIND_CLUSTER_NAME = "vllm-sr"
ML_MODELS_HOST_DIR = Path("/tmp/kind-ml-models")


class KindClusterManager:
    """Manage the default Kind cluster used by the CLI k8s dev path."""

    def __init__(self, cluster_name: str | None = None) -> None:
        self.cluster_name = (
            cluster_name
            or os.getenv("VLLM_SR_KIND_CLUSTER_NAME")
            or DEFAULT_KIND_CLUSTER_NAME
        )

    @property
    def context_name(self) -> str:
        return f"kind-{self.cluster_name}"

    def exists(self) -> bool:
        result = subprocess.run(
            ["kind", "get", "clusters"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        clusters = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        return self.cluster_name in clusters

    def ensure(self) -> None:
        if self.exists():
            return

        ML_MODELS_HOST_DIR.mkdir(parents=True, exist_ok=True)
        config_path = self._write_cluster_config()
        try:
            subprocess.run(
                [
                    "kind",
                    "create",
                    "cluster",
                    "--name",
                    self.cluster_name,
                    "--config",
                    config_path,
                    "--wait",
                    "5m",
                ],
                check=True,
            )
        finally:
            Path(config_path).unlink(missing_ok=True)

    def delete(self) -> None:
        if not self.exists():
            return
        subprocess.run(
            ["kind", "delete", "cluster", "--name", self.cluster_name],
            check=False,
        )

    def load_image(self, image_name: str) -> None:
        subprocess.run(
            [
                "kind",
                "load",
                "docker-image",
                image_name,
                "--name",
                self.cluster_name,
            ],
            check=True,
        )

    def _write_cluster_config(self) -> str:
        host_mount_path = self._host_mount_path()
        cluster_config = textwrap.dedent(
            f"""\
            kind: Cluster
            apiVersion: kind.x-k8s.io/v1alpha4
            name: {self.cluster_name}
            nodes:
              - role: control-plane
                extraMounts:
                  - hostPath: {host_mount_path}
                    containerPath: /mnt
                  - hostPath: {ML_MODELS_HOST_DIR}
                    containerPath: /tmp/ml-models
              - role: worker
                extraMounts:
                  - hostPath: {host_mount_path}
                    containerPath: /mnt
                  - hostPath: {ML_MODELS_HOST_DIR}
                    containerPath: /tmp/ml-models
            """
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"kind-{self.cluster_name}-",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(cluster_config)
            return handle.name

    def _host_mount_path(self) -> str:
        if platform.system().lower() == "linux":
            try:
                Path("/mnt").mkdir(parents=True, exist_ok=True)
                return "/mnt"
            except OSError:
                pass

        fallback = Path(tempfile.gettempdir()) / f"kind-mnt-{self.cluster_name}"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)
