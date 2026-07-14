"""Fail-closed Helm and kubectl inspection helpers for the K8s backend."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from cli.k8s_secret_history import deployment_secret_refs


class K8sResourceInspectionMixin:
    """Read structured cluster state without exposing command output."""

    def _namespace_exists(self) -> bool:
        payload = self._get_json(
            [
                *self._kubectl_base_cmd(),
                "get",
                "namespace",
                self.namespace,
                "-o",
                "json",
            ],
            allow_not_found=True,
        )
        if payload is None:
            return False
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("name") != self.namespace:
            raise RuntimeError("kubectl returned a mismatched namespace")
        return True

    def _helm_release_exists(self) -> bool:
        """Return an exact Helm release state, failing closed on ambiguity."""
        release_pattern = "^" + re.escape(self.release_name).replace(r"\-", "-") + "$"
        try:
            result = subprocess.run(
                [
                    *self._helm_base_cmd(),
                    "list",
                    "--all",
                    "--namespace",
                    self.namespace,
                    "--filter",
                    release_pattern,
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError("Helm release state query failed") from exc
        if result.returncode != 0:
            raise RuntimeError("Helm release state query failed")
        if not result.stdout.strip():
            raise RuntimeError("Helm returned an empty release state")
        try:
            releases = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Helm returned malformed release state JSON") from exc
        if not isinstance(releases, list) or len(releases) > 1:
            raise RuntimeError("Helm returned an unexpected release state")
        if not releases:
            return False
        release = releases[0]
        if not isinstance(release, dict):
            raise RuntimeError("Helm returned a malformed release state")
        name = release.get("name")
        status = release.get("status")
        if name != self.release_name or not isinstance(status, str) or not status:
            raise RuntimeError("Helm returned a mismatched release state")
        return True

    def _get_json(
        self,
        cmd: list[str],
        *,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr if isinstance(result.stderr, str) else ""
            not_found = "notfound" in stderr.replace(" ", "").lower()
            if allow_not_found and not_found:
                return None
            raise RuntimeError("kubectl query failed while inspecting runtime Secrets")
        if not result.stdout.strip():
            raise RuntimeError(
                "kubectl returned an empty response while inspecting Secrets"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("kubectl returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("kubectl returned an unexpected JSON document")
        return payload

    @staticmethod
    def _json_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        if payload is None:
            return []
        items = payload.get("items")
        if not isinstance(items, list):
            raise RuntimeError("kubectl returned a malformed resource list")
        if any(not isinstance(item, dict) for item in items):
            raise RuntimeError("kubectl returned malformed resource list items")
        return [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _deployment_secret_refs(deployment: dict[str, Any]) -> list[str]:
        try:
            return sorted(deployment_secret_refs(deployment))
        except ValueError as exc:
            raise RuntimeError("kubectl returned a malformed Deployment") from exc
