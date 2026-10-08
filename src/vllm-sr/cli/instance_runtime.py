"""Canonical local-container data plane behind the instance control socket.

The manifest is written by ``serve``; requests may select only a mode and an
existing canonical deployment. No client-supplied image, command or URL enters
this owner. Dashboard, storage and external inference backends are never stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from http import HTTPStatus
from pathlib import Path

import yaml

from cli.container_gpu_isolation import router_runtime_env
from cli.container_management_listener import resolve_managed_management_listener
from cli.container_start import container_start_vllm_sr
from cli.engine_container import (
    EngineRequest,
    PackageMounts,
    check_device,
    engine_command,
)
from cli.management_credential import stack_management_credential
from cli.parser import parse_user_config
from cli.runtime_config_lock import acquire_runtime_config_lock
from cli.runtime_lifecycle_lock import acquire_runtime_lifecycle_lock
from cli.runtime_stack import RuntimeStackLayout


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def bounded_request(url, body=None, headers=None, timeout=30):
    request = urllib.request.Request(url, data=body, headers=headers or {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as response_error:
        response = response_error
    with response:
        data = response.read((4 << 20) + 1)
        if len(data) > 4 << 20:
            raise ValueError("Runtime response exceeds the response limit")
        return response.status, data


def copy_private(source, target):
    data = Path(source).read_bytes()
    fd, temporary = tempfile.mkstemp(dir=Path(target).parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def restore_file(source, target):
    previous = Path(target).stat()
    fd, temporary = tempfile.mkstemp(dir=Path(target).parent)
    try:
        os.fchmod(fd, previous.st_mode & 0o777)
        os.fchown(fd, previous.st_uid, previous.st_gid)
        with os.fdopen(fd, "wb") as stream:
            stream.write(Path(source).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ContainerInstanceBackend:
    def __init__(self, manifest: dict, directory: Path):
        self.manifest, self.directory = manifest, directory
        self.layout = RuntimeStackLayout(**manifest["layout"])
        self.runtime = manifest["runtime"]
        self.engine_name = f"{self.layout.stack_name}-systemone-engine"
        self.locks = None

    def command(self, *args, required=True):
        try:
            result = subprocess.run(
                [self.runtime, *args],
                capture_output=True,
                timeout=5 if args[0] == "inspect" else 120,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Container runtime operation timed out") from error
        if required and result.returncode:
            raise RuntimeError("Container runtime operation failed")
        return result

    def inspect(self, name):
        result = self.command("inspect", name, required=False)
        return json.loads(result.stdout)[0] if result.returncode == 0 else None

    def observe(self):
        router, engine = (
            self.inspect(self.layout.router_container_name),
            self.inspect(self.engine_name),
        )

        def running(item):
            return bool(item and item["State"]["Running"])

        mode = "unknown"
        if running(router) != running(engine):
            mode = "router" if running(router) else "engine"
        labels = (engine or {}).get("Config", {}).get("Labels", {}) or {}
        deployment, model = None, None
        if mode == "engine":
            deployment, model = (
                labels.get("vllm-sr.deployment"),
                labels.get("vllm-sr.model"),
            )
        elif mode == "router":
            checkpoint = self.directory / "router-ready.yaml"
            if checkpoint.exists():
                document = yaml.safe_load(checkpoint.read_text()) or {}
                if not isinstance(document, dict):
                    document = {}
                catalog = (document.get("global") or {}).get("model_catalog") or {}
                binding = (catalog.get("system") or {}).get("decision_model") or {}
                deployment = (
                    binding.get("deployment") if isinstance(binding, dict) else None
                )
                model = ((catalog.get("deployments") or {}).get(deployment) or {}).get(
                    "artifact"
                )
        return {
            "observed_mode": mode,
            "active_deployment": deployment,
            "model": model,
        }

    def acquire(self):
        if self.locks is not None:
            return
        stack = ExitStack()
        try:
            stack.enter_context(
                acquire_runtime_lifecycle_lock(
                    runtime=self.runtime, stack_name=self.layout.stack_name
                )
            )
            stack.enter_context(
                acquire_runtime_config_lock(
                    runtime_config_path=self.manifest["config"],
                    state_root_dir=self.manifest["state_root"],
                    stack_name=self.layout.stack_name,
                )
            )
        except BaseException:
            stack.close()
            raise
        self.locks = stack

    def release(self):
        if self.locks:
            self.locks.close()
            self.locks = None

    def prepare(self, mode, deployment, operation):
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            updated = json.loads(manifest_path.read_text())
            if (
                updated["layout"] != self.manifest["layout"]
                or updated["runtime"] != self.runtime
            ):
                raise ValueError("Instance manifest belongs to a different stack")
            self.manifest = updated
        self.acquire()
        try:
            config = parse_user_config(self.manifest["config"], log_summary=False)
            resource = (
                (config.global_ or {})
                .get("model_catalog", {})
                .get("deployments", {})
                .get(deployment)
            )
            if mode == "engine" and (
                not resource or resource.get("provider") != "model_runtime"
            ):
                raise ValueError(
                    "Engine requires a configured model_runtime deployment"
                )
            if mode == "engine" and resource.get("endpoint"):
                raise ValueError(
                    "An externally owned engine must be managed by its owner"
                )
            if mode == "engine" and not resource.get("artifact"):
                raise ValueError("Engine deployment requires an artifact")
            if mode == "engine":
                env = self.manifest.get("env", {})
                platform = env.get(
                    "VLLM_SR_PLATFORM", env.get("DASHBOARD_PLATFORM", "")
                )
                check_device(resource.get("device", "auto"), platform)
            observed = self.observe()
            if observed["observed_mode"] == "unknown":
                raise ValueError(
                    "Instance ownership is ambiguous; inspect its containers"
                )
            suffix = hashlib.sha256(operation.encode()).hexdigest()[:16]
            files = []
            checkpoint = self.directory / "router-ready.yaml"
            if observed["observed_mode"] == "router" and checkpoint.exists():
                saved = self.directory / f"router-{suffix}.yaml"
                copy_private(checkpoint, saved)
                files.append({"source": str(saved), "target": self.manifest["config"]})
            envoy = (
                Path(self.manifest.get("state_root", self.directory))
                / ".vllm-sr"
                / "envoy.yaml"
            )
            if envoy.exists():
                saved = self.directory / f"envoy-{suffix}.yaml"
                copy_private(envoy, saved)
                files.append({"source": str(saved), "target": str(envoy)})
            old = []
            for name in (
                self.layout.router_container_name,
                self.layout.envoy_container_name,
                self.engine_name,
            ):
                item = self.inspect(name)
                if item:
                    old.append(
                        {
                            "name": name,
                            "id": item["Id"],
                            "backup": f"{name}-rollback-{suffix}",
                            "running": item["State"]["Running"],
                        }
                    )
            return {
                "mode": mode,
                "deployment": deployment,
                "resource": resource,
                "previous_mode": observed["observed_mode"],
                "previous_deployment": observed.get("active_deployment"),
                "old": old,
                "engine_models": f"engine-{suffix}.yaml",
                "operation": operation,
                "files": files,
            }
        except BaseException:
            self.release()
            raise

    def activate(self, plan, progress):
        progress("stopping")
        for item in plan["old"]:
            # Identity, not a configuration digest, establishes transaction ownership.
            current = self.inspect(item["name"])
            if not current or current["Id"] != item["id"]:
                raise RuntimeError("Instance container changed before cutover")
            if item["running"]:
                self.command("stop", "--time", "30", item["id"])
            self.command("rename", item["id"], item["backup"])
        progress("starting")
        if plan["mode"] == "engine":
            self.start_engine(plan)
        else:
            config = parse_user_config(self.manifest["config"], log_summary=False)
            code, _, _ = container_start_vllm_sr(
                self.manifest["source"],
                {
                    **self.manifest["env"],
                    "VLLM_SR_INSTANCE_OPERATION": plan["operation"],
                },
                [
                    listener.model_dump(exclude_none=True)
                    for listener in config.listeners
                ],
                router_image=self.manifest["router_image"],
                envoy_image=self.manifest.get("envoy_image"),
                dashboard_image=self.manifest["dashboard_image"],
                pull_policy="never",
                stack_layout=self.layout,
                state_root_dir=self.manifest["state_root"],
                runtime_config_file=self.manifest["config"],
                gateway=self.manifest["gateway"],
                services=("router", "envoy"),
            )
            if code:
                raise RuntimeError("Router startup failed")
        progress("verifying")
        self.wait_ready(plan["mode"])
        if plan["mode"] == "router":
            self.checkpoint()
        self.release()

    def checkpoint(self):
        copy_private(self.manifest["config"], self.directory / "router-ready.yaml")

    def start_engine(self, plan):
        resource = plan["resource"]
        packages = PackageMounts()
        artifact = resource["artifact"]
        model = packages.container_path(artifact) or artifact
        model_spec = {
            "model": model,
            "name": plan["deployment"],
            "device": resource.get("device", "auto"),
            "profile": resource.get("profile", "exact"),
        }
        if resource.get("revision"):
            model_spec["revision"] = resource["revision"]
        models = self.directory / plan["engine_models"]
        models.write_text(yaml.safe_dump({"models": [model_spec]}))
        os.chmod(models, 0o600)
        platform = self.manifest["env"].get(
            "VLLM_SR_PLATFORM", self.manifest["env"].get("DASHBOARD_PLATFORM", "")
        )
        request = EngineRequest(
            (),
            str(models),
            None,
            model_spec["device"],
            model_spec["profile"],
            "127.0.0.1",
            8100,
            None,
            platform,
        )
        cache = Path(self.manifest["models_dir"])
        command = engine_command(
            self.runtime,
            self.manifest["router_image"],
            request,
            cache_dir=cache,
            packages=packages,
            models_file=models,
        )
        command[command.index("--rm")] = "--detach"
        command[command.index("--name") + 1] = self.engine_name
        port_flag = "-p" if "-p" in command else "--publish"
        command[command.index(port_flag) + 1] = "127.0.0.1::8100"
        extras = [
            "--restart",
            "unless-stopped",
            "--network",
            self.layout.network_name,
            "--label",
            f"vllm-sr.deployment={plan['deployment']}",
            "--label",
            f"vllm-sr.model={artifact}",
            "--label",
            f"vllm-sr.operation={plan.get('operation', '')}",
        ]
        placement = {
            key: value
            for key, value in self.manifest["env"].items()
            if key
            in {"ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}
        }
        for key, value in router_runtime_env(placement, platform).items():
            extras += ["-e", f"{key}={value}"]
        command[2:2] = extras
        result = subprocess.run(command, capture_output=True, timeout=120, check=False)
        if result.returncode:
            raise RuntimeError("Engine startup failed")

    def endpoint(self, mode):
        if mode == "engine":
            item = self.inspect(self.engine_name)
            ports = (item or {}).get("NetworkSettings", {}).get("Ports", {}).get(
                "8100/tcp"
            ) or []
            if len(ports) != 1 or ports[0]["HostIp"] != "127.0.0.1":
                raise RuntimeError("Engine has no private managed listener")
            return f"http://127.0.0.1:{int(ports[0]['HostPort'])}", {}
        config = parse_user_config(self.manifest["config"], log_summary=False)
        listener = resolve_managed_management_listener(config, self.layout)
        token = stack_management_credential(
            self.manifest["state_root"], stack_layout=self.layout
        )
        return f"http://127.0.0.1:{listener['host_port']}", {
            "Authorization": f"Bearer {token}"
        }

    def wait_ready(self, mode):
        deadline = time.monotonic() + self.manifest.get("startup_timeout", 600)
        while time.monotonic() < deadline:
            try:
                base, headers = self.endpoint(mode)
                code, _ = bounded_request(base + "/health", headers=headers, timeout=3)
                if code == HTTPStatus.OK:
                    if mode == "engine":
                        models_code, models_body = bounded_request(
                            base + "/v1/models", headers=headers, timeout=3
                        )
                        cards = json.loads(models_body).get("data", [])
                        if (
                            models_code != HTTPStatus.OK
                            or not cards
                            or not all(
                                card.get("ready")
                                and "decisions" in card.get("surfaces", [])
                                for card in cards
                            )
                        ):
                            raise RuntimeError(
                                "Engine model does not advertise a ready SystemOne surface"
                            )
                    return
            except (OSError, ValueError, RuntimeError):
                pass
            time.sleep(1)
        raise RuntimeError("Instance readiness deadline expired")

    def rollback(self, plan):
        self.acquire()
        try:
            old_ids = {item["id"] for item in plan["old"]}
            for name in (
                self.layout.router_container_name,
                self.layout.envoy_container_name,
                self.engine_name,
            ):
                current = self.inspect(name)
                if current and current["Id"] not in old_ids:
                    config = current.get("Config", {})
                    owner = (config.get("Labels") or {}).get("vllm-sr.operation")
                    marker = f"VLLM_SR_INSTANCE_OPERATION={plan['operation']}"
                    if owner != plan["operation"] and marker not in (
                        config.get("Env") or []
                    ):
                        raise RuntimeError(
                            "A replacement container belongs to another operation"
                        )
                    self.command("rm", "--force", current["Id"])
            if plan.get("files"):
                copy_private(
                    self.manifest["config"],
                    self.directory
                    / f"failed-{hashlib.sha256(plan['operation'].encode()).hexdigest()[:16]}.yaml",
                )
                for saved in plan["files"]:
                    restore_file(saved["source"], saved["target"])
            for item in plan["old"]:
                current = self.inspect(item["id"])
                if not current:
                    raise RuntimeError("Retained rollback container is missing")
                if current["Name"].lstrip("/") != item["name"]:
                    self.command("rename", item["id"], item["name"])
                if item["running"]:
                    self.command("start", item["id"])
            self.wait_ready(plan["previous_mode"])
        finally:
            self.release()

    def native(self, path, payload=None):
        mode = self.observe()["observed_mode"]
        if mode not in {"router", "engine"}:
            raise RuntimeError("No ready data plane")
        base, headers = self.endpoint(mode)
        headers["Content-Type"] = "application/json"
        if path == "/models":
            target = (
                "/v1/models"
                if mode == "engine"
                else "/api/v1/diagnostics/models/systemone"
            )
            return bounded_request(base + target, headers=headers)
        deployment = payload["deployment"]
        body = payload["request"]
        if mode == "engine":
            observed = self.observe()
            active = observed.get("active_deployment")
            if deployment != active:
                return (
                    503,
                    b'{"error":"The requested deployment is not active in Engine mode"}',
                )
            if payload.get("expected_artifact") and payload[
                "expected_artifact"
            ] != observed.get("model"):
                return 409, b'{"error":"The published model requires deployment"}'
            body = {**body, "model": deployment}
            target = "/v1/systemone"
        else:
            body = {"deployment": deployment, "request": body}
            if payload.get("expected_artifact"):
                body["expected_artifact"] = payload["expected_artifact"]
            target = "/api/v1/diagnostics/models/systemone"
        return bounded_request(base + target, json.dumps(body).encode(), headers)
