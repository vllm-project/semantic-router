"""Attach a persistent instance controller to a successfully served local stack."""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

from cli.commands.runtime_paths import resolve_state_root_dir
from cli.container_runtime import get_container_runtime
from cli.container_start_paths import _prepare_runtime_paths
from cli.instance_paths import (
    instance_control_directory,
    prepare_instance_control_directory,
)
from cli.instance_state import write_private_json
from cli.runtime_stack import resolve_runtime_stack


def controller_request(directory, path, payload=None):
    connection = http.client.HTTPConnection("localhost", timeout=35)
    connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.sock.settimeout(35)
    try:
        connection.sock.connect(str(Path(directory) / "socket" / "control.sock"))
        body = json.dumps(payload).encode() if payload is not None else None
        connection.request(
            "POST" if body else "GET", path, body, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        data = response.read((4 << 20) + 1)
        if response.status not in {200, 202} or len(data) > 4 << 20:
            raise ValueError("Instance controller rejected request")
        return json.loads(data)
    finally:
        connection.close()


def instance_directory(source, env=None):
    root = resolve_state_root_dir(source, env)
    return instance_control_directory(root, resolve_runtime_stack().stack_name)


def ensure_controller(directory):
    try:
        return controller_request(directory, "/status")
    except (OSError, ValueError, http.client.HTTPException):
        pass
    log = directory / "controller.log"
    descriptor = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cli.instance_controller",
                "--directory",
                str(directory),
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            return controller_request(directory, "/status")
        except (OSError, ValueError, http.client.HTTPException):
            if process.poll() is not None:
                break
            time.sleep(0.1)
    raise RuntimeError("Instance controller failed to start; inspect its private log")


def attach_controller(source, config, env, gateway, startup_timeout):
    layout = resolve_runtime_stack()
    state_root = str(resolve_state_root_dir(source, env))
    _, paths, _ = _prepare_runtime_paths(source, config, state_root, layout)
    directory = prepare_instance_control_directory(state_root, layout.stack_name)
    runtime = get_container_runtime()
    images = {}
    for role in ("router", "envoy", "dashboard"):
        result = subprocess.run(
            [
                runtime,
                "inspect",
                "--format",
                "{{.Image}}",
                layout.service_container_name(role),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            images[f"{role}_image"] = result.stdout.strip()
    if "router_image" not in images or "dashboard_image" not in images:
        return  # Minimal stacks and external Dashboard owners have no mode controller.
    # Lower-level image rollout callers may not have the original CLI flags.
    # Preserve accelerator placement from the exact running Router, without
    # copying unrelated image defaults or inference credentials to Engine.
    runtime_env = dict(env or {})
    inspected_env = subprocess.run(
        [
            runtime,
            "inspect",
            "--format",
            "{{json .Config.Env}}",
            layout.router_container_name,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    for item in json.loads(inspected_env.stdout) or []:
        key, _, value = item.partition("=")
        if key in {
            "VLLM_SR_PLATFORM",
            "DASHBOARD_PLATFORM",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        }:
            runtime_env.setdefault(key, value)
    manifest = {
        "layout": asdict(layout),
        "runtime": runtime,
        "source": str(Path(source).absolute()),
        "config": paths["effective_config_path"],
        "state_root": state_root,
        "models_dir": paths["models_dir"],
        "env": runtime_env,
        "gateway": gateway,
        "startup_timeout": startup_timeout,
        **images,
    }
    write_private_json(directory / "manifest.json", manifest)
    ensure_controller(directory)


def stop_managed_controller():
    """Resolve this stack's actual state mount before stopping its control plane."""
    layout = resolve_runtime_stack()
    runtime = get_container_runtime()
    inspected = subprocess.run(
        [runtime, "inspect", layout.dashboard_container_name],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if inspected.returncode:
        return
    container = json.loads(inspected.stdout)[0]
    for mount in container.get("Mounts", []):
        if mount.get("Destination") == "/app/instance-control":
            directory = Path(mount["Source"]).parent
            if not (directory / "manifest.json").exists():
                return
            with suppress(FileNotFoundError, ConnectionRefusedError):
                controller_request(directory, "/shutdown", {"stop": True})
            return
