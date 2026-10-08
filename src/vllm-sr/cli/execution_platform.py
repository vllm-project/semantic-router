"""Resolve an execution backend on the deployment target, before config mutation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cli.container_runtime import get_container_runtime
from cli.container_gpu_isolation import AMD_ROUTER_VISIBLE_DEVICES_ENV

PLATFORMS = ("auto", "cpu", "cuda", "rocm")
RENDER_NODE_GLOB = "renderD*/device/vendor"  # codespell:ignore renderd
GPU_RESOURCES = {"cuda": "nvidia.com/gpu", "rocm": "amd.com/gpu"}


def _output(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            f"Cannot inspect execution target with {command[0]}"
        ) from error
    if result.returncode:
        raise ValueError(
            f"Cannot inspect execution target with {command[0]}; choose an explicit platform or check target access"
        )
    return result.stdout


def _cluster_devices(context=None):
    command = ["kubectl"]
    if context:
        command += ["--context", context]
    nodes = json.loads(_output([*command, "get", "nodes", "-o", "json"]))
    counts = dict.fromkeys(GPU_RESOURCES, 0)
    for node in nodes.get("items", []):
        if node.get("spec", {}).get("unschedulable"):
            continue
        status = node.get("status", {})
        if not any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in status.get("conditions", [])
        ):
            continue
        for platform, resource in GPU_RESOURCES.items():
            counts[platform] = max(
                counts[platform], int(status.get("allocatable", {}).get(resource, 0))
            )
    return {platform: tuple(range(count)) for platform, count in counts.items()}


def _local_container_host():
    if sys.platform != "linux":
        return False
    runtime = get_container_runtime()
    endpoint = (
        os.getenv("DOCKER_HOST", "")
        if runtime == "docker"
        else os.getenv("CONTAINER_HOST", "")
    )
    if runtime == "docker" and not endpoint:
        endpoint = json.loads(
            _output(
                [
                    runtime,
                    "context",
                    "inspect",
                    "--format",
                    "{{json .Endpoints.docker.Host}}",
                ]
            )
        )
    if endpoint and not endpoint.startswith("unix://"):
        raise ValueError(
            "Automatic device discovery requires a local container host; select an explicit platform for a remote daemon"
        )
    info = json.loads(_output([runtime, "info", "--format", "{{json .}}"]))
    # Docker Desktop/Podman machines run a VM. Host GPU nodes are not its hardware.
    system = str(info.get("OperatingSystem", ""))
    return "docker desktop" not in system.lower()


def _host_devices():
    devices = {"cuda": (), "rocm": ()}
    if not _local_container_host():
        return devices
    if shutil.which("nvidia-smi"):
        try:
            result = _output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
            )
            devices["cuda"] = tuple(
                int(line.strip()) for line in result.splitlines() if line.strip()
            )
        except ValueError:
            pass
    if Path("/dev/kfd").exists():
        cards = [
            path
            for path in Path("/sys/class/drm").glob(RENDER_NODE_GLOB)
            if path.read_text().strip().lower() == "0x1002"
        ]
        devices["rocm"] = tuple(range(len(cards)))
    return devices


def visible_device_ids(platform, available):
    """Host IDs in runtime-visible order; refuse UUID masks instead of guessing."""
    masks = list(runtime_gpu_environment(platform).items())
    if len(masks) > 1:
        raise ValueError(
            "Set only one ROCm visibility mask so host device indices are unambiguous"
        )
    if not masks:
        return tuple(available)
    name, value = masks[0]
    if value in ("", "-1"):
        return ()
    pieces = value.split(",")
    if any(not part.isdecimal() for part in pieces):
        raise ValueError(f"{name} must use numeric host indices with --device-ids")
    ids = tuple(int(part) for part in pieces)
    if len(set(ids)) != len(ids) or not set(ids).issubset(available):
        raise ValueError(f"{name} names duplicate or unavailable GPU indices")
    return ids


def resolve_execution_platform(platform=None, *, target="docker", context=None):
    requested = (
        platform
        if platform is not None
        else os.getenv("VLLM_SR_PLATFORM", os.getenv("DASHBOARD_PLATFORM", "auto"))
    )
    if requested not in PLATFORMS:
        raise ValueError("--platform must be auto, cpu, cuda or rocm")
    if requested != "auto":
        if target == "docker" and sys.platform != "linux" and requested != "cpu":
            raise ValueError("GPU container platforms require a Linux execution host")
        return requested
    inventory = _cluster_devices(context) if target == "kubernetes" else _host_devices()
    candidates = [
        name
        for name, ids in inventory.items()
        if ids and (target == "kubernetes" or visible_device_ids(name, ids))
    ]
    if len(candidates) > 1:
        raise ValueError(
            "Both CUDA and ROCm are available; choose --platform explicitly"
        )
    return candidates[0] if candidates else "cpu"


def placement_devices(platform, *, target="docker", context=None):
    if platform == "cpu":
        return ()
    inventory = _cluster_devices(context) if target == "kubernetes" else _host_devices()
    ids = inventory[platform]
    return ids if target == "kubernetes" else visible_device_ids(platform, ids)


def device_ordinals(ids, platform, available):
    if not set(ids).issubset(available):
        raise ValueError(
            "--device-ids selects a GPU outside the available execution devices"
        )
    masked = bool(runtime_gpu_environment(platform))
    return tuple(available.index(value) if masked else value for value in ids)


def runtime_gpu_environment(platform):
    if platform == "rocm" and os.getenv(AMD_ROUTER_VISIBLE_DEVICES_ENV, "").strip():
        if "HIP_VISIBLE_DEVICES" in os.environ:
            raise ValueError("Use only one ROCm visibility mask")
        return {
            "ROCR_VISIBLE_DEVICES": os.environ[AMD_ROUTER_VISIBLE_DEVICES_ENV].strip()
        }
    names = (
        ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
        if platform == "rocm"
        else ("CUDA_VISIBLE_DEVICES",) if platform == "cuda" else ()
    )
    return {name: os.environ[name] for name in names if name in os.environ}
