"""Kubernetes manifests for ws-synth: model downloads, the vLLM server Job + Service, and workers.

Prints a JSON manifest list (valid YAML) for ``kubectl apply -f -``. Manifests follow the Vega worker
rules: campaign labels on every object and pod template, GPUs only as ``amd.com/gpu`` requests equal
to limits, node choice by ``nodeSelector`` hostname, unprivileged pods, ``backoffLimit: 0``,
``restartPolicy: Never`` and ``ttlSecondsAfterFinished``. Node hostnames are arguments, never
defaults, so none land in git.

    python -m d25.vega.data.synth.k8s job --name d25-vega-synth-x --node <hostname> --script run.sh
    python -m d25.vega.data.synth.k8s server --name d25-vega-synth-llm --node <hostname> --gpus 8 \
        --script serve.sh --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAMESPACE = "semantic-router"
IMAGE = "docker.io/vllm/vllm-openai-rocm:v0.31.0"
CPU_PER_GPU = 14
MEMORY_GI_PER_GPU = 120
WS = "synth"


def labels(role: str | None = None) -> dict[str, str]:
    meta = {
        "app.kubernetes.io/part-of": "decision-2.5",
        "d25/campaign": "vega",
        "d25/ws": WS,
    }
    if role:
        meta["d25/role"] = role
    return meta


def job(
    name: str,
    node: str,
    script: str,
    *,
    gpus: int = 0,
    cpu: int | None = None,
    memory_gi: int | None = None,
    image: str = IMAGE,
    env: dict[str, str] | None = None,
    readonly: list[tuple[str, str]] | None = None,
    shm_gi: int = 64,
    role: str | None = None,
    port: int | None = None,
) -> dict:
    """One-pod Job. ``readonly`` mounts prior host paths (type Directory) read-only."""
    if not name.startswith(f"d25-vega-{WS}-"):
        raise ValueError(f"Job names must start with d25-vega-{WS}-")
    if not 0 <= gpus <= 8:
        raise ValueError("gpus must be 0..8")
    cpu = cpu or (CPU_PER_GPU * gpus if gpus else 8)
    memory_gi = memory_gi or (MEMORY_GI_PER_GPU * gpus if gpus else 32)
    resources = {"cpu": str(cpu), "memory": f"{memory_gi}Gi"}
    if gpus:
        resources["amd.com/gpu"] = str(gpus)
    environment = [
        {"name": "HF_HOME", "value": "/data/d25/shared/hf-home"},
        {"name": "HF_HUB_DISABLE_XET", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}},
        },
    ]
    environment += [{"name": k, "value": v} for k, v in (env or {}).items()]
    volumes = [
        {"name": "d25", "hostPath": {"path": "/data/d25", "type": "DirectoryOrCreate"}},
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": f"{shm_gi}Gi"}},
    ]
    mounts = [
        {"name": "d25", "mountPath": "/data/d25"},
        {"name": "shm", "mountPath": "/dev/shm"},
    ]
    for i, (host, path) in enumerate(readonly or []):
        volumes.append(
            {"name": f"ro{i}", "hostPath": {"path": host, "type": "Directory"}}
        )
        mounts.append({"name": f"ro{i}", "mountPath": path, "readOnly": True})
    container = {
        "name": "main",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["bash", "-lc"],
        "args": [script],
        "env": environment,
        "resources": {"requests": resources, "limits": dict(resources)},
        "securityContext": {
            "privileged": False,
            "capabilities": {"add": ["SYS_PTRACE"]},
            "seccompProfile": {"type": "Unconfined"},
        },
        "volumeMounts": mounts,
    }
    if port:
        container["ports"] = [{"name": "http", "containerPort": port}]
        container["readinessProbe"] = {
            "httpGet": {"path": "/health", "port": port},
            "periodSeconds": 20,
            "failureThreshold": 3,
        }
    meta = labels(role)
    pod_meta = dict(meta, **{"d25/job": name})
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": meta},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {"labels": pod_meta},
                "spec": {
                    "restartPolicy": "Never",
                    "enableServiceLinks": False,
                    "terminationGracePeriodSeconds": 60,
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "containers": [container],
                    "volumes": volumes,
                },
            },
        },
    }


def service(name: str, job_name: str, port: int) -> dict:
    """ClusterIP Service selecting the server Job's pod (``d25/job`` label)."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": labels("llm")},
        "spec": {
            "type": "ClusterIP",
            "selector": {"d25/ws": WS, "d25/job": job_name},
            "ports": [{"name": "http", "port": port, "targetPort": port}],
        },
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=("job", "server"))
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--node", required=True, help="kubernetes.io/hostname of the target node"
    )
    ap.add_argument(
        "--script", required=True, help="bash script file run in the container"
    )
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--memory-gi", type=int)
    ap.add_argument("--shm-gi", type=int, default=64)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--role")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument(
        "--readonly",
        action="append",
        default=[],
        help="HOST_PATH:MOUNT_PATH, repeatable",
    )
    a = ap.parse_args(argv)
    env = dict(item.split("=", 1) for item in a.env)
    readonly = [tuple(item.split(":", 1)) for item in a.readonly]
    script = Path(a.script).read_text()
    if a.kind == "server":
        manifests = [
            job(
                a.name,
                a.node,
                script,
                gpus=a.gpus,
                cpu=a.cpu,
                memory_gi=a.memory_gi,
                image=a.image,
                env=env,
                readonly=readonly,
                shm_gi=a.shm_gi,
                role="llm",
                port=a.port,
            ),
            service(a.name, a.name, a.port),
        ]
    else:
        manifests = [
            job(
                a.name,
                a.node,
                script,
                gpus=a.gpus,
                cpu=a.cpu,
                memory_gi=a.memory_gi,
                image=a.image,
                env=env,
                readonly=readonly,
                shm_gi=a.shm_gi,
                role=a.role,
            )
        ]
    json.dump(
        {"apiVersion": "v1", "kind": "List", "items": manifests}, sys.stdout, indent=1
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
