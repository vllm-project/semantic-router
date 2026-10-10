"""Kubernetes Job manifests for the d3 family, printed as JSON.

    python -m d25.family.k8s runner --lane fam-02a --hostname <node> --gpus 1 --cpu 16 --mem-gi 128
    python -m d25.family.k8s xfer --name data-03 --hostname <node> --dir /data/d25/shared/data/v1
    python -m d25.family.k8s cpu --name edge-build --hostname <node> --tag fam03 --command '...'

``runner``: an Omni node runner (``d25.omni.runner``) working the queue ``/data/d25/omni/queue/<lane>``
with its own GPU count, so several lanes can share one node (the device plugin gives each pod its
own GPUs). ``xfer``: a read-only HTTP server for one host directory, used to copy files between
nodes from a pod on the destination node. ``cpu``: a one-shot CPU Job on a staged code tag; its
working directory is the tag root, so ``python -m`` cannot pick up another tag's modules.
Hostnames are arguments, never stored in git.
"""

from __future__ import annotations

import argparse
import json

from d25.omni.k8s import runner_job

LABELS = {
    "app.kubernetes.io/part-of": "decision-2.5",
    "d25/campaign": "family",
}


def _relabel(job: dict, name: str, ws: str) -> dict:
    labels = {**LABELS, "d25/ws": ws}
    job["metadata"]["name"] = name
    job["metadata"]["labels"] = labels
    job["spec"]["template"]["metadata"]["labels"] = labels
    return job


def runner(
    lane: str, hostname: str, gpus: int, cpu: int, mem_gi: int, shm_gi: int
) -> dict:
    job = runner_job.manifest(lane, hostname, gpus, cpu, mem_gi, shm_gi)
    return _relabel(job, f"d25-fam-runner-{lane}", "runner")


def xfer(name: str, hostname: str, directory: str, hours: float) -> dict:
    pod = {
        "restartPolicy": "Never",
        "enableServiceLinks": False,
        "nodeSelector": {"kubernetes.io/hostname": hostname},
        "containers": [
            {
                "name": "xfer",
                "image": runner_job.IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "command": [
                    "bash",
                    "-c",
                    f"exec timeout {int(hours * 3600)} python3 -m http.server 8080 "
                    "--bind 0.0.0.0 --directory /srv",
                ],
                "ports": [{"containerPort": 8080}],
                "resources": {
                    "requests": {"cpu": "2", "memory": "4Gi"},
                    "limits": {"cpu": "2", "memory": "4Gi"},
                },
                "volumeMounts": [
                    {"name": "src", "mountPath": "/srv", "readOnly": True}
                ],
            }
        ],
        "volumes": [
            {"name": "src", "hostPath": {"path": directory, "type": "Directory"}}
        ],
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "", "namespace": "semantic-router", "labels": {}},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {"metadata": {"labels": {}}, "spec": pod},
        },
    }
    return _relabel(job, f"d25-fam-xfer-{name}", "xfer")


def cpu(
    name: str, hostname: str, tag: str, command: str, cores: int, mem_gi: int
) -> dict:
    """A one-shot CPU Job running ``command`` with the code tag ``tag`` (no GPU; at most 8 CPU / 32Gi)."""
    if cores > 8 or mem_gi > 32:
        raise SystemExit("family CPU Jobs are limited to 8 CPU and 32Gi")
    code = f"/data/d25/omni/src/{tag}"
    resources = {"cpu": str(cores), "memory": f"{mem_gi}Gi"}
    pod = {
        "restartPolicy": "Never",
        "enableServiceLinks": False,
        "nodeSelector": {"kubernetes.io/hostname": hostname},
        "containers": [
            {
                "name": "cpu",
                "image": runner_job.IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-c", command],
                "workingDir": code,
                "env": [
                    {"name": "PYTHONPATH", "value": f"{code}/src/training/decision2"},
                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                    {"name": "HF_HOME", "value": "/data/d25/omni/hf-home"},
                    {"name": "HF_HUB_DISABLE_XET", "value": "1"},
                    {"name": "OMP_NUM_THREADS", "value": str(cores)},
                    {
                        "name": "HF_TOKEN",
                        "valueFrom": {
                            "secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}
                        },
                    },
                ],
                "resources": {"requests": resources, "limits": resources},
                "volumeMounts": [
                    {"name": "omni", "mountPath": "/data/d25/omni"},
                    {
                        "name": "shared",
                        "mountPath": "/data/d25/shared",
                        "readOnly": True,
                    },
                ],
            }
        ],
        "volumes": [
            {
                "name": "omni",
                "hostPath": {"path": "/data/d25/omni", "type": "Directory"},
            },
            {
                "name": "shared",
                "hostPath": {"path": "/data/d25/shared", "type": "Directory"},
            },
        ],
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "", "namespace": "semantic-router", "labels": {}},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {"metadata": {"labels": {}}, "spec": pod},
        },
    }
    return _relabel(job, f"d25-fam-cpu-{name}", "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--name", required=True)
    c.add_argument("--hostname", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--command", required=True)
    c.add_argument("--cpu", type=int, default=8)
    c.add_argument("--mem-gi", type=int, default=32)
    r = sub.add_parser("runner")
    r.add_argument("--lane", required=True)
    r.add_argument("--hostname", required=True)
    r.add_argument("--gpus", type=int, required=True)
    r.add_argument("--cpu", type=int, default=16)
    r.add_argument("--mem-gi", type=int, default=128)
    r.add_argument("--shm-gi", type=int, default=16)
    x = sub.add_parser("xfer")
    x.add_argument("--name", required=True)
    x.add_argument("--hostname", required=True)
    x.add_argument("--dir", required=True)
    x.add_argument("--hours", type=float, default=6.0)
    args = parser.parse_args()
    if args.cmd == "runner":
        job = runner(
            args.lane, args.hostname, args.gpus, args.cpu, args.mem_gi, args.shm_gi
        )
    elif args.cmd == "cpu":
        job = cpu(
            args.name, args.hostname, args.tag, args.command, args.cpu, args.mem_gi
        )
    else:
        job = xfer(args.name, args.hostname, args.dir, args.hours)
    print(json.dumps(job, indent=1))


if __name__ == "__main__":
    main()
