"""Kubernetes Job manifests for the d3 family, printed as JSON.

    python -m d25.family.k8s runner --lane fam-02a --hostname <node> --gpus 1 --cpu 16 --mem-gi 128
    python -m d25.family.k8s xfer --name data-03 --hostname <node> --dir /data/d25/shared/data/v1

``runner``: an Omni node runner (``d25.omni.runner``) working the queue ``/data/d25/omni/queue/<lane>``
with its own GPU count, so several lanes can share one node (the device plugin gives each pod its
own GPUs). ``xfer``: a read-only HTTP server for one host directory, used to copy files between
nodes from a pod on the destination node. Hostnames are arguments, never stored in git.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
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
    else:
        job = xfer(args.name, args.hostname, args.dir, args.hours)
    print(json.dumps(job, indent=1))


if __name__ == "__main__":
    main()
