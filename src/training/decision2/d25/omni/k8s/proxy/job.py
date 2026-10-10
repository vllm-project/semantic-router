"""CPU Job manifests for the Omni proxy workstream.

    python -m d25.omni.k8s.proxy.job --what build-charts --node <node hostname> --tag <src tag> \
        --cmd 'python -m d25.omni.proxy.build charts ...' > job.json

The node hostname and the source tag are arguments, so no fleet detail lives in git. Jobs follow the
Omni worker rules: no GPU, requests equal limits (at most 8 CPU and 32Gi), ``backoffLimit: 0``,
``restartPolicy: Never``, a one-day TTL, hostPath only under ``/data/d25/omni`` plus the shared HF
cache.
"""

from __future__ import annotations

import argparse
import json
import sys

IMAGE = "docker.io/vllm/vllm-openai-rocm:v0.31.0"
NAMESPACE = "semantic-router"
MAX_CPU = 8
MAX_MEMORY_GI = 32
LABELS = {
    "app.kubernetes.io/part-of": "decision-2.5",
    "d25/campaign": "omni",
    "d25/ws": "proxy",
}
OMNI = "/data/d25/omni"
PYLIB = f"{OMNI}/pylib/proxy"
BROWSERS = f"{OMNI}/pylib/proxy-browsers"
SYSLIB = f"{OMNI}/pylib/proxy-syslib"


def manifest(
    what: str,
    node: str,
    tag: str,
    command: str,
    cpu: int = MAX_CPU,
    memory_gi: int = MAX_MEMORY_GI,
    shm_gi: int = 4,
) -> dict:
    if not 0 < cpu <= MAX_CPU or not 0 < memory_gi <= MAX_MEMORY_GI:
        raise ValueError(
            f"a proxy Job takes at most {MAX_CPU} CPU and {MAX_MEMORY_GI}Gi"
        )
    workdir = f"{OMNI}/src/{tag}/src/training/decision2"
    resources = {"cpu": str(cpu), "memory": f"{memory_gi}Gi"}
    env = [
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}},
        },
        {"name": "HF_HOME", "value": "/data/d25/shared/hf-home"},
        {"name": "HF_HUB_DISABLE_XET", "value": "1"},
        {"name": "PYTHONPATH", "value": f"{workdir}:{PYLIB}"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "PYTHONHASHSEED", "value": "0"},
        {"name": "OMP_NUM_THREADS", "value": "1"},
        {"name": "MPLBACKEND", "value": "Agg"},
        {"name": "MPLCONFIGDIR", "value": "/tmp/mpl"},
        {"name": "PLAYWRIGHT_BROWSERS_PATH", "value": BROWSERS},
        {"name": "D25_PROXY_SYSLIB", "value": SYSLIB},
        {"name": "D25_OMNI_PROXY", "value": f"{OMNI}/proxy"},
        {"name": "D25_WORKERS", "value": str(cpu)},
    ]
    mounts = [
        ("work", f"{OMNI}/proxy", False, "DirectoryOrCreate"),
        ("pylib", f"{OMNI}/pylib", False, "DirectoryOrCreate"),
        ("src", f"{OMNI}/src", True, "DirectoryOrCreate"),
        ("hfhome", "/data/d25/shared/hf-home", False, "DirectoryOrCreate"),
    ]
    container = {
        "name": "main",
        "image": IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "workingDir": workdir,
        "command": ["bash", "-c", command],
        "env": env,
        "resources": {"requests": resources, "limits": dict(resources)},
        "securityContext": {"privileged": False, "allowPrivilegeEscalation": False},
        "volumeMounts": [
            {"name": name, "mountPath": path, **({"readOnly": True} if ro else {})}
            for name, path, ro, _ in mounts
        ]
        + [{"name": "dshm", "mountPath": "/dev/shm"}],
    }
    volumes = [
        {"name": name, "hostPath": {"path": path, "type": kind}}
        for name, path, _, kind in mounts
    ] + [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": f"{shm_gi}Gi"}}]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"d25-omni-proxy-{what}",
            "namespace": NAMESPACE,
            "labels": LABELS,
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {"labels": LABELS},
                "spec": {
                    "restartPolicy": "Never",
                    "enableServiceLinks": False,
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "containers": [container],
                    "volumes": volumes,
                },
            },
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--what", required=True)
    parser.add_argument(
        "--node", required=True, help="kubernetes.io/hostname of an allowed node"
    )
    parser.add_argument(
        "--tag", required=True, help="code tag under /data/d25/omni/src"
    )
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--cpu", type=int, default=MAX_CPU)
    parser.add_argument("--memory-gi", type=int, default=MAX_MEMORY_GI)
    parser.add_argument("--shm-gi", type=int, default=4)
    args = parser.parse_args(argv)
    job = manifest(
        args.what, args.node, args.tag, args.cmd, args.cpu, args.memory_gi, args.shm_gi
    )
    json.dump(job, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
