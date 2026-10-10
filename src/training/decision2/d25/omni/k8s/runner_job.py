"""Kubernetes Job manifest for an Omni node runner (``d25.omni.runner``), printed as JSON.

    python -m d25.omni.k8s.runner_job --node 03 --hostname <node hostname> [--gpus 8] > job.json

The hostname is passed in by the caller and never stored in git. The pod runs the code tag that
``/data/d25/omni/src/current`` points to, restarts on failure and resumes the running item.
"""

from __future__ import annotations

import argparse
import json

IMAGE = "docker.io/vllm/vllm-openai-rocm:v0.31.0"
BASE_MODEL_HOSTPATH = "/data/decision20-20260926/models/Qwen3.8-27B"
LABELS = {
    "app.kubernetes.io/part-of": "decision-2.5",
    "d25/campaign": "omni",
    "d25/ws": "runner",
}


def manifest(
    node: str, hostname: str, gpus: int, cpu: int, mem_gi: int, shm_gi: int
) -> dict:
    env = {
        "HF_HOME": "/data/d25/omni/hf-home",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRITON_CACHE_DIR": "/data/d25/omni/triton-cache",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_MIN_NCHANNELS": "64",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "PYTHONUNBUFFERED": "1",
        "D25_BASE_DIR": "/models/qwen38",
        "PYTHONPATH": "/data/d25/omni/src/current/src/training/decision2",
    }
    env_list = [{"name": k, "value": v} for k, v in env.items()]
    env_list.append(
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}},
        }
    )
    resources = {"cpu": str(cpu), "memory": f"{mem_gi}Gi"}
    if gpus:
        resources["amd.com/gpu"] = str(gpus)
    pod = {
        "restartPolicy": "OnFailure",
        "enableServiceLinks": False,
        "terminationGracePeriodSeconds": 120,
        "nodeSelector": {"kubernetes.io/hostname": hostname},
        "containers": [
            {
                "name": "runner",
                "image": IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-c"],
                "args": [
                    "mkdir -p /data/d25/omni/triton-cache /data/d25/omni/hf-home && "
                    f"exec python -m d25.omni.runner --node {node} --gpus {gpus}"
                ],
                "env": env_list,
                "resources": {"requests": resources, "limits": resources},
                "securityContext": {
                    "privileged": False,
                    "capabilities": {"add": ["SYS_PTRACE"]},
                    "seccompProfile": {"type": "Unconfined"},
                },
                "volumeMounts": [
                    {"name": "omni", "mountPath": "/data/d25/omni"},
                    {
                        "name": "shared",
                        "mountPath": "/data/d25/shared",
                        "readOnly": True,
                    },
                    {"name": "base", "mountPath": "/models/qwen38", "readOnly": True},
                    {"name": "dshm", "mountPath": "/dev/shm"},
                ],
            }
        ],
        "volumes": [
            {
                "name": "omni",
                "hostPath": {"path": "/data/d25/omni", "type": "DirectoryOrCreate"},
            },
            {
                "name": "shared",
                "hostPath": {"path": "/data/d25/shared", "type": "Directory"},
            },
            {
                "name": "base",
                "hostPath": {"path": BASE_MODEL_HOSTPATH, "type": "Directory"},
            },
            {
                "name": "dshm",
                "emptyDir": {"medium": "Memory", "sizeLimit": f"{shm_gi}Gi"},
            },
        ],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"d25-omni-runner-{node}",
            "namespace": "semantic-router",
            "labels": LABELS,
        },
        "spec": {
            "backoffLimit": 30,
            "template": {"metadata": {"labels": LABELS}, "spec": pod},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--node", required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=100)
    parser.add_argument("--mem-gi", type=int, default=900)
    parser.add_argument("--shm-gi", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            manifest(
                args.node, args.hostname, args.gpus, args.cpu, args.mem_gi, args.shm_gi
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
