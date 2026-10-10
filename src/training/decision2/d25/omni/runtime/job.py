"""Kubernetes Job for the runtime evidence chain: one GPU, ``run_all.sh`` with staged code.

    python -m d25.omni.runtime.job --tag rt01 --hostname <node hostname> [--steps "latency smoke"] > job.json

Mounts (hostPath): ``/data/d25/omni/runtime`` read-write; the inits, suite and results directories and the
flash-linear-attention overlay read-only. No Hub access (``HF_HUB_OFFLINE``), so no token is mounted.
"""

from __future__ import annotations

import argparse
import json

LABELS = {
    "app.kubernetes.io/part-of": "decision-2.5",
    "d25/campaign": "omni",
    "d25/ws": "runtime",
}
IMAGE = "docker.io/vllm/vllm-openai-rocm:v0.31.0"
ROOT = "/data/d25/omni/runtime"
FLA = "/data/d25/shared/pylib/fla052"
MOUNTS = (
    ("runtime", ROOT, False),
    ("inits", "/data/d25/omni/inits", True),
    ("suite", "/data/d25/omni/suite", True),
    ("results", "/data/d25/omni/results", True),
    ("fla", FLA, True),
)


def manifest(args: argparse.Namespace) -> dict:
    code = f"{ROOT}/src/{args.tag}/src/training/decision2"
    env = {
        "HF_HOME": "/tmp/hf-home",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRITON_CACHE_DIR": f"{ROOT}/triton-cache",
        "TRITON_CACHE_AUTOTUNING": "1",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{FLA}:{code}:{ROOT}/kit",
    }
    resources = {"cpu": str(args.cpu), "memory": f"{args.memory_gi}Gi"}
    if args.gpus:
        resources["amd.com/gpu"] = str(args.gpus)
    pod = {
        "restartPolicy": "Never",
        "enableServiceLinks": False,
        "terminationGracePeriodSeconds": 30,
        "nodeSelector": {"kubernetes.io/hostname": args.hostname},
        "containers": [
            {
                "name": "runtime",
                "image": IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-c"],
                "args": [
                    f"bash {code}/d25/omni/runtime/run_all.sh {args.tag} {args.steps}".strip()
                ],
                "env": [{"name": k, "value": v} for k, v in env.items()],
                "resources": {"requests": resources, "limits": dict(resources)},
                "securityContext": {"privileged": False},
                "volumeMounts": [
                    {"name": name, "mountPath": path, "readOnly": ro}
                    for name, path, ro in MOUNTS
                ]
                + [{"name": "dshm", "mountPath": "/dev/shm"}],
            }
        ],
        "volumes": [
            {"name": name, "hostPath": {"path": path, "type": "Directory"}}
            for name, path, _ in MOUNTS
        ]
        + [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}}],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": args.name,
            "namespace": "semantic-router",
            "labels": dict(LABELS),
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": int(args.deadline_hours * 3600),
            "ttlSecondsAfterFinished": 6 * 3600,
            "template": {"metadata": {"labels": dict(LABELS)}, "spec": pod},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hostname", required=True)
    ap.add_argument("--name", default="d25-omni-runtime-evidence")
    ap.add_argument("--steps", default="")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpu", type=int, default=16)
    ap.add_argument("--memory-gi", type=int, default=160)
    ap.add_argument("--deadline-hours", type=float, default=3.0)
    args = ap.parse_args()
    print(json.dumps(manifest(args), indent=1))


if __name__ == "__main__":
    main()
