"""Kubernetes Job manifests for ws-eval work (suite build/staging, suite and row runs).

Prints a JSON manifest (valid YAML) for ``kubectl apply -f -``. The manifests follow the Vega worker
rules: campaign labels on the Job and the pod template, GPUs only as ``amd.com/gpu`` requests equal to
limits, node choice by ``nodeSelector`` hostname, unprivileged pods, ``backoffLimit: 0`` and
``ttlSecondsAfterFinished``. Node hostnames are arguments, never defaults, so none land in git.

    python -m d25.vega.eval.k8s_jobs --name d25-vega-eval-x --node <hostname> --gpus 7 \
        --script run.sh --env KEY=VALUE | kubectl --context vllm-sr apply -f -
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAMESPACE = "semantic-router"
IMAGE = "docker.io/vllm/vllm-openai-rocm:v0.31.0"
CPU_IMAGE = "docker.io/library/python:3.12-slim"
FLA_PYLIB = "/data/d25/shared/pylib/fla052"
CPU_PER_GPU = 14
MEMORY_GI_PER_GPU = 120


def labels(ws: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/part-of": "decision-2.5",
        "d25/campaign": "vega",
        "d25/ws": ws,
    }


def job(
    name: str,
    node: str,
    script: str,
    *,
    ws: str = "eval",
    gpus: int = 0,
    cpu: int | None = None,
    memory_gi: int | None = None,
    image: str | None = None,
    env: dict[str, str] | None = None,
    readonly: list[tuple[str, str]] | None = None,
    shm_gi: int = 64,
    hf_token: bool = True,
) -> dict:
    """One-pod Job. ``readonly`` mounts prior host paths (type Directory) read-only."""
    if gpus < 0 or gpus > 8:
        raise ValueError("gpus must be 0..8")
    cpu = cpu or (max(CPU_PER_GPU * gpus, 8) if gpus else 8)
    memory_gi = memory_gi or (MEMORY_GI_PER_GPU * gpus if gpus else 32)
    image = image or (IMAGE if gpus else CPU_IMAGE)
    resources = {"cpu": str(cpu), "memory": f"{memory_gi}Gi"}
    if gpus:
        resources["amd.com/gpu"] = str(gpus)
    environment = [
        {"name": "HF_HOME", "value": "/data/d25/shared/hf-home"},
        {"name": "HF_HUB_DISABLE_XET", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
    ]
    if hf_token:
        environment.append(
            {
                "name": "HF_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}},
            }
        )
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
    meta = labels(ws)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": meta},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {"labels": meta},
                "spec": {
                    "restartPolicy": "Never",
                    "enableServiceLinks": False,
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "containers": [
                        {
                            "name": "main",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["bash", "-lc"],
                            "args": [script],
                            "env": environment,
                            "resources": {
                                "requests": resources,
                                "limits": dict(resources),
                            },
                            "securityContext": {
                                "privileged": False,
                                "capabilities": {"add": ["SYS_PTRACE"]},
                                "seccompProfile": {"type": "Unconfined"},
                            },
                            "volumeMounts": mounts,
                        }
                    ],
                    "volumes": volumes,
                },
            },
        },
    }


KIT_DIR = "/data/d25/shared/decision-index-kit"
SUITE_DIR = "/data/d25/shared/index-suite-0.3"


def preset_script(
    kind: str,
    tag: str,
    ckpt: str,
    out: str,
    gpus: int,
    rows: list[str],
    triton: str,
    extra: str,
) -> str:
    """Bash for a preset run: ``suite`` (run_suite over the public suite) or ``rows`` (run_rows)."""
    devices = f"0-{gpus - 1}" if gpus > 1 else "0"
    lines = [
        "set -euo pipefail",
        f"export PYTHONPATH={FLA_PYLIB}:/data/d25/vega/src/{tag}",
        f"export TRITON_CACHE_DIR={triton} TRITON_CACHE_AUTOTUNING=1",
        f"cd /data/d25/vega/src/{tag}",
    ]
    if kind == "suite":
        lines.append(
            f"python -m d25.vega.eval.run_suite run --ckpt {ckpt} --kit {KIT_DIR} --suite-dir {SUITE_DIR} "
            f"--out {out} --devices {devices} --cpu-workers {max(4, CPU_PER_GPU * gpus - 4)} {extra}".rstrip()
        )
    elif kind == "rows":
        lines.append(
            f"python -m d25.vega.eval.run_rows run --ckpt {ckpt} --rows {' '.join(rows)} --out {out} "
            f"--devices {devices} --cpu-workers {max(4, CPU_PER_GPU * gpus - 4)} {extra}".rstrip()
        )
    else:
        raise ValueError(kind)
    lines.append("echo ALL-DONE")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Presets: --preset suite|rows --tag <code tag under /data/d25/vega/src> --ckpt <ckpt> --out <dir> "
            "[--rows f ...] [--extra='<more flags>'] (note the =) build the run_suite / run_rows command instead of --script."
        ),
    )
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--node", required=True, help="kubernetes.io/hostname of the target node"
    )
    ap.add_argument("--script", help="bash script file run in the container")
    ap.add_argument("--preset", choices=("suite", "rows"))
    ap.add_argument("--tag")
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument(
        "--triton-cache", help="default /data/d25/vega/<ws>/triton-cache/<name>"
    )
    ap.add_argument(
        "--extra",
        default="",
        help="extra flags for run_suite/run_rows, e.g. '--index-only'",
    )
    ap.add_argument("--ws", default="eval")
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--memory-gi", type=int)
    ap.add_argument("--image")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument(
        "--readonly",
        action="append",
        default=[],
        help="HOST_PATH:MOUNT_PATH, repeatable",
    )
    ap.add_argument("--no-hf-token", action="store_true")
    a = ap.parse_args(argv)
    if not a.name.startswith(f"d25-vega-{a.ws}-"):
        raise SystemExit(f"Job names must start with d25-vega-{a.ws}-")
    env = dict(item.split("=", 1) for item in a.env)
    readonly = [tuple(item.split(":", 1)) for item in a.readonly]
    if a.preset:
        if not (a.tag and a.ckpt and a.out and a.gpus):
            raise SystemExit("--preset needs --tag, --ckpt, --out and --gpus")
        triton = a.triton_cache or f"/data/d25/vega/{a.ws}/triton-cache/{a.name}"
        script = preset_script(
            a.preset, a.tag, a.ckpt, a.out, a.gpus, a.rows, triton, a.extra
        )
    elif a.script:
        script = Path(a.script).read_text()
    else:
        raise SystemExit("give --script or --preset")
    manifest = job(
        a.name,
        a.node,
        script,
        ws=a.ws,
        gpus=a.gpus,
        cpu=a.cpu,
        memory_gi=a.memory_gi,
        image=a.image,
        env=env,
        readonly=readonly,
        hf_token=not a.no_hf_token,
    )
    json.dump(manifest, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
