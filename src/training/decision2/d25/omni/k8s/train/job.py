"""Job manifests for Omni training (render only; the lead submits).

    python -m d25.omni.k8s.train.job train --run graft-frozen-r50 --node <hostname> --tag <src tag> \\
        --init /data/d25/omni/ckpt/init-graft --arm O-graft-frozen \\
        --rows '/data/d25/omni/data/<corpus>/<version>/rows-*.jsonl.gz' \\
        --replay-rows '/data/d25/shared/data/<corpus>/<mix>/rows-*.jsonl.gz' --replay-ratio 0.5 > job.json
    python -m d25.omni.k8s.train.job assemble --run init-graft --node <hostname> --tag <src tag> \\
        --vega /data/d25/vega/ckpt/<run>/<step> > job.json
    python -m d25.omni.k8s.train.job eval --run graft-frozen-r50 --node <hostname> --tag <src tag> \\
        --ckpt /data/d25/omni/ckpt/graft-frozen-r50/checkpoints/step-02500 \\
        --suite /data/d25/omni/suite/vision-0.3.1 > job.json

``train`` is one single-node pod with 8 GPUs running ``torchrun`` over ``d25.omni.train.train``
(rendezvous on loopback; RDMA is down) after a preflight of the GPU count, Transformers 5.17 and the
fla overlay. ``assemble`` is a CPU Job (8 CPU, 32Gi) that builds an init checkpoint from the shared
Hugging Face cache. ``eval`` runs one suite shard per GPU, then merges. Node names, the source tag
and paths are arguments, so no fleet detail lives in git; nodes 02, 07 and 08 are refused. Writes go
only to ``/data/d25/omni``; shared and Vega directories are mounted read-only.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys

from d25.omni.k8s.train import memory_plan
from d25.omni.model.checkpoint import BASE_REVISION

IMAGE = "docker.io/vllm/vllm-openai-rocm:v0.31.0"
NAMESPACE = "semantic-router"
OMNI = "/data/d25/omni"
SHARED = "/data/d25/shared"
VEGA_CKPT = "/data/d25/vega/ckpt"
FLA_OVERLAY = f"{SHARED}/pylib/fla052"
STOCK_SNAPSHOT = (
    f"{SHARED}/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots/{BASE_REVISION}"
)
FORBIDDEN_NODE_SUFFIXES = ("-02", "-07", "-08")
TRAIN_RESOURCES = {"cpu": "96", "memory": "640Gi", "amd.com/gpu": "8"}
CPU_RESOURCES = {"cpu": "8", "memory": "32Gi"}
NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
PATTERN = re.compile(r"^[A-Za-z0-9_./*?\[\]=-]+$")
ARMS = ("O-graft-frozen", "O-graft-lowlr", "O-fresh")


def labels(run: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/part-of": "decision-2.5",
        "d25/campaign": "omni",
        "d25/ws": "train",
        "d25/run": run,
    }


def check(args: argparse.Namespace) -> None:
    if not NAME.match(args.run) or len(f"d25-omni-train-{args.kind}-{args.run}") > 63:
        raise SystemExit(
            "--run must be a short DNS label (lowercase letters, digits, dashes)"
        )
    if args.node.endswith(FORBIDDEN_NODE_SUFFIXES):
        raise SystemExit("nodes 02, 07 and 08 are not allowed for Omni")
    for name in ("init", "out", "ckpt"):
        value = getattr(args, name, None)
        if value and not value.startswith(f"{OMNI}/ckpt/"):
            raise SystemExit(f"--{name} must live under {OMNI}/ckpt/")
    for name in ("rows", "replay_rows", "dev_rows"):
        for value in getattr(args, name, None) or []:
            if not value.startswith(
                (f"{OMNI}/data/", f"{SHARED}/data/")
            ) or not PATTERN.match(value):
                raise SystemExit(
                    f"--{name.replace('_', '-')} must be a plain path or glob under {OMNI}/data "
                    f"or {SHARED}/data"
                )
    vega = getattr(args, "vega", None)
    if vega and not vega.startswith(f"{VEGA_CKPT}/"):
        raise SystemExit(f"--vega must be a Vega checkpoint under {VEGA_CKPT}/")
    suite = getattr(args, "suite", None)
    if suite and not suite.startswith((f"{OMNI}/suite/", f"{OMNI}/proxy/")):
        raise SystemExit(f"--suite must live under {OMNI}/suite or {OMNI}/proxy")


def volumes(writable: set[str], vega: bool) -> tuple[list[dict], list[dict]]:
    """Omni dirs (DirectoryOrCreate; ``writable`` ones read-write), shared and Vega dirs read-only."""
    entries = [
        (name, f"{OMNI}/{name}", name not in writable, "DirectoryOrCreate")
        for name in ("ckpt", "evals", "data", "suite", "proxy", "src")
    ]
    entries += [
        (f"shared-{name}", f"{SHARED}/{name}", True, "Directory")
        for name in ("pylib", "hf-home", "data")
    ]
    if vega:
        entries.append(("vega-ckpt", VEGA_CKPT, True, "Directory"))
    mounts = [
        {"name": n, "mountPath": p, **({"readOnly": True} if ro else {})}
        for n, p, ro, _ in entries
    ]
    specs = [
        {"name": n, "hostPath": {"path": p, "type": kind}} for n, p, _, kind in entries
    ]
    return mounts, specs


def job(
    kind: str,
    args: argparse.Namespace,
    script: str,
    resources: dict[str, str],
    writable: set[str],
    shm_gi: int,
    vega: bool = False,
) -> dict:
    workdir = f"{OMNI}/src/{args.tag}/src/training/decision2"
    mounts, specs = volumes(writable, vega)
    gpu = "amd.com/gpu" in resources
    security = {"privileged": False, "allowPrivilegeEscalation": False}
    if gpu:
        security.update(
            capabilities={"add": ["SYS_PTRACE"]}, seccompProfile={"type": "Unconfined"}
        )
    env = [
        {"name": "PYTHONPATH", "value": f"{workdir}:{FLA_OVERLAY}"},
        {"name": "HF_HOME", "value": f"{SHARED}/hf-home"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "HF_HUB_DISABLE_XET", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "OMP_NUM_THREADS", "value": "8"},
    ]
    if gpu:
        env += [
            {"name": "PYTORCH_HIP_ALLOC_CONF", "value": "expandable_segments:True"},
            {"name": "NCCL_SOCKET_IFNAME", "value": "lo"},
            {"name": "GLOO_SOCKET_IFNAME", "value": "lo"},
            {"name": "TRITON_CACHE_AUTOTUNING", "value": "1"},
        ]
    container = {
        "name": kind,
        "image": IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "workingDir": workdir,
        "command": ["bash", "-c", script],
        "env": env,
        "resources": {"requests": dict(resources), "limits": dict(resources)},
        "securityContext": security,
        "volumeMounts": mounts + [{"name": "dshm", "mountPath": "/dev/shm"}],
    }
    shm = {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": f"{shm_gi}Gi"}}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"d25-omni-train-{kind}-{args.run}",
            "namespace": NAMESPACE,
            "labels": labels(args.run),
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {"labels": labels(args.run)},
                "spec": {
                    "restartPolicy": "Never",
                    "enableServiceLinks": False,
                    "nodeSelector": {"kubernetes.io/hostname": args.node},
                    "containers": [container],
                    "volumes": specs + [shm],
                },
            },
        },
    }


PREFLIGHT = """python - <<'PY'
import importlib.util, json, torch, transformers
fla = importlib.util.find_spec("fla") is not None
print(json.dumps({{"torch": torch.__version__, "hip": torch.version.hip, "transformers": transformers.__version__,
                  "gpus": torch.cuda.device_count(), "fla": fla}}), flush=True)
assert torch.cuda.device_count() == {gpus} and transformers.__version__ == "5.17.0" and fla
PY"""


def train_script(args: argparse.Namespace) -> str:
    out = args.out or f"{OMNI}/ckpt/{args.run}"
    flags = [
        "--init",
        args.init,
        "--arm",
        args.arm,
        "--out",
        out,
        "--code-tag",
        args.tag,
        "--replay-ratio",
        str(args.replay_ratio),
        "--token-budget",
        str(args.token_budget),
        "--effective-batch-size",
        str(args.effective_batch_size),
    ]
    command = " ".join(shlex.quote(part) for part in flags)
    for name, values in (
        ("--rows", args.rows),
        ("--replay-rows", args.replay_rows),
        ("--dev-rows", args.dev_rows),
    ):
        if values:
            command += f" {name} " + " ".join(values)
    if args.extra:
        command += " " + args.extra
    return "\n".join(
        [
            "set -euo pipefail",
            PREFLIGHT.format(gpus=8),
            f"mkdir -p {shlex.quote(out)}",
            f"export TRITON_CACHE_DIR={shlex.quote(out)}/triton-cache",
            f"torchrun --standalone --nproc_per_node 8 -m d25.omni.train.train {command} "
            f"2>&1 | tee -a {shlex.quote(out)}/train.log",
        ]
    )


def assemble_script(args: argparse.Namespace) -> str:
    out = args.out or f"{OMNI}/ckpt/{args.run}"
    flags = ["--stock", args.stock, "--out", out, "--max-length", str(args.max_length)]
    if args.vega:
        flags += ["--vega", args.vega]
    if args.attention_mode:
        flags += ["--attention-mode", args.attention_mode]
    return "set -euo pipefail\npython -m d25.omni.model.assemble " + " ".join(
        shlex.quote(p) for p in flags
    )


def eval_script(args: argparse.Namespace) -> str:
    step = args.ckpt.rstrip("/").rsplit("/", 1)[-1]
    out = args.out_eval or f"{OMNI}/evals/{args.run}/{step}/public"
    common = " ".join(
        shlex.quote(p)
        for p in [
            "--ckpt",
            args.ckpt,
            "--suite",
            args.suite,
            "--out",
            out,
            "--num-shards",
            str(args.gpus),
        ]
    )
    lines = [
        "set -euo pipefail",
        PREFLIGHT.format(gpus=args.gpus),
        f"mkdir -p {shlex.quote(out)}",
        "pids=()",
    ]
    for shard in range(args.gpus):
        lines.append(
            f"python -m d25.omni.eval.run_suite run {common} --shard {shard} --device cuda:{shard} "
            f"> {shlex.quote(out)}/shard-{shard}.log 2>&1 & pids+=($!)"
        )
    lines += [
        'for pid in "${pids[@]}"; do wait "$pid"; done',
        f"python -m d25.omni.eval.run_suite merge {common}",
    ]
    return "\n".join(lines)


def build(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    commands = {name: sub.add_parser(name) for name in ("train", "assemble", "eval")}
    for command in commands.values():
        command.add_argument("--run", required=True)
        command.add_argument(
            "--node", required=True, help="kubernetes.io/hostname of an allowed node"
        )
        command.add_argument(
            "--tag", required=True, help="code tag under /data/d25/omni/src"
        )
    train = commands["train"]
    train.add_argument("--init", required=True)
    train.add_argument("--arm", required=True, choices=ARMS)
    train.add_argument("--rows", nargs="*", default=[])
    train.add_argument("--replay-rows", nargs="*", default=[])
    train.add_argument("--dev-rows", nargs="*", default=[])
    train.add_argument("--replay-ratio", type=float, default=0.5)
    train.add_argument("--token-budget", type=int, default=32_768)
    train.add_argument("--effective-batch-size", type=int, default=256)
    train.add_argument("--out")
    train.add_argument("--extra", default="", help="further d25.omni.train.train flags")
    assemble = commands["assemble"]
    assemble.add_argument("--vega")
    assemble.add_argument("--stock", default=STOCK_SNAPSHOT)
    assemble.add_argument(
        "--attention-mode", choices=("causal", "noncausal_full_attention")
    )
    assemble.add_argument("--max-length", type=int, default=16_384)
    assemble.add_argument("--out")
    evaluate = commands["eval"]
    evaluate.add_argument("--ckpt", required=True)
    evaluate.add_argument("--suite", required=True)
    evaluate.add_argument("--gpus", type=int, default=8)
    evaluate.add_argument("--out-eval")
    args = parser.parse_args(argv)
    check(args)
    if args.kind == "train":
        if not args.rows and not args.replay_rows:
            raise SystemExit("--rows or --replay-rows is required")
        peak = memory_plan.plan(
            args.token_budget, encoder_trainable=args.arm == "O-graft-lowlr"
        ).peak
        if peak > 0.8 * memory_plan.HBM_GB:
            raise SystemExit(
                f"--token-budget {args.token_budget} plans {peak:.0f} GB per GPU (> 80% of HBM)"
            )
        return job("train", args, train_script(args), TRAIN_RESOURCES, {"ckpt"}, 64)
    if args.kind == "assemble":
        if not args.stock.startswith(f"{SHARED}/"):
            raise SystemExit(
                f"--stock must be a snapshot under {SHARED}/ (read-only reuse)"
            )
        return job(
            "assemble",
            args,
            assemble_script(args),
            CPU_RESOURCES,
            {"ckpt"},
            4,
            vega=bool(args.vega),
        )
    if not 1 <= args.gpus <= 8:
        raise SystemExit("--gpus must be 1 to 8")
    resources = {
        "cpu": str(12 * args.gpus),
        "memory": f"{64 * args.gpus}Gi",
        "amd.com/gpu": str(args.gpus),
    }
    return job("eval", args, eval_script(args), resources, {"evals"}, 16)


def main(argv: list[str] | None = None) -> None:
    json.dump(build(argv), sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
