"""Kubernetes Job generator for Vega training arms and trainer tools (ws train).

Node facts (hostname, SSH target, where the bf16 Qwen3.8-27B lives) come from a private mapping
file that is never committed (``$D25_NODES_FILE``, default ``~/.config/d25-vega/nodes.json``):

    {"05": {"hostname": "<k8s node name>", "ssh": "root@<address>",
            "base_hostpath": "<dir on the node>", "base_subdir": "<subdir with config.json or empty>"}}

Subcommands:

    stage  --node 05 [--tag T]              copy this worktree's ``d25/`` package to /data/d25/vega/src/<tag>/
    arm    --name <arm> --node 05 --tag T ...  training Job YAML (8 GPUs, auto-resume from the latest DCP)
    tool   --name <what> --node 04 --gpus 1 --tag T -- <command...>   any command in the training pod layout

Write the YAML with ``> job.yaml`` and submit with ``kubectl --context vllm-sr apply -f job.yaml``
after re-checking free GPUs and appending a ledger line (see WORKER_RULES).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile

from pathlib import Path

IMAGE = "docker.io/vllm/vllm-openai-rocm:v0.31.0"
NAMESPACE = "semantic-router"
WS = "train"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../src/training/decision2/d25
FLA_OVERLAY = "/data/d25/shared/pylib/fla052"
BASE_MOUNT = "/models/base-root"
HF_HOME = "/data/d25/shared/hf-home"
TRITON_CACHE = "/data/d25/vega/train/triton-cache"


def nodes() -> dict:
    path = Path(
        os.environ.get("D25_NODES_FILE", "~/.config/d25-vega/nodes.json")
    ).expanduser()
    if not path.exists():
        sys.exit(f"node mapping {path} not found (private; see module docstring)")
    return json.loads(path.read_text())


def node_info(node: str) -> dict:
    table = nodes()
    if node not in table:
        sys.exit(f"node {node!r} not in mapping ({sorted(table)})")
    return table[node]


def base_path(node: str) -> str:
    info = node_info(node)
    if not info.get("base_hostpath"):
        sys.exit(
            f"node {node} has no local Qwen3.8-27B; pass --init pointing at a downloaded copy"
        )
    sub = info.get("base_subdir") or ""
    return f"{BASE_MOUNT}/{sub}".rstrip("/")


# ---------------------------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------------------------


STAGED = ("__init__.py", "vega/__init__.py", "vega/common", "vega/train", "vega/k8s")
OVERLAY = ("vega/train", "vega/k8s")


def package_tar(
    full: bool = False, entries: tuple[str, ...] | None = None
) -> tuple[bytes, str]:
    """The trainer's import closure, or (``full``) the whole ``d25`` package incl. ws-measure's eval code."""
    buffer = io.BytesIO()
    files = []
    for entry in entries or (("",) if full else STAGED):
        path = PACKAGE_ROOT / entry
        files.extend([path] if path.is_file() else path.rglob("*"))
    files = sorted(
        p
        for p in files
        if p.is_file()
        and "__pycache__" not in p.parts
        and not p.name.endswith((".pyc", ".pyo"))
    )
    digest = hashlib.sha256()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in files:
            rel = Path("d25") / path.relative_to(PACKAGE_ROOT)
            data = path.read_bytes()
            digest.update(str(rel).encode() + b"\0" + data)
            info = tarfile.TarInfo(str(rel))
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue(), digest.hexdigest()


def stage_overlay(args: argparse.Namespace) -> None:
    """New tag = the node's current tag with this worktree's train/ and k8s/ on top (eval code untouched)."""
    payload, digest = package_tar(entries=OVERLAY)
    ssh = node_info(args.node)["ssh"]
    base = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ssh, "readlink /data/d25/vega/src/current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not base or "/" in base:
        sys.exit(f"unexpected current tag {base!r} on node {args.node}")
    content = f"{base}+{digest}"
    tag = f"o-{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    remote = f"/data/d25/vega/src/{tag}"
    existing = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            ssh,
            f"cat {remote}/CONTENT_SHA256 2>/dev/null || true",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if existing != content:
        tmp = Path(f"/tmp/d25-vega-{tag}.tar.gz")
        tmp.write_bytes(payload)
        local_sha = hashlib.sha256(payload).hexdigest()
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                ssh,
                f"rm -rf {remote}.incoming && mkdir -p {remote}.incoming",
            ],
            check=True,
        )
        subprocess.run(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                str(tmp),
                f"{ssh}:{remote}.incoming/overlay.tar.gz",
            ],
            check=True,
        )
        script = (
            f'set -e; cd {remote}.incoming; test "$(sha256sum overlay.tar.gz | cut -c1-64)" = {local_sha}; '
            f"cp -a /data/d25/vega/src/{base}/d25 .; find d25 -name __pycache__ -prune -exec rm -rf {{}} +; "
            f"tar -xzf overlay.tar.gz; rm overlay.tar.gz; echo {content} > CONTENT_SHA256; "
            f"rm -rf {remote}; mv {remote}.incoming {remote}; ls {remote}/d25/vega/train | wc -l"
        )
        subprocess.run(["ssh", "-o", "BatchMode=yes", ssh, script], check=True)
        tmp.unlink()
    if args.set_current:
        set_current(ssh, tag)
    print(
        json.dumps(
            {
                "node": args.node,
                "tag": tag,
                "base": base,
                "overlay_sha256": digest,
                "current": bool(args.set_current),
            }
        )
    )


def stage(args: argparse.Namespace) -> None:
    if args.overlay:
        return stage_overlay(args)
    payload, digest = package_tar(full=args.full)
    tag = args.tag or f"{'f' if args.full else 't'}-{digest[:12]}"
    ssh = node_info(args.node)["ssh"]
    remote = f"/data/d25/vega/src/{tag}"
    probe = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            ssh,
            f"cat {remote}/CONTENT_SHA256 2>/dev/null || true",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if probe == digest:
        if args.set_current:
            set_current(ssh, tag)
        print(
            json.dumps(
                {
                    "node": args.node,
                    "tag": tag,
                    "path": remote,
                    "content_sha256": digest,
                    "already": True,
                    "current": bool(args.set_current),
                }
            )
        )
        return
    if probe and not args.force:
        sys.exit(
            f"{remote} exists with different content ({probe[:12]} != {digest[:12]}); use a new tag or --force"
        )
    tmp = Path(f"/tmp/d25-vega-{tag}.tar.gz")
    tmp.write_bytes(payload)
    local_sha = hashlib.sha256(payload).hexdigest()
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ssh, f"mkdir -p {remote}.incoming"], check=True
    )
    subprocess.run(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            str(tmp),
            f"{ssh}:{remote}.incoming/code.tar.gz",
        ],
        check=True,
    )
    script = (
        f'set -e; cd {remote}.incoming; test "$(sha256sum code.tar.gz | cut -c1-64)" = {local_sha}; '
        f"tar -xzf code.tar.gz; rm code.tar.gz; echo {digest} > CONTENT_SHA256; "
        f"rm -rf {remote}; mv {remote}.incoming {remote}; ls {remote}/d25/vega"
    )
    subprocess.run(["ssh", "-o", "BatchMode=yes", ssh, script], check=True)
    tmp.unlink()
    if args.set_current:
        set_current(ssh, tag)
    print(
        json.dumps(
            {
                "node": args.node,
                "tag": tag,
                "path": remote,
                "content_sha256": digest,
                "current": bool(args.set_current),
            }
        )
    )


def set_current(ssh: str, tag: str) -> None:
    """Atomically point /data/d25/vega/src/current at a staged tag (runners re-resolve it per arm/eval)."""
    script = (
        f"set -e; cd /data/d25/vega/src; test -f {tag}/CONTENT_SHA256; "
        f"ln -sfn {tag} current.new; mv -T current.new current; readlink current"
    )
    subprocess.run(["ssh", "-o", "BatchMode=yes", ssh, script], check=True)


# ---------------------------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------------------------


def q(value: str) -> str:
    return json.dumps(value)


def resources(
    gpus: int, cpu: int | None = None, mem_gi: int | None = None
) -> tuple[int, int]:
    if cpu and mem_gi:
        return cpu, mem_gi
    if gpus >= 8:
        return 120, 1000
    if gpus == 0:
        return 16, 64
    return min(120, 14 * gpus), min(1000, 120 * gpus)


def base_dir_in_pod(node: str) -> str:
    info = node_info(node)
    if info.get("base_pod_path"):
        return info["base_pod_path"]
    if not info.get("base_hostpath"):
        return ""
    return f"{BASE_MOUNT}/{info.get('base_subdir') or ''}".rstrip("/")


def job_yaml(
    *,
    name: str,
    node: str,
    gpus: int,
    tag: str,
    command: str,
    arm: str | None = None,
    shm_gi: int = 64,
    grace: int = 900,
    extra_env: dict[str, str] | None = None,
    mount_base: bool = True,
    ttl: int = 86400,
    restart_policy: str = "Never",
    backoff: int = 0,
    extra_labels: dict[str, str] | None = None,
    cpu: int | None = None,
    mem_gi: int | None = None,
    hf_token: bool = True,
) -> str:
    info = node_info(node)
    cpu, mem = resources(gpus, cpu, mem_gi)
    src = f"/data/d25/vega/src/{tag}"
    env = {
        "PYTHONPATH": f"{FLA_OVERLAY}:{src}",
        "HF_HOME": HF_HOME,
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRITON_CACHE_DIR": TRITON_CACHE,
        "TRITON_CACHE_AUTOTUNING": "1",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_DISABLE": "1",
        # RCCL picks 4 channels on these virtualised xGMI nodes: 13 GB/s all-gather; 64 channels: ~95 GB/s.
        "NCCL_MIN_NCHANNELS": "64",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "PYTHONUNBUFFERED": "1",
        "D25_CODE_TAG": tag,
        "D25_BASE_DIR": base_dir_in_pod(node),
    }
    env.update(extra_env or {})
    labels = {
        "app.kubernetes.io/part-of": "decision-2.5",
        "d25/campaign": "vega",
        "d25/ws": WS,
    }
    if arm:
        labels["d25/arm"] = arm
    labels.update(extra_labels or {})
    label_yaml = "\n".join(f"    {k}: {q(v)}" for k, v in labels.items())
    pod_labels = "\n".join(f"        {k}: {q(v)}" for k, v in labels.items())
    env_yaml = "\n".join(
        f"        - {{name: {k}, value: {q(v)}}}" for k, v in env.items()
    )
    gpu_req = f', amd.com/gpu: "{gpus}"' if gpus else ""
    base_mount = base_volume = ""
    if mount_base and info.get("base_hostpath"):
        base_mount = (
            f"\n        - {{name: base, mountPath: {BASE_MOUNT}, readOnly: true}}"
        )
        base_volume = f"\n      - {{name: base, hostPath: {{path: {q(info['base_hostpath'])}, type: Directory}}}}"
    token_yaml = (
        "\n        - {name: HF_TOKEN, valueFrom: {secretKeyRef: {name: hf-token, key: HF_TOKEN}}}"
        if hf_token
        else ""
    )
    script = f"set -euo pipefail\ncd {src}\n{command}\n"
    script_yaml = "\n".join("          " + line for line in script.splitlines())
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {NAMESPACE}
  labels:
{label_yaml}
spec:
  backoffLimit: {backoff}
  ttlSecondsAfterFinished: {ttl}
  template:
    metadata:
      labels:
{pod_labels}
    spec:
      restartPolicy: {restart_policy}
      enableServiceLinks: false
      terminationGracePeriodSeconds: {grace}
      nodeSelector: {{kubernetes.io/hostname: {q(info['hostname'])}}}
      containers:
      - name: main
        image: {IMAGE}
        imagePullPolicy: IfNotPresent
        command: ["bash", "-c"]
        args:
        - |
{script_yaml}
        env:
{env_yaml}{token_yaml}
        resources:
          requests: {{cpu: "{cpu}", memory: {mem}Gi{gpu_req}}}
          limits: {{cpu: "{cpu}", memory: {mem}Gi{gpu_req}}}
        securityContext:
          privileged: false
          capabilities: {{add: ["SYS_PTRACE"]}}
          seccompProfile: {{type: Unconfined}}
        volumeMounts:
        - {{name: vega, mountPath: /data/d25/vega}}
        - {{name: shared, mountPath: /data/d25/shared}}
        - {{name: dshm, mountPath: /dev/shm}}{base_mount}
      volumes:
      - {{name: vega, hostPath: {{path: /data/d25/vega, type: DirectoryOrCreate}}}}
      - {{name: shared, hostPath: {{path: /data/d25/shared, type: DirectoryOrCreate}}}}
      - {{name: dshm, emptyDir: {{medium: Memory, sizeLimit: {shm_gi}Gi}}}}{base_volume}
"""


def arm(args: argparse.Namespace) -> None:
    init = args.init or base_path(args.node)
    run = args.run or args.name
    out = f"/data/d25/vega/ckpt/{run}"
    flags = [
        f"--run {shlex.quote(run)}",
        f"--train {shlex.quote(args.train)}",
        f"--output {out}",
        f"--init {shlex.quote(init)}",
        f"--init-kind {args.init_kind}",
        f"--attention-mode {args.attention_mode}",
        f"--lr {args.lr}",
        f"--readout-lr {args.readout_lr if args.readout_lr is not None else args.lr}",
        f"--warmup-ratio {args.warmup_ratio}",
        f"--schedule {args.schedule}",
        f"--min-lr-ratio {args.min_lr_ratio}",
        f"--epochs {args.epochs}",
        f"--seed {args.seed}",
        f"--rows-per-update {args.rows_per_update}",
        f"--max-length {args.max_length}",
        f"--token-budget {args.token_budget}",
        f"--ac {args.ac}",
        f"--save-every {args.save_every}",
        f"--keep-dcp {args.keep_dcp}",
        f"--dev-every {args.dev_every}",
    ]
    if args.tokenizer:
        flags.append(f"--tokenizer {shlex.quote(args.tokenizer)}")
    if args.dev:
        flags.append(f"--dev {shlex.quote(args.dev)}")
    if args.export_steps:
        flags.append(f"--export-steps {args.export_steps}")
    if args.export_every:
        flags.append(f"--export-every {args.export_every}")
    if args.teacher:
        flags.append(
            f"--teacher {shlex.quote(args.teacher)} --teacher-weight {args.teacher_weight}"
        )
    if args.brier_weight:
        flags.append(f"--brier-weight {args.brier_weight}")
    if args.max_steps:
        flags.append(f"--max-steps {args.max_steps}")
    flags.extend(args.extra)
    joined = " \\\n  ".join(flags)
    command = (
        f"mkdir -p {out}/logs {TRITON_CACHE}\n"
        f'echo "job $(date -u +%FT%TZ) tag {args.tag} host $(hostname)" >> {out}/logs/jobs.txt\n'
        f"exec python -m d25.vega.train.launch --nproc {args.gpus} --log-dir {out}/logs d25.vega.train.train \\\n"
        f"  {joined}"
    )
    name = f"d25-vega-train-{args.name}"
    print(
        job_yaml(
            name=name,
            node=args.node,
            gpus=args.gpus,
            tag=args.tag,
            command=command,
            arm=args.name,
            shm_gi=args.shm_gi,
            grace=args.grace,
        )
    )


def runner(args: argparse.Namespace) -> None:
    """Long-running per-node queue runner (pod restarts resume the running arm)."""
    command = (
        f"exec python -m d25.vega.train.runner --node {args.node} --gpus {args.gpus}"
    )
    print(
        job_yaml(
            name=f"d25-vega-runner-{args.node}",
            node=args.node,
            gpus=args.gpus,
            tag="current",
            command=command,
            shm_gi=args.shm_gi,
            grace=args.grace,
            restart_policy="OnFailure",
            backoff=6,
            ttl=7 * 86400,
            extra_labels={"d25/runner": args.node},
        )
    )


def xfer(args: argparse.Namespace) -> None:
    """Per-node export server + background puller (CPU only) and its ClusterIP Service."""
    name = f"d25-vega-train-xfer-{args.node}"
    job = job_yaml(
        name=name,
        node=args.node,
        gpus=0,
        tag="current",
        command="exec python -m d25.vega.train.xfer serve --port 8080",
        shm_gi=1,
        grace=30,
        mount_base=False,
        restart_policy="OnFailure",
        backoff=20,
        ttl=7 * 86400,
        extra_labels={"d25/xfer": args.node},
        cpu=args.cpu,
        mem_gi=args.mem_gi,
        hf_token=False,
    )
    service = f"""---
apiVersion: v1
kind: Service
metadata:
  name: {name}
  namespace: {NAMESPACE}
  labels:
    app.kubernetes.io/part-of: "decision-2.5"
    d25/campaign: "vega"
    d25/ws: "{WS}"
    d25/xfer: "{args.node}"
spec:
  type: ClusterIP
  selector:
    d25/ws: "{WS}"
    d25/xfer: "{args.node}"
  ports:
  - {{name: http, port: 8080, targetPort: 8080, protocol: TCP}}
"""
    print(job + service)


def tool(args: argparse.Namespace) -> None:
    command = (
        " ".join(shlex.quote(part) for part in args.command)
        if not args.shell
        else " ".join(args.command)
    )
    print(
        job_yaml(
            name=f"d25-vega-train-{args.name}",
            node=args.node,
            gpus=args.gpus,
            tag=args.tag,
            command=command,
            shm_gi=args.shm_gi,
            grace=args.grace,
            mount_base=not args.no_base,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--node", required=True)
    s.add_argument("--tag", help="Default: t-<content sha256 prefix>")
    s.add_argument("--force", action="store_true")
    s.add_argument(
        "--full",
        action="store_true",
        help="Stage the whole d25 package (trainer + eval + data)",
    )
    s.add_argument(
        "--set-current",
        action="store_true",
        help="Point /data/d25/vega/src/current at the tag",
    )
    s.add_argument(
        "--overlay",
        action="store_true",
        help="Copy the node's current tag and replace only train/ and k8s/ (keeps the deployed eval code)",
    )
    x = sub.add_parser("xfer")
    x.add_argument("--node", required=True)
    x.add_argument("--cpu", type=int, default=4)
    x.add_argument("--mem-gi", type=int, default=16)
    r = sub.add_parser("runner")
    r.add_argument("--node", required=True)
    r.add_argument("--gpus", type=int, default=8)
    r.add_argument("--shm-gi", type=int, default=128)
    r.add_argument("--grace", type=int, default=1800)
    a = sub.add_parser("arm")
    a.add_argument(
        "--name",
        required=True,
        help="Arm name (Job d25-vega-train-<name>, label d25/arm)",
    )
    a.add_argument("--run", help="Checkpoint dir name (default: --name)")
    a.add_argument("--node", required=True)
    a.add_argument("--tag", required=True, help="Staged code tag on that node")
    a.add_argument("--gpus", type=int, default=8)
    a.add_argument(
        "--train",
        required=True,
        help="Mixture file/dir on the node, e.g. /data/d25/shared/data/v1/<mix>",
    )
    a.add_argument("--dev")
    a.add_argument(
        "--init",
        help="Init checkpoint dir in the pod (default: the node's base Qwen3.8-27B)",
    )
    a.add_argument("--init-kind", choices=("base", "warm", "export"), default="base")
    a.add_argument("--tokenizer")
    a.add_argument(
        "--attention-mode",
        choices=("causal", "noncausal_full_attention"),
        default="noncausal_full_attention",
    )
    a.add_argument("--lr", type=float, default=2e-6)
    a.add_argument("--readout-lr", type=float)
    a.add_argument("--warmup-ratio", type=float, default=0.15)
    a.add_argument(
        "--schedule", choices=("cosine", "linear", "constant"), default="cosine"
    )
    a.add_argument("--min-lr-ratio", type=float, default=0.1)
    a.add_argument("--epochs", type=int, default=1)
    a.add_argument("--seed", type=int, default=20260920)
    a.add_argument("--rows-per-update", type=int, default=256)
    a.add_argument("--max-length", type=int, default=8192)
    a.add_argument("--token-budget", type=int, default=16384)
    a.add_argument("--ac", default="full")
    a.add_argument("--save-every", type=int, default=200)
    a.add_argument("--keep-dcp", type=int, default=2)
    a.add_argument("--dev-every", type=int, default=100)
    a.add_argument("--export-steps", default="")
    a.add_argument("--export-every", type=int, default=0)
    a.add_argument("--teacher")
    a.add_argument("--teacher-weight", type=float, default=0.0)
    a.add_argument("--brier-weight", type=float, default=0.0)
    a.add_argument("--max-steps", type=int)
    a.add_argument("--shm-gi", type=int, default=64)
    a.add_argument("--grace", type=int, default=1200)
    a.add_argument("extra", nargs="*", help="Extra trainer flags after --")
    t = sub.add_parser("tool")
    t.add_argument("--name", required=True)
    t.add_argument("--node", required=True)
    t.add_argument("--tag", required=True)
    t.add_argument("--gpus", type=int, default=0)
    t.add_argument("--shm-gi", type=int, default=32)
    t.add_argument("--grace", type=int, default=300)
    t.add_argument(
        "--shell", action="store_true", help="Pass the command to bash unquoted"
    )
    t.add_argument("--no-base", action="store_true")
    t.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cmd == "tool" and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    {"stage": stage, "arm": arm, "tool": tool, "runner": runner, "xfer": xfer}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
