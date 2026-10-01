"""Single-GPU container launcher for the ~27B track (host side, stdlib only).

Only the track's GPUs are accepted: node B GPU5-7 and, with ``DEV2_NODE=a``,
node A GPU2-4. Both nodes share one PCI layout, so the node comes from
``DEV2_NODE`` (default ``b``) and the lease file, which must belong to the
track, guards against the wrong node. The launcher checks the render node's PCI
address, updates the track's lease file, runs one network-less container with
exactly that device, enforces a wall-clock cap and writes a GPU-hour receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

TRACK = "27b"
IMAGE_ID = "sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1"
NODE_GPUS = {
    "b": {
        5: ("0000:ab:00.0", "renderD169"),
        6: ("0000:b3:00.0", "renderD177"),
        7: ("0000:bb:00.0", "renderD185"),
    },
    "a": {
        2: ("0000:93:00.0", "renderD145"),
        3: ("0000:9b:00.0", "renderD153"),
        4: ("0000:a3:00.0", "renderD161"),
    },
}
ALLOWED_GPUS = NODE_GPUS["b"]
MAX_CAP_HOURS = 13.0
# Milestone 6 (opt-in with DEV2_27B_ALLOC=m6): the 27B leases node B GPU0 / GPU1 / GPU5 and node A GPU2, and a 20 GPU-hour
# ceiling per launch (rank-128 arm-seeds on A20 + IB1 / + IB2 project to about 12.4 / 14.8 GPU-h).
M6_NODE_GPUS = {
    "b": {
        0: ("0000:83:00.0", "renderD129"),
        1: ("0000:8b:00.0", "renderD137"),
        5: NODE_GPUS["b"][5],
    },
    "a": {2: NODE_GPUS["a"][2]},
}
M6_MAX_CAP_HOURS = 20.0
LEASE_ROOT = Path("/data/dev2/leases")


def m6_allocation() -> bool:
    value = os.environ.get("DEV2_27B_ALLOC", "")
    if value not in ("", "m6"):
        raise ValueError(f"DEV2_27B_ALLOC must be unset or m6, not {value!r}")
    return value == "m6"


def max_cap_hours() -> float:
    return M6_MAX_CAP_HOURS if m6_allocation() else MAX_CAP_HOURS


def node_name() -> str:
    node = os.environ.get("DEV2_NODE", "b")
    if node not in NODE_GPUS:
        raise ValueError(f"DEV2_NODE must be one of {sorted(NODE_GPUS)}, not {node!r}")
    return node


def allowed_gpus(node: str | None = None) -> dict[int, tuple[str, str]]:
    return (M6_NODE_GPUS if m6_allocation() else NODE_GPUS)[node or node_name()]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render_node(
    gpu: int, sysfs: Path = Path("/sys/class/drm"), node_id: str | None = None
) -> Path:
    gpus = allowed_gpus(node_id)
    if gpu not in gpus:
        where = (
            "node B GPU0/1/5, node A GPU2"
            if m6_allocation()
            else "node B GPU5-7, node A GPU2-4"
        )
        raise ValueError(f"GPU{gpu} is outside the ~27B allocation ({where})")
    pci, node = gpus[gpu]
    actual = (sysfs / node / "device").resolve().name.lower()
    if actual != pci:
        raise ValueError(f"{node} maps to {actual}, expected {pci}")
    return Path("/dev/dri") / node


def read_lease(owner: Path) -> dict:
    """Parse this launcher's JSON owner file or the eval runner's KEY=VALUE lines."""
    text = owner.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = [line.split("=", 1) for line in text.splitlines() if "=" in line]
        return {key.strip(): value.strip() for key, value in lines}


def write_lease(
    gpu: int, fields: dict, root: Path = LEASE_ROOT, shared: str | None = None
) -> None:
    """Update the track's lease; ``shared`` adds/removes a co-located job."""
    lease = root / f"gpu{gpu}.lock"
    owner = lease / "owner"
    if not owner.is_file():
        raise ValueError(f"GPU{gpu} lease is missing; create it before launching")
    current = read_lease(owner)
    if current.get("track") != TRACK:
        raise ValueError(f"GPU{gpu} lease belongs to {current.get('track')}")
    if shared is not None:
        jobs = set(current.get("shared_containers") or [])
        jobs.add(shared) if fields.get("status") == "running" else jobs.discard(shared)
        fields = {"shared_containers": sorted(jobs)}
    elif fields.get("status") == "running" and current.get("status") == "running":
        raise ValueError(f"GPU{gpu} already runs {current.get('container')}")
    current.update(fields)
    pending = owner.with_name("owner.pending")
    pending.write_text(json.dumps(current, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(pending, owner)


def vram_percent(gpu: int) -> str:
    output = subprocess.run(
        ["rocm-smi", "-d", str(gpu), "--showmemuse"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    values = [
        line.rsplit(":", 1)[-1].strip()
        for line in output.splitlines()
        if "VRAM%" in line
    ]
    return values[-1] if values else "unknown"


def wait_until_idle(gpu: int, timeout: float = 300, poll: float = 10) -> None:
    """VRAM stays allocated for a short while after a container exits."""
    deadline = time.monotonic() + timeout
    while (usage := vram_percent(gpu)) != "0":
        if time.monotonic() > deadline:
            raise ValueError(f"GPU{gpu} is not idle (VRAM% {usage})")
        time.sleep(poll)


def docker(*args: str, timeout: int = 300) -> str:
    return subprocess.check_output(
        ["docker", *args], text=True, timeout=timeout, stderr=subprocess.STDOUT
    ).strip()


def create_argv(args: argparse.Namespace, device: Path) -> list[str]:
    argv = [
        "create",
        "--name",
        args.name,
        "--network",
        "none",
        "--shm-size",
        "16g",
        "--device",
        "/dev/kfd",
        "--device",
        str(device),
        "-e",
        "ROCR_VISIBLE_DEVICES=0",
        "-e",
        "HIP_VISIBLE_DEVICES=0",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "TOKENIZERS_PARALLELISM=false",
    ]
    for item in args.env:
        if "=" not in item or item.split("=", 1)[0].upper().endswith(
            ("TOKEN", "KEY", "SECRET")
        ):
            raise ValueError(
                "Only non-secret KEY=VALUE environment entries are allowed"
            )
        argv.extend(["-e", item])
    for spec in args.mount:
        parts = spec.split(":")
        if len(parts) not in (2, 3) or (
            len(parts) == 3 and parts[2] not in ("ro", "rw")
        ):
            raise ValueError(f"Invalid mount {spec}; use HOST:DEST[:ro|rw]")
        host = Path(parts[0]).resolve(strict=True)
        readonly = len(parts) == 2 or parts[2] == "ro"
        argv.extend(
            [
                "--mount",
                f"type=bind,src={host},dst={parts[1]}"
                + (",readonly" if readonly else ""),
            ]
        )
    argv.extend(["-w", args.workdir, IMAGE_ID, *args.command])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cap-hours", type=float, required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mount", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--workdir", default="/code")
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Co-locate a small job on a GPU already running this track's job",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        raise ValueError("Missing container command after --")
    if not 0 < args.cap_hours <= max_cap_hours():
        raise ValueError(f"cap-hours must be in (0, {max_cap_hours():g}]")
    if args.receipt.exists():
        raise FileExistsError("Receipt already exists; use a fresh run name")
    device = render_node(args.gpu)
    if docker("image", "inspect", IMAGE_ID, "--format", "{{.Id}}") != IMAGE_ID:
        raise ValueError("Pinned runtime image is absent or changed")
    if docker("ps", "-a", "--filter", f"name=^{args.name}$", "--format", "{{.Names}}"):
        raise FileExistsError("Container name already exists")
    if not args.shared:
        wait_until_idle(args.gpu)
    argv = create_argv(args, device)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    cap = int(args.cap_hours * 3600)
    started_utc = utc()
    write_lease(
        args.gpu,
        {
            "status": "running",
            "container": args.name,
            "purpose": args.purpose,
            "start_utc": started_utc,
            "expected_end_utc": datetime.fromtimestamp(
                time.time() + cap, timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        shared=args.name if args.shared else None,
    )
    cid = ""
    exit_code = None
    timed_out = threading.Event()
    done = threading.Event()
    start = time.monotonic()
    try:
        cid = docker(*argv)
        docker("start", cid)

        def watchdog() -> None:
            if not done.wait(cap):
                timed_out.set()
                subprocess.run(
                    ["docker", "stop", "-t", "60", cid], timeout=120, check=False
                )

        thread = threading.Thread(target=watchdog, daemon=True)
        thread.start()
        exit_code = int(docker("wait", cid, timeout=cap + 600).splitlines()[-1])
    finally:
        done.set()
        elapsed = time.monotonic() - start
        log_path = args.receipt.with_name(args.receipt.stem + ".container.log")
        if cid:
            logs = subprocess.run(["docker", "logs", cid], capture_output=True)
            log_path.write_bytes(logs.stdout + logs.stderr)
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True,
                timeout=120,
                check=False,
            )
        receipt = {
            "schema_version": "decision2-27b-launch-receipt/1",
            "track": TRACK,
            "name": args.name,
            "purpose": args.purpose,
            "node": node_name(),
            "gpu_index": args.gpu,
            "pci_bus": allowed_gpus()[args.gpu][0],
            "render_node": device.name,
            "image_id": IMAGE_ID,
            "container_id": cid,
            "docker_create_argv": argv,
            "started_utc": started_utc,
            "finished_utc": utc(),
            "elapsed_seconds": round(elapsed, 3),
            "gpu_hours": round(elapsed / 3600, 5),
            "cap_hours": args.cap_hours,
            "shared_gpu": args.shared,
            "watchdog_fired": timed_out.is_set(),
            "exit_code": exit_code,
            "container_log_sha256": sha_file(log_path) if log_path.exists() else None,
        }
        fd = os.open(args.receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(receipt, out, indent=1, sort_keys=True)
            out.write("\n")
        write_lease(
            args.gpu,
            {
                "status": "reserved-idle",
                "container": None,
                "last_receipt": str(args.receipt),
            },
            shared=args.name if args.shared else None,
        )
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("name", "exit_code", "gpu_hours", "watchdog_fired")
            }
        )
    )
    raise SystemExit(0 if exit_code == 0 and not timed_out.is_set() else 1)


if __name__ == "__main__":
    main()
