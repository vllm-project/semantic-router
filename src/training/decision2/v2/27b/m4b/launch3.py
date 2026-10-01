"""Multi-GPU container launcher for Milestone 4b (host side, stdlib only).

Only the allocation's GPUs are accepted: by default node B GPU0-2 (lent to track
``27b-m4b``); ``DEV2_27B_LAUNCH_ALLOC=m5-b`` / ``m5-a`` selects Milestone 5's
(track ``27b``: node B GPU0-2 and GPU5-7, node A GPU2-4) and ``m6-b`` / ``m6-a``
Milestone 6's (track ``27b``: node B GPU0, GPU1, GPU5, node A GPU2). The CLI and the
receipt follow ``v2/27b/launch.py`` except that ``--gpus`` (a comma list)
replaces ``--gpu``: every render node's PCI address is checked, each GPU's lease
is updated, one network-less container gets exactly those devices (visible
inside as 0..n-1), a wall-clock cap is enforced and the receipt charges
elapsed hours x GPUs.

``lease`` takes over or updates the main owner file of each GPU: a foreign idle
owner moves to ``owner.prev-<UTC>`` and the file is rewritten as ``KEY=VALUE``
lines with ``track=27b-m4b`` (the eval runner reads that format).

Run from ``src/training/decision2`` as ``python3 -m v2.27b.m4b.launch3``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

base = importlib.import_module("v2.27b.launch")

IMAGE_ID = base.IMAGE_ID
NODE_B_GPU0_2 = {
    0: ("0000:83:00.0", "renderD129"),
    1: ("0000:8b:00.0", "renderD137"),
    2: ("0000:93:00.0", "renderD145"),
}
# DEV2_27B_LAUNCH_ALLOC picks the allocation at import (default: M4b's). Both nodes share one PCI layout;
# the lease file, which must belong to the allocation's track, guards against the wrong node.
ALLOCATIONS = {
    "m4b": ("27b-m4b", NODE_B_GPU0_2, "the M4b allocation (node B GPU0-2)"),
    "m5-b": (
        "27b",
        {**NODE_B_GPU0_2, **base.NODE_GPUS["b"]},
        "the M5 allocation on node B (GPU0-2, GPU5-7)",
    ),
    "m5-a": ("27b", dict(base.NODE_GPUS["a"]), "the M5 allocation on node A (GPU2-4)"),
    "m6-b": (
        "27b",
        {gpu: base.M6_NODE_GPUS["b"][gpu] for gpu in (0, 1, 5)},
        "the M6 allocation on node B (GPU0, GPU1, GPU5)",
    ),
    "m6-a": ("27b", dict(base.M6_NODE_GPUS["a"]), "the M6 allocation on node A (GPU2)"),
    "m6-d": (
        "27b",
        dict(base.M6_NODE_GPUS["d"]),
        "the M6 allocation on node D (GPU0-7)",
    ),
}
ALLOCATION = os.environ.get("DEV2_27B_LAUNCH_ALLOC", "m4b")
if ALLOCATION not in ALLOCATIONS:
    raise ValueError(
        f"DEV2_27B_LAUNCH_ALLOC must be one of {sorted(ALLOCATIONS)}, not {ALLOCATION!r}"
    )
TRACK, ALLOWED_GPUS, ALLOCATION_TEXT = ALLOCATIONS[ALLOCATION]
LEASE_ROOT = base.LEASE_ROOT
IDLE_STATUSES = {"idle", "released", "reserved-idle"}


def idle_status(status) -> bool:
    """A foreign owner may be moved aside when idle; other tracks also write free text ("released (IX1 complete)")."""
    text = str(status or "")
    return text in IDLE_STATUSES or text.startswith(("released", "idle"))


COTENANT_OK = {"released", "idle", "ended", "done"}
COTENANT_RELEASED_PREFIXES = ("idle-released", "released")
COTENANT_END_KEYS = ("released_utc", "last_job_end_utc", "end_utc")
LEASE_STATUSES = ("running", "reserved-idle", "idle")
MULTI_GPU_SHM = "64g"


def parse_gpus(text: str) -> list[int]:
    try:
        gpus = [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"--gpus takes a comma list of integers, got {text!r}"
        ) from exc
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus needs distinct GPU indices")
    outside = [gpu for gpu in gpus if gpu not in ALLOWED_GPUS]
    if outside:
        raise ValueError(f"GPU{outside[0]} is outside {ALLOCATION_TEXT}")
    return sorted(gpus)


def render_node(gpu: int, sysfs: Path = Path("/sys/class/drm")) -> Path:
    if gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU{gpu} is outside {ALLOCATION_TEXT}")
    pci, node = ALLOWED_GPUS[gpu]
    actual = (sysfs / node / "device").resolve().name.lower()
    if actual != pci:
        raise ValueError(f"{node} maps to {actual}, expected {pci}")
    return Path("/dev/dri") / node


def render_lease(fields: dict) -> str:
    lines = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        text = str(value)
        if "\n" in text or "=" in key:
            raise ValueError(
                f"Lease field {key} cannot be written as one KEY=VALUE line"
            )
        lines.append(f"{key}={text}")
    return "\n".join(lines) + "\n"


def write_owner(owner: Path, fields: dict) -> None:
    pending = owner.with_name("owner.pending")
    pending.write_text(render_lease(fields), encoding="utf-8")
    os.replace(pending, owner)


def shared_jobs(value) -> set[str]:
    if isinstance(value, list):
        return set(value)
    return {item for item in str(value or "").split(",") if item}


def cotenant_released(entry: dict) -> bool:
    """Other tracks write free-text statuses ("idle-released (...)", "ended", "done"); an entry
    without a status counts as released only if it records an end time."""
    status = entry.get("status")
    if status is None:
        return any(entry.get(key) for key in COTENANT_END_KEYS)
    word = str(status).strip().lower().split(" ", 1)[0].split("(", 1)[0]
    return word in COTENANT_OK or word.startswith(COTENANT_RELEASED_PREFIXES)


def check_cotenants(gpu: int, root: Path = LEASE_ROOT) -> None:
    """Every co-tenant entry (``owner.<name>``) must be released or idle."""
    lease = root / f"gpu{gpu}.lock"
    for path in sorted(lease.glob("owner.*")):
        name = path.name[len("owner.") :]
        if name.startswith("prev-") or name == "pending":
            continue
        entry = base.read_lease(path)
        if not cotenant_released(entry):
            raise ValueError(
                f"GPU{gpu} co-tenant {path.name} has status {entry.get('status')!r}; wait for its release"
            )


def write_lease(
    gpu: int, fields: dict, root: Path = LEASE_ROOT, shared: str | None = None
) -> None:
    """Update the track's lease; ``shared`` adds/removes a co-located job."""
    owner = root / f"gpu{gpu}.lock" / "owner"
    if not owner.is_file():
        raise ValueError(f"GPU{gpu} lease is missing; take it with the lease command")
    current = base.read_lease(owner)
    if current.get("track") != TRACK:
        raise ValueError(f"GPU{gpu} lease belongs to {current.get('track')}")
    if shared is not None:
        jobs = shared_jobs(current.get("shared_containers"))
        jobs.add(shared) if fields.get("status") == "running" else jobs.discard(shared)
        fields = {"shared_containers": sorted(jobs)}
    elif fields.get("status") == "running" and current.get("status") == "running":
        raise ValueError(f"GPU{gpu} already runs {current.get('container')}")
    current.update(fields)
    write_owner(owner, current)


def take_lease(
    gpu: int,
    *,
    purpose: str,
    expected_end: str | None,
    status: str,
    root: Path = LEASE_ROOT,
    now: str | None = None,
) -> dict:
    """Make the main owner file ours (moving an idle foreign owner aside) and set its status."""
    if status not in LEASE_STATUSES:
        raise ValueError(f"lease status must be one of {LEASE_STATUSES}")
    lease = root / f"gpu{gpu}.lock"
    owner = lease / "owner"
    stamp = now or base.utc()
    moved = None
    current: dict = {}
    if owner.is_file():
        current = base.read_lease(owner)
        if current.get("track") != TRACK:
            if not idle_status(current.get("status")):
                raise ValueError(
                    f"GPU{gpu} owner {current.get('track')} is {current.get('status')!r}, not idle"
                )
            moved = lease / f"owner.prev-{stamp.replace('-', '').replace(':', '')}"
            if moved.exists():
                raise FileExistsError(f"{moved} exists")
            os.replace(owner, moved)
            current = {}
        elif current.get("status") == "running" and current.get("container"):
            raise ValueError(
                f"GPU{gpu} runs {current.get('container')}; its launcher releases it"
            )
    elif not lease.is_dir():
        raise ValueError(f"GPU{gpu} has no lease directory {lease}")
    fields = {
        **current,
        "track": TRACK,
        "purpose": purpose,
        "start_utc": current.get("start_utc") or stamp,
        "expected_end": expected_end or current.get("expected_end") or "unknown",
        "status": status,
        "updated_utc": stamp,
    }
    write_owner(owner, fields)
    return {"gpu": gpu, "moved_previous_owner": str(moved) if moved else None, **fields}


def create_argv(args: argparse.Namespace, devices: list[Path]) -> list[str]:
    visible = ",".join(str(index) for index in range(len(devices)))
    argv = [
        "create",
        "--name",
        args.name,
        "--network",
        "none",
        "--shm-size",
        "16g" if len(devices) == 1 else MULTI_GPU_SHM,
        "--device",
        "/dev/kfd",
    ]
    for device in devices:
        argv.extend(["--device", str(device)])
    if len(devices) > 1:
        argv.extend(["--ulimit", "memlock=-1:-1"])
    argv.extend(
        [
            "-e",
            f"ROCR_VISIBLE_DEVICES={visible}",
            "-e",
            f"HIP_VISIBLE_DEVICES={visible}",
            "-e",
            f"CUDA_VISIBLE_DEVICES={visible}",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "TOKENIZERS_PARALLELISM=false",
        ]
    )
    if len(devices) > 1:
        # RCCL and the c10d store bootstrap over sockets; --network none leaves only loopback.
        argv.extend(["-e", "NCCL_SOCKET_IFNAME=lo", "-e", "GLOO_SOCKET_IFNAME=lo"])
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


def gpu_hours(elapsed_seconds: float, gpus: int) -> float:
    return round(elapsed_seconds / 3600 * gpus, 5)


def build_receipt(
    args: argparse.Namespace,
    gpus: list[int],
    devices: list[Path],
    argv: list[str],
    *,
    cid: str,
    started_utc: str,
    elapsed: float,
    timed_out: bool,
    exit_code: int | None,
    log_sha256: str | None,
) -> dict:
    return {
        "schema_version": "decision2-27b-m4b-launch-receipt/1",
        "track": TRACK,
        "name": args.name,
        "purpose": args.purpose,
        "gpu_indices": gpus,
        "pci_buses": [ALLOWED_GPUS[gpu][0] for gpu in gpus],
        "render_nodes": [device.name for device in devices],
        "image_id": IMAGE_ID,
        "container_id": cid,
        "docker_create_argv": argv,
        "started_utc": started_utc,
        "finished_utc": base.utc(),
        "elapsed_seconds": round(elapsed, 3),
        "gpu_hours": gpu_hours(elapsed, len(gpus)),
        "cap_hours": args.cap_hours,
        "shared_gpu": args.shared,
        "watchdog_fired": timed_out,
        "exit_code": exit_code,
        "container_log_sha256": log_sha256,
    }


def lease_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="launch3 lease", description="Take over or update the M4b main owner files"
    )
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--expected-end", help="UTC time the lease is expected to end")
    parser.add_argument("--status", choices=LEASE_STATUSES, default="reserved-idle")
    args = parser.parse_args(argv)
    gpus = parse_gpus(args.gpus)
    for gpu in gpus:
        render_node(gpu)
    results = [
        take_lease(
            gpu,
            purpose=args.purpose,
            expected_end=args.expected_end,
            status=args.status,
        )
        for gpu in gpus
    ]
    print(json.dumps(results, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["lease"]:
        lease_main(argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--gpus", required=True, help=f"Comma list, subset of {sorted(ALLOWED_GPUS)}"
    )
    parser.add_argument("--cap-hours", type=float, required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mount", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--workdir", default="/code")
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Co-locate a small job on GPUs already running this track's job",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        raise ValueError("Missing container command after --")
    if not 0 < args.cap_hours <= 12:
        raise ValueError("cap-hours must be in (0, 12]")
    if args.receipt.exists():
        raise FileExistsError("Receipt already exists; use a fresh run name")
    gpus = parse_gpus(args.gpus)
    devices = [render_node(gpu) for gpu in gpus]
    if base.docker("image", "inspect", IMAGE_ID, "--format", "{{.Id}}") != IMAGE_ID:
        raise ValueError("Pinned runtime image is absent or changed")
    if base.docker(
        "ps", "-a", "--filter", f"name=^{args.name}$", "--format", "{{.Names}}"
    ):
        raise FileExistsError("Container name already exists")
    for gpu in gpus:
        owner = base.read_lease(LEASE_ROOT / f"gpu{gpu}.lock" / "owner")
        if owner.get("track") != TRACK:
            raise ValueError(f"GPU{gpu} lease belongs to {owner.get('track')}")
        check_cotenants(gpu)
    if not args.shared:
        for gpu in gpus:
            base.wait_until_idle(gpu)
    argv_create = create_argv(args, devices)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    cap = int(args.cap_hours * 3600)
    started_utc = base.utc()
    expected_end = datetime.fromtimestamp(time.time() + cap, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    taken: list[int] = []
    cid = ""
    exit_code = None
    timed_out = threading.Event()
    done = threading.Event()
    start = time.monotonic()
    try:
        for gpu in gpus:
            write_lease(
                gpu,
                {
                    "status": "running",
                    "container": args.name,
                    "purpose": args.purpose,
                    "start_utc": started_utc,
                    "expected_end_utc": expected_end,
                },
                shared=args.name if args.shared else None,
            )
            taken.append(gpu)
        cid = base.docker(*argv_create)
        base.docker("start", cid)

        def watchdog() -> None:
            if not done.wait(cap):
                timed_out.set()
                subprocess.run(
                    ["docker", "stop", "-t", "60", cid], timeout=120, check=False
                )

        thread = threading.Thread(target=watchdog, daemon=True)
        thread.start()
        exit_code = int(base.docker("wait", cid, timeout=cap + 600).splitlines()[-1])
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
        receipt = build_receipt(
            args,
            gpus,
            devices,
            argv_create,
            cid=cid,
            started_utc=started_utc,
            elapsed=elapsed,
            timed_out=timed_out.is_set(),
            exit_code=exit_code,
            log_sha256=base.sha_file(log_path) if log_path.exists() else None,
        )
        fd = os.open(args.receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(receipt, out, indent=1, sort_keys=True)
            out.write("\n")
        for gpu in taken:
            write_lease(
                gpu,
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
