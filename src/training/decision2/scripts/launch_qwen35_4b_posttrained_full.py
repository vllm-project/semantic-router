"""Single-shot host supervisor for the sealed Qwen3.5-4B Posttrained arm.

Dry-run is the default. ``--execute`` requires separate operator approval.
Only private paths are accepted at runtime; no host inventory is embedded.
The supervisor must be run under a persistent host service or ``nohup`` so
its watchdog remains alive after an SSH disconnect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

IMAGE_ID = "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54"
LOCK_SHA256 = "7a3fe34eef8a4845edb71bead0aa12660993d0b81ad3a99184714f69e37809cd"
MIRROR_SHA256 = "8d7809920cd48f38b9f2980f2e888006ae3f07753714ec666a8cec6fff009118"
CONTAINER_NAME = "decision2-qwen35-4b-posttrained-full-v1"
MAX_SECONDS = 10_800


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_sha(root: Path) -> str:
    files = {
        str(path.relative_to(root)): sha(path)
        for path in root.rglob("*.py")
        if path.parts[-2] in ("scripts", "model")
        and ("/training/model/" in str(path) or "/scripts/" in str(path))
    }
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_once(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def _docker(*args: str, timeout: int = 120) -> str:
    return subprocess.check_output(
        ["docker", *args], text=True, timeout=timeout, stderr=subprocess.STDOUT
    ).strip()


def _private_lock(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Private full-arm lock must be a regular file")
    if (
        stat.S_IMODE(path.stat().st_mode) != 0o600
        or stat.S_IMODE(path.parent.stat().st_mode) != 0o700
        or sha(path) != LOCK_SHA256
    ):
        raise ValueError("Private full-arm lock mode/hash differs")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("status") != "LOCKED_NO_GPU"
        or lock.get("image_id") != IMAGE_ID
        or lock.get("planned_updates") != 466
        or lock.get("gpu_hour_cap") != 3.0
        or lock.get("checkpoint_steps") != [64, 128, 192, 256, 320, 384, 448, 466]
        or lock.get("runtime", {}).get("python") != "/usr/bin/python"
        or lock.get("source_revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or lock.get("private_paths")
        != {
            "train": "/data/rights_clean.train.jsonl",
            "select": "/data/select.jsonl",
            "cal": "/data/cal.jsonl",
            "source": "/source",
            "arm_output": "/arm/out",
        }
    ):
        raise ValueError("Full-arm lock schema or training boundary differs")
    argv = lock["trainer_argv"]
    if (
        argv[:3] != ["/usr/bin/python", "-m", "training.model.train"]
        or "--resume" in argv
        or "--max-steps" in argv
        or "--zero-step-only" in argv
        or "--replay" in argv
        or "--inline-teacher" in argv
        or argv[argv.index("--cal") + 1] != "/data/cal.jsonl"
    ):
        raise ValueError("Sealed trainer argv differs")
    return lock


def prepare(args: argparse.Namespace) -> tuple[list[str], dict]:
    lock = _private_lock(args.lock)
    code, source, data, arm = (
        getattr(args, name).resolve() for name in ("code", "source", "data", "arm")
    )
    if (
        mirror_sha(code) != MIRROR_SHA256
        or sha(code / "scripts/prepare_qwen35_4b_posttrained_full.py")
        != lock["lock_preparer_sha256"]
    ):
        raise ValueError("Exact code mirror differs from sealed CPU lock")
    trainer = code / "training/model"
    if any(
        sha(trainer / name) != digest
        for name, digest in lock["trainer_code_sha256"].items()
    ):
        raise ValueError("Trainer source differs from completed Base control")
    if not args.render_device.is_char_device() or not Path("/dev/kfd").is_char_device():
        raise ValueError("Selected ROCm device nodes are absent")
    if args.render_device.name != "renderD129":
        raise ValueError("Sealed first-node GPU0 render device differs")
    if _docker("image", "inspect", IMAGE_ID, "--format", "{{.Id}}") != IMAGE_ID:
        raise ValueError("Pinned runtime image is absent or changed")
    for name, root in (("source", source), ("data", data)):
        if not root.is_dir():
            raise ValueError(f"{name} bind source is absent")
    for role, name in (
        ("train", "rights_clean.train.jsonl"),
        ("select", "select.jsonl"),
        ("cal", "cal.jsonl"),
    ):
        if sha(data / name) != lock["data_sha256"][role]:
            raise ValueError("Data bytes differ from sealed split")
    if any(
        sha(source / name) != digest
        for name, digest in lock["source_fingerprint"]["files_sha256"].items()
    ):
        raise ValueError("Official source bytes differ from sealed revision")
    if arm.exists():
        raise FileExistsError("Fresh arm directory already exists")
    if _docker(
        "ps", "-a", "--filter", f"name=^{CONTAINER_NAME}$", "--format", "{{.Names}}"
    ):
        raise FileExistsError("Single-shot container name already exists")
    command = [
        "docker",
        "create",
        "--name",
        CONTAINER_NAME,
        "--network",
        "none",
        "--shm-size",
        "16g",
        "--device",
        "/dev/kfd",
        "--device",
        str(args.render_device),
        "-e",
        "ROCR_VISIBLE_DEVICES=0",
        "-e",
        "PYTHONPATH=/code",
    ]
    for host, dest, readonly in (
        (code, "/code", True),
        (source, "/source", True),
        (data, "/data", True),
        (arm, "/arm", False),
    ):
        command.extend(
            [
                "--mount",
                f"type=bind,src={host},dst={dest}" + (",readonly" if readonly else ""),
            ]
        )
    command.extend(["-w", "/code", IMAGE_ID, *lock["trainer_argv"]])
    return command, lock


def execute(args: argparse.Namespace, command: list[str], lock: dict) -> int:
    arm = args.arm.resolve()
    arm.mkdir(mode=0o700)
    if stat.S_IMODE(arm.stat().st_mode) != 0o700 or any(arm.iterdir()):
        raise ValueError("Fresh private arm output must be empty and mode 0700")
    cid = _docker(*command[1:])
    start = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    timed_out = threading.Event()
    done = threading.Event()

    def watchdog() -> None:
        if not done.wait(MAX_SECONDS):
            timed_out.set()
            subprocess.run(["docker", "stop", "-t", "30", cid], timeout=60, check=False)

    thread = threading.Thread(target=watchdog, name="qwen4b-wall-cap", daemon=False)
    thread.start()
    try:
        _docker("start", cid)
        exit_text = _docker("wait", cid, timeout=MAX_SECONDS + 120)
        exit_code = int(exit_text.splitlines()[-1])
    except BaseException:
        subprocess.run(["docker", "stop", "-t", "30", cid], timeout=60, check=False)
        raise
    finally:
        done.set()
        thread.join(timeout=70)
        logs = subprocess.run(["docker", "logs", cid], capture_output=True)
        log_path = arm / "container.log"
        fd = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(logs.stdout + logs.stderr)
        subprocess.run(
            ["docker", "rm", cid], timeout=60, check=False, capture_output=True
        )
    elapsed = time.monotonic() - start
    complete = arm / "out/COMPLETE.json"
    provenance = arm / "out/provenance.json"
    result = {
        "schema_version": "decision2-qwen35-4b-posttrained-full-launch/1",
        "lock_sha256": LOCK_SHA256,
        "mirror_sha256": MIRROR_SHA256,
        "image_id": IMAGE_ID,
        "container_id": cid,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600,
        "watchdog_fired": timed_out.is_set(),
        "exit_code": exit_code,
        "container_log_sha256": sha(arm / "container.log"),
        "complete_sha256": sha(complete) if complete.exists() else None,
        "provenance_sha256": sha(provenance) if provenance.exists() else None,
        "status": (
            "COMPLETE_UNSELECTED"
            if exit_code == 0 and complete.exists() and not timed_out.is_set()
            else "HOLD"
        ),
    }
    _write_once(arm / "launch-receipt.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "elapsed_seconds",
                    "gpu_hours",
                    "watchdog_fired",
                    "exit_code",
                    "complete_sha256",
                    "provenance_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "COMPLETE_UNSELECTED" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "code", "source", "data", "arm", "render_device"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command, lock = prepare(args)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_NO_GPU",
                    "docker_create_argv": command,
                    "watchdog_seconds": MAX_SECONDS,
                },
                indent=2,
            )
        )
        return
    raise SystemExit(execute(args, command, lock))


if __name__ == "__main__":
    main()
