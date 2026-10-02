"""Arm factory arms, recipes and data locks (prereg ``v2/af/records/af-prereg-2026-10-02.md``).

Every arm is an owner's recipe (4B: M17 stage 2; 9B: M10's K-a13IB recipe) with one stated change. The trainer
arguments are COMMON[size] + the arm's locked data flags + the arm's EXTRA flags (later flags win in argparse).
Data locks live on each node at ``/data/dev2/runs/af/<size>/data/READY-af.json``: for every data key the files
(relative to ``/data/dev2/runs/af``) with SHA-256 and the trainer's data flags (container paths, ``/runs`` =
``/data/dev2/runs/af``). A lock entry is written once and never changed.

usage:
  af_arms.py recipe <ARM>                    print "<size> <start> <trainer args...>"
  af_arms.py locked <ARM>                    exit 0 iff the arm's data key has a lock entry
  af_arms.py ready <ARM>                     exit 0 iff every locked file of the arm's data hashes as locked
  af_arms.py lock <size> <DATA> <args-json> <rel>=<sha256> [...]   add a data entry (each file is re-hashed)
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path("/data/dev2/runs/af")

REV_4B = "1001bb4d826a52d1f399e183466143f4da7b741b"
START = {"4b": f"/models/Qwen--Qwen3.5-4B-Base/{REV_4B}", "9b": "/lux"}
COMMON = {
    # M17 stage 2 (m17-chains.sh args_for), unchanged.
    "4b": (
        "--teacher-kl-weight 1.0 --teacher-partial --batching tokens --max-batch-tokens 32768 --max-batch-rows 64"
        f" --update-rows 64 --init base --revision {REV_4B} --train-mode lora --lora-rank 128 --lora-alpha 256"
        " --lora-dropout 0.05 --lora-lr 1e-4 --head-lr 1e-4 --head-init-seed 20261001"
    ).split(),
    # M10 (chains.sh COMMON), unchanged: the K-a13IB recipe.
    "9b": (
        "--teacher-kl-weight 1.0 --teacher-partial --brier-weight 0.5 --weight-decay 0.01 --warmup-ratio 0.05"
        " --epochs 1 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64 --eval-batch 2"
        " --checkpoint-schedule even8 --selection matrix-v1 --init decision1 --train-mode full --backbone-lr 1e-5"
        " --head-lr 1e-4 --max-length 8192"
    ).split(),
}
# arm -> (size, data key, extra trainer flags)
ARMS = {
    # 4B wave 1: more seeds of M17's strongest arms on their locked TRAIN files.
    "4b-LHS17IB4": ("4b", "4b-LHS17IB4", []),
    "4b-SDMLIB4": ("4b", "4b-SDMLIB4", []),
    "4b-LHS17UP": ("4b", "4b-LHS17UP", []),
    # 4B LR variant: 4b-LHS17IB4's TRAIN at half the LoRA / head LR.
    "4b-LHS17IB4-lrh": (
        "4b",
        "4b-LHS17IB4",
        ["--lora-lr", "5e-5", "--head-lr", "5e-5"],
    ),
    # 4B wave 2 (new TRAIN files, af-prep4b.py): IB4 + ML on the S17 base; IB4 on the S23 swap dose.
    "4b-LHS17IB4ML": ("4b", "4b-LHS17IB4ML", []),
    "4b-LHS23IB4": ("4b", "4b-LHS23IB4", []),
    # 9B: KIB4 seeds, an IB4 x2 loss-weight dose and a 2x backbone LR, all on KIB4's TRAIN.
    "KIB4": ("9b", "KIB4", []),
    "KIB4W2": ("9b", "KIB4W2", []),
    "KIB4L2": ("9b", "KIB4", ["--backbone-lr", "2e-5"]),
    # Batch 2 (amendment 4). 4B: more seeds of M17's 4b-LHS17IB4X / 4b-LHS17ML; 4b-SDMLIB4's TRAIN at half LR.
    "4b-LHS17IB4X": ("4b", "4b-LHS17IB4X", []),
    "4b-LHS17ML": ("4b", "4b-LHS17ML", []),
    "4b-SDMLIB4-lrh": ("4b", "4b-SDMLIB4", ["--lora-lr", "5e-5", "--head-lr", "5e-5"]),
    # 9B: KIB4's TRAIN with the IB4 rows x3; KIB4's construction with a new x60 cut seed (af-prep9b.sh).
    "KIB4W3": ("9b", "KIB4W3", []),
    "KIB4R": ("9b", "KIB4R", []),
    # Batch 3 (amendment 5). 4B: 4b-SDMLIB4's TRAIN with the IB4 rows x2; the same TRAIN for two epochs.
    "4b-SDMLIB4W2": ("4b", "4b-SDMLIB4W2", []),
    "4b-SDMLIB4-e2": ("4b", "4b-SDMLIB4", ["--epochs", "2"]),
    # 9B: KIB4's construction with a second new x60 cut seed.
    "KIB4R2": ("9b", "KIB4R2", []),
    # Amendment 7: the S17-base counterparts (two epochs; IB4 rows x2).
    "4b-LHS17IB4-e2": ("4b", "4b-LHS17IB4", ["--epochs", "2"]),
    "4b-LHS17IB4W2": ("4b", "4b-LHS17IB4W2", []),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def lock_path(size: str) -> Path:
    return ROOT / size / "data" / "READY-af.json"


def load(size: str) -> dict:
    path = lock_path(size)
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"schema": "af-ready/1", "data": {}}
    )


def main() -> None:
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "recipe":
        size, key, extra = ARMS[rest[0]]
        entry = load(size)["data"].get(key)
        if entry is None:
            sys.exit(f"{key} is not in {lock_path(size)}")
        print(
            " ".join(
                shlex.quote(x)
                for x in [size, START[size], *entry["args"], *COMMON[size], *extra]
            )
        )
    elif mode == "locked":
        size, key, _ = ARMS[rest[0]]
        sys.exit(0 if key in load(size)["data"] else 1)
    elif mode == "ready":
        size, key, _ = ARMS[rest[0]]
        entry = load(size)["data"].get(key)
        ok = entry is not None and all(
            sha256(ROOT / rel) == want for rel, want in entry["files"].items()
        )
        sys.exit(0 if ok else 1)
    elif mode == "lock":
        size, key, args, *pairs = rest
        lock = load(size)
        if key in lock["data"]:
            sys.exit(f"{key} is already locked")
        files = {}
        for pair in pairs:
            rel, want = pair.split("=", 1)
            got = sha256(ROOT / rel)
            if got != want:
                sys.exit(f"{rel} is {got}, not {want}")
            files[rel] = want
        lock["data"][key] = {"args": json.loads(args), "files": files}
        path = lock_path(size)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(lock, indent=1))
        os.replace(tmp, path)
        print(json.dumps({key: lock["data"][key]}))
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
