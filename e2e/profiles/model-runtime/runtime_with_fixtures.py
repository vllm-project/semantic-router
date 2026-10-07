#!/usr/bin/env python3
"""Write the profile's tiny fixture packages, then run the runtime command.

The Router starts its managed runtimes through VLLM_SRUN_COMMAND, and the
attached runtime pod starts through this script as well, so every runtime of
the profile serves the same deterministic random-weight packages without
downloading a model. The Router may start several processes at once; a file
lock lets one of them write each package, and packages survive restarts in the
pod's /tmp volume. The script needs only the standard library: it drives the
runtime's own command line, wherever the image installed it.
"""

import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("VSR_E2E_FIXTURES", "/tmp/vsr-fixtures"))
# The CPU runtime image puts the runtime on PATH; router images keep it in its
# own environment.
RUNTIME_COMMANDS = ("vllm-srun", "/opt/vllm-srun/bin/vllm-srun")
# name: (family, variant, seed); values.yaml and the attached models file name
# these directories.
PACKAGES = {
    "decision": ("decision2", "qwen3", 0),
    "domain": ("task_heads", "sequence", 1),
    "pii": ("task_heads", "token", 2),
    "guard": ("task_heads", "guard", 3),
    "feedback": ("task_heads", "feedback", 4),
    "decision-attached": ("decision2", "qwen3", 5),
    "embedding": ("task_heads", "embedding", 6),
    "reranker": ("task_heads", "reranker", 7),
    "modality": ("task_heads", "modality", 8),
    "vela2-attached": ("vela2", "encoder", 9),
}


def runtime_command() -> str:
    for candidate in RUNTIME_COMMANDS:
        found = shutil.which(candidate)
        if found:
            return found
    sys.exit(f"no model runtime found; looked for {', '.join(RUNTIME_COMMANDS)}")


def write_packages(runtime: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, (family, variant, seed) in PACKAGES.items():
            target = ROOT / name
            if target.exists():
                continue
            staging = ROOT / f".{name}.partial"
            shutil.rmtree(staging, ignore_errors=True)
            subprocess.run(
                [
                    runtime,
                    "fixture",
                    str(staging),
                    "--family",
                    family,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            staging.rename(target)


if __name__ == "__main__":
    runtime = runtime_command()
    write_packages(runtime)
    os.execv(runtime, [runtime, *sys.argv[1:]])
