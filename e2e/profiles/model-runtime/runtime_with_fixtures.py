#!/usr/bin/env python3
"""Write the profile's tiny fixture packages, then run the runtime command.

The Router starts its managed runtimes through VLLM_SR_RUNTIME_COMMAND, and the
attached runtime pod starts through this script as well, so every runtime of
the profile serves the same deterministic random-weight packages without
downloading a model. The Router may start several processes at once; a file
lock lets one of them write each package, and packages survive restarts in the
pod's /tmp volume.
"""

import fcntl
import os
import shutil
import sys
from pathlib import Path

from vllm_sr_runtime.testing.fixtures import write_fixture

ROOT = Path(os.environ.get("VSR_E2E_FIXTURES", "/tmp/vsr-fixtures"))
# name: (family, variant, seed); values.yaml and the attached models file name
# these directories.
PACKAGES = {
    "decision": ("decision2", "qwen3", 0),
    "domain": ("task_heads", "sequence", 1),
    "pii": ("task_heads", "token", 2),
    "guard": ("task_heads", "guard", 3),
    "feedback": ("task_heads", "feedback", 4),
    "decision-attached": ("decision2", "qwen3", 5),
}


def write_packages() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, (family, variant, seed) in PACKAGES.items():
            target = ROOT / name
            if target.exists():
                continue
            staging = ROOT / f".{name}.partial"
            shutil.rmtree(staging, ignore_errors=True)
            write_fixture(staging, family=family, variant=variant, seed=seed)
            staging.rename(target)


if __name__ == "__main__":
    write_packages()
    os.execvp("vllm-sr-runtime", ["vllm-sr-runtime", *sys.argv[1:]])
