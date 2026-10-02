#!/usr/bin/env python3
"""Prepare supported Omni variants once and revalidate every reused bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / "tools/models/vela_omni"


def fingerprint() -> str:
    digest = hashlib.sha256()
    inputs = [*PREPARATION.rglob("*"), ROOT / ".dockerignore"]
    for path in sorted(inputs):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def verify(artifact: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(PREPARATION / "bundle.py"), "verify", str(artifact)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def prepare(output: Path, variants: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    identity = fingerprint()
    runtime = shlex.split(os.environ.get("CONTAINER_RUNTIME") or "docker")
    for variant in variants:
        name = "vela-1.0-omni-" + variant
        artifact = output / name
        receipt = output / (name + ".preparation.json")
        expected = {"variant": variant, "inputs_sha256": identity}
        try:
            cached = json.loads(receipt.read_text())
        except (OSError, ValueError):
            cached = None
        if cached == expected and verify(artifact):
            print(f"Reusing verified {name}", flush=True)
            continue
        # Export each variant independently: adding Mini does not re-export Nano,
        # and a later calibration invocation reuses the same verified Nano bytes.
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
            subprocess.run(
                [
                    *runtime,
                    "build",
                    "-f",
                    str(PREPARATION / "Dockerfile"),
                    "--build-arg",
                    "VELA_OMNI_VARIANTS=" + variant,
                    "--output",
                    "type=local,dest=" + temporary,
                    ".",
                ],
                cwd=ROOT,
                check=True,
            )
            prepared = Path(temporary) / name
            if not verify(prepared):
                raise ValueError(f"prepared {name} failed pinned bundle verification")
            if artifact.exists():
                shutil.rmtree(artifact)
            prepared.rename(artifact)
            receipt.write_text(json.dumps(expected, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=("nano", "mini"), required=True
    )
    args = parser.parse_args()
    prepare(args.output.resolve(), list(dict.fromkeys(args.variants)))


if __name__ == "__main__":
    main()
