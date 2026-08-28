#!/usr/bin/env python3
"""Generate reference scorecards for all MoM V1 entrypoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "config/recipes/built-in/latest/mom-v1/mom-evaluation.yaml"

ENTRYPOINTS = [
    "vllm-sr/mom-v1-blend",
    "vllm-sr/mom-v1-lite",
    "vllm-sr/mom-v1-flash",
    "vllm-sr/mom-v1-ultra",
    "vllm-sr/mom-v1-vault",
]


def main() -> int:
    for entrypoint in ENTRYPOINTS:
        slug = entrypoint.split("/")[-1]
        output_dir = (
            REPO_ROOT / "config/evaluation/scorecards/mom-v1" / slug / "1.0.0"
        )
        cmd = [
            sys.executable,
            str(REPO_ROOT / "bench/mom_eval/run_mom_eval.py"),
            "--manifest",
            str(MANIFEST),
            "--entrypoint",
            entrypoint,
            "--run-mode",
            "smoke",
            "--synthesize",
            "--output-dir",
            str(output_dir),
        ]
        print(f"generating reference scorecard for {entrypoint}")
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
