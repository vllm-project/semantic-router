#!/usr/bin/env python3
"""Build immutable Vela Omni artifacts, including mandatory reference parity.

This is an explicit model preparation/build command; never invoke it in serving.
Example: python tools/models/vela_omni/prepare.py --variants nano mini
         --output /opt/router-model-artifacts --work /tmp/vela-omni-build
HF_HOME controls the immutable source cache. Python/Torch and native source are
only needed during this preparation; output contains verified runtime data only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bundle import stage
from contract import VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=["nano"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--keep-golden", action="store_true")
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique")
    for variant in args.variants:
        output = args.work / variant
        command = [
            sys.executable,
            str(Path(__file__).with_name("export.py")),
            "--variant",
            variant,
            "--download",
            "--output",
            str(output),
            "--threads",
            str(args.threads),
        ]
        if args.full_context:
            command.append("--full-context")
        subprocess.run(command, check=True)
        stage(
            output,
            args.output / f"vela-1.0-omni-{variant}",
            keep_golden=args.keep_golden,
        )


if __name__ == "__main__":
    main()
