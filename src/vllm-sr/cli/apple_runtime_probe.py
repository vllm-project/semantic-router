"""Run with the image-matched native interpreter, not the CLI interpreter.

Use the runtime's own package verification and placement, without loading an
engine or allocating model weights. Do not duplicate family dtype policies here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from vllm_srun.config import ServeConfig, load_models_file
from vllm_srun.runtime import Runtime


def main() -> int:
    models = load_models_file(Path(sys.argv[1]))
    runtime = Runtime(ServeConfig(models=models, cache_dir=sys.argv[2]))
    failures = []
    for served in runtime.served:
        try:
            served.prepare()
        except Exception as error:
            detail = str(error)
            if "no BF16 support" in detail:
                detail += (
                    "; set this deployment's dtype: float32 explicitly, or device: cpu. "
                    "FP32 weights require 4 bytes per parameter (twice BF16 weight storage), "
                    "plus activations and temporary allocations; MPS goldens remain unverified."
                )
            failures.append(f"{served.label}: {detail}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
