"""examples.py with every fused kernel of the package raising, so each decoder layer call takes the eager fallback.

    python3 -I fallback_parity.py --fallback-receipt FILE <examples.py arguments with --package DIR>

The package's own ``decision2.fast_kernels`` functions are replaced by one that raises, as a kernel that fails to
compile does; ``decision2/fast.py`` then runs each layer's eager forward for that input shape. The answers must
equal the released runtime's bit for bit (fast.sh --hotfix compares them). FILE records how many kernel calls
failed (each layer and input shape fails once, then stays on the eager forward).
"""

from __future__ import annotations

import atexit
import json
import runpy
import sys
from pathlib import Path

KERNELS = (
    "add_rmsnorm",
    "silu_mul",
    "gdn_prep",
    "gated_rmsnorm",
    "attn_prep",
    "sigmoid_gate",
)


def main() -> None:
    argv = sys.argv[1:]
    at = argv.index("--fallback-receipt")
    receipt = Path(argv[at + 1])
    del argv[at : at + 2]
    package = Path(argv[argv.index("--package") + 1]).resolve()
    sites = [argv[i + 1] for i, a in enumerate(argv) if a == "--site"]
    # The order examples.py gives them: kernel sites, then the package in front.
    for site in reversed(sites):
        sys.path.insert(0, site)
    sys.path.insert(0, str(package))
    import decision2.fast_kernels as kernels

    failed: dict[str, int] = {name: 0 for name in KERNELS}

    def failing(name: str):
        def fail(*args, **kwargs):
            failed[name] += 1
            raise RuntimeError(f"{name}: fused kernel disabled (fallback parity)")

        return fail

    for name in KERNELS:
        setattr(kernels, name, failing(name))
    atexit.register(
        lambda: receipt.write_text(
            json.dumps(
                {
                    "schema": "dev2-fallback-parity/1",
                    "package": str(package),
                    "failed_kernel_calls": failed,
                    "total": sum(failed.values()),
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
    )
    examples = Path(__file__).resolve().parents[3] / "examples.py"
    sys.argv = [str(examples), *argv]
    runpy.run_path(str(examples), run_name="__main__")


if __name__ == "__main__":
    main()
