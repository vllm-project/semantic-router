"""examples.py with the package's HIP graph cache cut to a few graphs, so almost every input shape finds it full.

    python3 -I full_cache_parity.py --graph-cache N --cache-receipt FILE <examples.py arguments with --package DIR>

The package's own ``decision2.fast.Graphs`` keeps at most N graphs instead of 512: the first N shapes seen twice
are captured and keep replaying, every other shape runs the eager forward, and no graph is ever destroyed. The
answers must equal the released runtime's bit for bit (fast.sh --evict compares them). FILE records the
statistics of every graph cache the run installed (captures, replays, eager forwards, refused captures).
"""

from __future__ import annotations

import atexit
import json
import runpy
import sys
import types
from pathlib import Path


def option(argv: list[str], name: str) -> str:
    at = argv.index(name)
    value = argv[at + 1]
    del argv[at : at + 2]
    return value


def main() -> None:
    argv = sys.argv[1:]
    limit = int(option(argv, "--graph-cache"))
    receipt = Path(option(argv, "--cache-receipt"))
    package = Path(argv[argv.index("--package") + 1]).resolve()
    sites = [argv[i + 1] for i, a in enumerate(argv) if a == "--site"]
    # The order examples.py gives them: kernel sites, then the package in front.
    for site in reversed(sites):
        sys.path.insert(0, site)
    sys.path.insert(0, str(package))
    from decision2 import fast

    if "full" not in fast.Graphs(types.SimpleNamespace(forward=None), None, None).stats:
        raise SystemExit(f"{package}: its graph cache still evicts")
    caches: list = []
    init = fast.Graphs.__init__

    def small(self, *args, **kwargs):
        kwargs["max_graphs"] = limit
        init(self, *args, **kwargs)
        caches.append(self)

    fast.Graphs.__init__ = small
    atexit.register(
        lambda: receipt.write_text(
            json.dumps(
                {
                    "schema": "dev2-full-cache-parity/1",
                    "package": str(package),
                    "max_graphs": limit,
                    "caches": [
                        {**cache.stats, "cached": len(cache.graphs)} for cache in caches
                    ],
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
