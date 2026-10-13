"""Print a model's autotuned kernel choices from a Triton autotune cache, for registry/kernel_choices.json.

    python3 tools/kernel_choices.py vllm-sr/Decision-2.0-Eos-0.8B --autotune-cache RELEASE_CACHE_DIR \\
        --device-class rocm:gfx942 --fla 0.5.2 --triton 3.7.1

The cache is the one the model's released runtime ran with (``TRITON_CACHE_DIR`` with
``TRITON_CACHE_AUTOTUNING=1``, as in the parity records). Every forward kernel's cached choice per
tuning key (the fastest timed configuration) is kept; backward kernels are dropped. Entries are
ordered by tuning key, and the first entry of each kernel is its choice for keys the release never
saw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIG_FIELDS = (
    "kwargs",
    "num_warps",
    "num_stages",
    "num_ctas",
    "maxnreg",
    "ir_override",
)
SUFFIX = ".autotune.json"


def timing(value):
    return value[0] if isinstance(value, list) else value


def choices(directory: Path) -> dict[str, list[dict]]:
    kernels: dict[str, list[dict]] = {}
    for path in sorted(directory.rglob(f"*{SUFFIX}")):
        name = path.name[: -len(SUFFIX)]
        if "bwd" in name:
            continue
        cached = json.loads(path.read_text(encoding="utf-8"))
        best = min(cached["configs_timings"], key=lambda row: timing(row[1]))[0]
        config = {field: best.get(field) for field in CONFIG_FIELDS}
        kernels.setdefault(name, []).append({"key": cached["key"], "config": config})
    for name, entries in kernels.items():
        entries.sort(key=lambda entry: json.dumps(entry["key"]))
        keys = [json.dumps(entry["key"]) for entry in entries]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{name}: one tuning key has two cached choices")
    return dict(sorted(kernels.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--device-class", required=True)
    parser.add_argument("--fla", required=True)
    parser.add_argument("--triton", required=True)
    args = parser.parse_args()
    kernels = choices(args.autotune_cache)
    if not kernels:
        raise SystemExit(f"{args.autotune_cache} holds no autotune choices")
    print(
        json.dumps(
            {
                "model": args.model,
                "device_class": args.device_class,
                "choices": {"fla": args.fla, "triton": args.triton, "kernels": kernels},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
