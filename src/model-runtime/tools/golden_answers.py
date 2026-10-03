"""Print a model's golden answers on this host's device class, for registry/builtin.py.

    python3 tools/golden_answers.py vllm-sr/Decision-2.0-Kai-0.6B --device cpu
    python3 tools/golden_answers.py vllm-sr/Decision-2.0-Lux-9B --device rocm:0 \\
        --autotune-cache /var/cache/vllm-sr-runtime/autotune
"""

from __future__ import annotations

import argparse
import json

from vllm_sr_runtime.config import ServeConfig
from vllm_sr_runtime.families.decision2.family import GOLDEN_QUESTIONS, GOLDEN_STATE
from vllm_sr_runtime.runtime import Runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--base-path")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--autotune-cache")
    args = parser.parse_args()
    runtime = Runtime(
        ServeConfig(
            model=args.model,
            revision=args.revision,
            device=args.device,
            cache_dir=args.cache_dir,
            base_path=args.base_path,
            offline=args.offline,
            autotune_cache=args.autotune_cache,
        )
    )
    try:
        runtime.load()
        assert runtime.placement is not None and runtime.model is not None
        answers = runtime._golden_run(GOLDEN_STATE, GOLDEN_QUESTIONS)
        print(
            json.dumps(
                {
                    "model": runtime.model.info.id,
                    "revision": runtime.model.info.revision,
                    "device": runtime.placement.device.label,
                    "golden_answers": {runtime.placement.device.accelerator: answers},
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
