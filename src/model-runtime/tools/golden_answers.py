"""Print a model's golden answers on this host's device class, for its family's golden_answers file.

    python3 tools/golden_answers.py vllm-sr/Decision-2.0-Kai-0.6B --device cpu
    python3 tools/golden_answers.py vllm-sr/Decision-1.0-Lux-9B --device rocm:0 \\
        --autotune-cache /var/cache/vllm-sr-runtime/autotune

The model is served as ``vllm-sr-runtime serve`` serves it (verification,
pinned kernel choices, readiness) and answers its family's golden requests on
the exact profile. The output's ``entry`` is the model's entry of
``registry/golden_answers*.json`` for this device class.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from vllm_sr_runtime.config import ServeConfig
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
        served = runtime.primary
        assert served.model is not None and served.placement is not None
        assert served.family is not None and served.package is not None
        answers: dict[str, object] = {}
        for golden in served.family.golden(served.package):
            if "surface" in golden:
                continue
            body = {
                "state": golden["state"],
                "questions": golden["questions"],
                "options": {"profile": "exact", "return_meta": False},
            }
            status, response = asyncio.run(runtime.call("decisions", body))
            if status != 200:
                raise SystemExit(
                    f"golden request failed with HTTP {status}: {response}"
                )
            answers.update(response["answers"])
        info = served.model.info
        print(
            json.dumps(
                {
                    "model": info.id,
                    "revision": info.revision,
                    "model_sha256": info.model_sha256,
                    "device": served.placement.device.label,
                    "readiness": served.health.golden.describe(),
                    "entry": {
                        info.repo: {
                            "revision": info.revision,
                            "answers": {served.placement.device.accelerator: answers},
                        }
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
