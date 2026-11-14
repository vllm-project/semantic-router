"""Print the registry pin of a local Vela 3.1 pico package: file digests, identity, loaded parameters, peak memory.

    python3 tools/vela3_pico_pins.py --package DIR [--device cpu] > pins.json

The package is verified as an unpinned local package (its ``SHA256SUMS`` must list and match every file the
family loads), loaded through the vela3_pico family on the native engine, and answers the family's golden
request once. The output is what ``registry/tables/vela3_pico.py`` pins for the revision the directory holds:
``files`` (SHA-256 of every loaded file), ``model_sha256``, ``loaded_parameters`` and the process's peak
resident memory in GiB, from which ``min_device_memory_gib`` is set with headroom.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_srun.accel.cpu import CPUAccelerator  # noqa: E402
from vllm_srun.accel.cuda import CUDAAccelerator  # noqa: E402
from vllm_srun.accel.rocm import ROCmAccelerator  # noqa: E402
from vllm_srun.families.vela3_pico.family import (  # noqa: E402
    GOLDEN_QUESTIONS,
    GOLDEN_STATE,
    Vela3PicoFamily,
)
from vllm_srun.plugins.base import EngineOptions, PackageRef, SurfacePlan  # noqa: E402
from vllm_srun.plugins.registry import instantiate  # noqa: E402

ACCELERATORS = {"cpu": CPUAccelerator, "cuda": CUDAAccelerator, "rocm": ROCmAccelerator}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    family = Vela3PicoFamily()
    verified = family.verify(PackageRef(args.package.resolve()))
    spec = family.describe(verified)
    accelerator = ACCELERATORS[args.device.split(":", maxsplit=1)[0]]()
    device = accelerator.devices()[
        int(args.device.split(":")[1]) if ":" in args.device else 0
    ]
    engine_model = instantiate("engines", "native").load(
        spec, accelerator, device, EngineOptions()
    )
    model = family.load(verified, spec, engine_model)
    plan = model.plan(GOLDEN_STATE, GOLDEN_QUESTIONS)
    response = model.finish_surface(
        SurfacePlan("decisions", plan.items, plan.input_tokens, plan),
        model.run(plan.items),
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = peak / (1 << 30) if sys.platform == "darwin" else peak / (1 << 20)
    details = verified.details["package"]
    print(
        json.dumps(
            {
                "name": details.pico.get("name"),
                "files": details.files,
                "model_sha256": verified.model_sha256,
                "loaded_parameters": model.info.parameters,
                "peak_rss_gib": round(peak_gib, 2),
                "golden_one_pass": plan.one_pass is not None,
                "golden_input_tokens": response["usage"]["input_tokens"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
