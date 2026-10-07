"""Print a Vela 2.0 package's built-in table entry and its golden answers on this host's device class.

    python3 tools/vela2_table.py --package DIR --repo vllm-sr/Vela-2.0-4B --revision SHA [--device rocm:0]

The entry (for ``registry/tables/vela2.py``) holds the SHA-256 of every file the
family loads, the identity those digests define and the parameter count the
family loads; the golden answers (for ``registry/golden_answers_vela2.json``)
are the exact profile's answers to the family's golden request on the device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))
sys.path.insert(0, str(TOOLS))

from vela2_parity import (  # noqa: E402
    answer,
    device_executor,
    load_runtime,
    pin_choices,
)
from vllm_srun.families.vela2.family import (  # noqa: E402
    GOLDEN_QUESTIONS,
    GOLDEN_STATE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pinned = pin_choices(args.package, args.device)
    execute = device_executor(args.device)
    model = execute(lambda: load_runtime(args.package, args.device))
    response = execute(lambda: answer(model, GOLDEN_STATE, GOLDEN_QUESTIONS)[0])
    package = model.package
    print(
        json.dumps(
            {
                "repo_id": args.repo,
                "revision": args.revision,
                "member": package.member,
                "model_sha256": package.model_sha256,
                "manifest_sha256": package.manifest_sha256,
                "loaded_parameters": model.info.parameters,
                "files": package.files,
                "device_class": args.device.split(":", maxsplit=1)[0],
                "kernel_choices": "pinned" if pinned else "none",
                "golden_answers": response["answers"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
