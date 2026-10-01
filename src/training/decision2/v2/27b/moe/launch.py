"""Single-GPU container launcher for the 27B MoE milestone (host side, stdlib only).

``v2.27b.launch`` with this milestone's track and GPUs: node A GPU3-5 (``DEV2_NODE=a``)
and node B GPU6-7 (``DEV2_NODE=b``, the default). Both nodes share one PCI layout;
the render node's PCI address and the lease file (track ``27b-moe``) are checked
before every launch, and every launch writes a GPU-hour receipt.
"""

from __future__ import annotations

import importlib

base = importlib.import_module("v2.27b.launch")

TRACK = "27b-moe"
NODE_GPUS = {
    "a": {
        3: ("0000:9b:00.0", "renderD153"),
        4: ("0000:a3:00.0", "renderD161"),
        5: ("0000:ab:00.0", "renderD169"),
    },
    "b": {
        6: ("0000:b3:00.0", "renderD177"),
        7: ("0000:bb:00.0", "renderD185"),
    },
}


# Amendment 1: a full MoE attempt needs up to about 20 GPU-hours at the probe's speed.
MAX_CAP_HOURS = 21.0


def configure() -> None:
    base.TRACK = TRACK
    base.NODE_GPUS = NODE_GPUS
    base.ALLOWED_GPUS = NODE_GPUS["b"]
    base.MAX_CAP_HOURS = MAX_CAP_HOURS


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
