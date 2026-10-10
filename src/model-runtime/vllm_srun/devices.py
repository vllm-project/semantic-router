"""``vllm-srun devices``: the devices of this host and the one ``--device auto`` takes.

The router asks before it groups its managed deployments into processes, so
``auto`` resolves the way placement resolves it: through the accelerator
plugins and their ``auto_priority``.
"""

from __future__ import annotations

from typing import Any

from .placement import device_kind
from .plugins import registry


def auto_device() -> str:
    """The device ``--device auto`` tries first: the first device of the accelerator it lands on.

    Placement may still move a model to a later device that has the memory it needs.
    """
    kind = device_kind("auto")
    devices = registry.instantiate("accelerators", kind).devices()
    return devices[0].label if devices else kind


def report() -> dict[str, Any]:
    """``auto`` and the devices of every available accelerator, by accelerator name."""
    labels = []
    for name in registry.names("accelerators"):
        accelerator = registry.instantiate("accelerators", name)
        if accelerator.available():
            labels += [device.label for device in accelerator.devices()]
    return {"auto": auto_device(), "devices": labels}
