"""Hardware specifications for GPU fleet simulation."""

from .catalog import (
    A100_SXM,
    B60,
    B200_SXM,
    GB200,
    GB300,
    H100_SXM,
    H200_SXM,
    L40S,
)
from .catalog import (
    get as get_hardware,
)
from .catalog import (
    list_names as list_hardware,
)
from .spec import MEM_BW_SCALE, MEM_CONST_LAT, P2P_LATENCY, HardwareSpec

__all__ = [
    "A100_SXM",
    "B60",
    "B200_SXM",
    "GB200",
    "GB300",
    "H100_SXM",
    "H200_SXM",
    "L40S",
    "MEM_BW_SCALE",
    "MEM_CONST_LAT",
    "P2P_LATENCY",
    "HardwareSpec",
    "get_hardware",
    "list_hardware",
]
