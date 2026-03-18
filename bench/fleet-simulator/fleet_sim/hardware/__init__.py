"""Hardware specifications for GPU fleet simulation."""
from .spec import HardwareSpec, MEM_BW_SCALE, MEM_CONST_LAT, P2P_LATENCY
from .catalog import (
    A100_SXM, H100_SXM, H200_SXM, B200_SXM, GB200, GB300, L40S, B60,
    get as get_hardware, list_names as list_hardware,
)

__all__ = [
    "HardwareSpec",
    "MEM_BW_SCALE", "MEM_CONST_LAT", "P2P_LATENCY",
    "A100_SXM", "H100_SXM", "H200_SXM", "B200_SXM", "GB200", "GB300",
    "L40S", "B60",
    "get_hardware", "list_hardware",
]
