"""Hardware specification: GPU and node physical constants.

All values are self-contained — no external dependencies required.
Empirical correction factors are derived from production observations
across Ampere, Hopper, and Blackwell generation hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# Empirical constants (hardware-generation-agnostic)
MEM_BW_SCALE: float = 0.80        # effective fraction of peak memory bandwidth
MEM_CONST_LAT: float = 3e-6       # per-layer constant overhead (seconds)
P2P_LATENCY: float = 10e-6        # GPU-to-GPU P2P transfer base latency (s)
OTHER_MEM: int = 3_758_096_384    # 3.5 GiB safety buffer (CUDA context, etc.)

# NVLink/NVSwitch NCCL workspace memory by TP degree (bytes)
NCCL_MEM: Dict[int, int] = {
    1: 0,
    2: 358_612_992,   # ~342 MiB
    4: 411_041_792,   # ~392 MiB
    8: 411_041_792,   # ~392 MiB
}


@dataclass(frozen=True)
class HardwareSpec:
    """Physical constants for one GPU type.

    All bandwidth and capacity values are raw hardware specs; the
    ``effective_mem_bw`` property applies the empirical 0.80 scaling factor
    observed across production deployments.

    Attributes
    ----------
    name            : human-readable label (e.g. "H100-SXM")
    mem_bw          : peak HBM bandwidth, bytes/s
    mem_capacity    : total HBM capacity, bytes
    fp16_tc_flops   : FP16 tensor-core peak, FLOPS
    fp8_tc_flops    : FP8 tensor-core peak, FLOPS (0 if unsupported)
    nvlink_bw       : intra-node NVLink bandwidth per GPU, bytes/s
    inter_node_bw   : inter-node bandwidth per GPU, bytes/s
    pcie_bw         : PCIe bandwidth per GPU, bytes/s
    sm_version      : CUDA SM version (e.g. 90 for H100)
    power           : TDP, Watts
    cost_per_hr     : cloud on-demand $/GPU-hr
    mem_bw_scale    : empirical BW utilisation factor (default 0.80)
    mem_const_lat   : per-layer constant overhead, seconds (default 3µs)
    p2p_latency     : GPU-to-GPU P2P base latency, seconds (default 10µs)
    nccl_mem        : NCCL workspace bytes by TP size
    other_mem       : misc. GPU memory overhead, bytes
    """
    name: str
    mem_bw: float
    mem_capacity: int
    fp16_tc_flops: float
    fp8_tc_flops: float
    nvlink_bw: float
    inter_node_bw: float
    pcie_bw: float
    sm_version: int
    power: float
    cost_per_hr: float
    mem_bw_scale: float = MEM_BW_SCALE
    mem_const_lat: float = MEM_CONST_LAT
    p2p_latency: float = P2P_LATENCY
    nccl_mem: Dict[int, int] = field(default_factory=lambda: dict(NCCL_MEM))
    other_mem: int = OTHER_MEM

    @property
    def effective_mem_bw(self) -> float:
        """Sustained memory bandwidth after empirical scaling."""
        return self.mem_bw * self.mem_bw_scale

    def free_vram(self, model_bytes_per_gpu: int, tp: int = 1) -> int:
        """Available HBM for KV-cache after model weights and overheads."""
        nccl = self.nccl_mem.get(tp, self.nccl_mem.get(8, 0))
        return max(0, self.mem_capacity - model_bytes_per_gpu - nccl - self.other_mem)
