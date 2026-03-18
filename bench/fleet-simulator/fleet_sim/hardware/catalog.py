"""Pre-built HardwareSpec instances for common NVIDIA GPU types.

Values sourced from official hardware specifications and empirically
validated correction factors. All eight GPU types span Ampere through
Blackwell generations.

Cloud pricing (cost_per_hr) reflects AWS on-demand rates per GPU:
  A100: p4d.24xlarge ($32.77/hr ÷ 8)
  H100: p5.48xlarge ($98.32/hr ÷ 8) [approximate]
  H200: not yet widely available, estimated
  B200/GB200/GB300: estimated pre-GA pricing
  L40S: g6e.48xlarge (~$16/hr ÷ 8) [approximate]
  B60:  estimated
"""
from .spec import HardwareSpec, NCCL_MEM

# ── Ampere ────────────────────────────────────────────────────────────────────

A100_SXM = HardwareSpec(
    name="A100-SXM",
    mem_bw=2_039_000_000_000,        # 2.039 TB/s
    mem_capacity=85_899_345_920,      # 80 GiB
    fp16_tc_flops=312_000_000_000_000,  # 312 TFLOPS
    fp8_tc_flops=0,                   # not supported on Ampere
    nvlink_bw=300_000_000_000,        # 300 GB/s intra-node
    inter_node_bw=25_000_000_000,     # 25 GB/s (NDR 200Gb/s per node ÷ 8)
    pcie_bw=32_000_000_000,           # PCIe 4.0
    sm_version=80,
    power=400.0,
    cost_per_hr=2.21,
)

L40S = HardwareSpec(
    name="L40S",
    mem_bw=864_000_000_000,           # 864 GB/s
    mem_capacity=51_539_607_552,      # 48 GiB
    fp16_tc_flops=362_000_000_000_000,  # 362 TFLOPS
    fp8_tc_flops=733_000_000_000_000,   # 733 TFLOPS
    nvlink_bw=32_000_000_000,         # PCIe-only (no NVSwitch)
    inter_node_bw=25_000_000_000,
    pcie_bw=64_000_000_000,           # PCIe 5.0
    sm_version=89,
    power=350.0,
    cost_per_hr=1.80,
)

B60 = HardwareSpec(
    name="B60",
    mem_bw=456_000_000_000,           # 456 GB/s
    mem_capacity=25_769_803_776,      # 24 GiB
    fp16_tc_flops=98_000_000_000_000,   # 98 TFLOPS
    fp8_tc_flops=0,
    nvlink_bw=32_000_000_000,
    inter_node_bw=25_000_000_000,
    pcie_bw=64_000_000_000,
    sm_version=100,
    power=200.0,
    cost_per_hr=1.00,
)

# ── Hopper ────────────────────────────────────────────────────────────────────

H100_SXM = HardwareSpec(
    name="H100-SXM",
    mem_bw=3_350_000_000_000,         # 3.35 TB/s
    mem_capacity=85_899_345_920,      # 80 GiB
    fp16_tc_flops=989_000_000_000_000,  # 989 TFLOPS
    fp8_tc_flops=1_978_000_000_000_000, # 1978 TFLOPS
    nvlink_bw=450_000_000_000,        # 450 GB/s intra-node (NVSwitch)
    inter_node_bw=50_000_000_000,     # 50 GB/s (NDR 400Gb/s per node ÷ 8)
    pcie_bw=64_000_000_000,           # PCIe 5.0
    sm_version=90,
    power=700.0,
    cost_per_hr=4.02,
)

H200_SXM = HardwareSpec(
    name="H200-SXM",
    mem_bw=4_800_000_000_000,         # 4.8 TB/s (HBM3e)
    mem_capacity=151_397_597_184,     # 141 GiB
    fp16_tc_flops=989_000_000_000_000,
    fp8_tc_flops=1_978_000_000_000_000,
    nvlink_bw=450_000_000_000,
    inter_node_bw=50_000_000_000,
    pcie_bw=64_000_000_000,
    sm_version=90,
    power=700.0,
    cost_per_hr=6.00,
)

# ── Blackwell ─────────────────────────────────────────────────────────────────

B200_SXM = HardwareSpec(
    name="B200-SXM",
    mem_bw=8_000_000_000_000,         # 8 TB/s (HBM3e)
    mem_capacity=193_273_528_320,     # 180 GiB
    fp16_tc_flops=2_250_000_000_000_000,
    fp8_tc_flops=4_500_000_000_000_000,
    nvlink_bw=900_000_000_000,        # 900 GB/s (NVLink 5)
    inter_node_bw=100_000_000_000,
    pcie_bw=128_000_000_000,
    sm_version=100,
    power=1000.0,
    cost_per_hr=8.00,
)

GB200 = HardwareSpec(
    name="GB200",
    mem_bw=8_000_000_000_000,
    mem_capacity=214_748_364_800,     # 200 GiB (NVL72 per-GPU share)
    fp16_tc_flops=2_500_000_000_000_000,
    fp8_tc_flops=5_000_000_000_000_000,
    nvlink_bw=900_000_000_000,
    inter_node_bw=100_000_000_000,
    pcie_bw=128_000_000_000,
    sm_version=100,
    power=1200.0,
    cost_per_hr=10.00,
)

GB300 = HardwareSpec(
    name="GB300",
    mem_bw=8_000_000_000_000,
    mem_capacity=321_122_547_712,     # 299 GiB
    fp16_tc_flops=2_500_000_000_000_000,
    fp8_tc_flops=5_000_000_000_000_000,
    nvlink_bw=900_000_000_000,
    inter_node_bw=100_000_000_000,
    pcie_bw=128_000_000_000,
    sm_version=100,
    power=1400.0,
    cost_per_hr=12.00,
)

# ── Lookup by name ────────────────────────────────────────────────────────────

_CATALOG: dict = {
    "a100":     A100_SXM,
    "a100_sxm": A100_SXM,
    "h100":     H100_SXM,
    "h100_sxm": H100_SXM,
    "h200":     H200_SXM,
    "h200_sxm": H200_SXM,
    "b200":     B200_SXM,
    "b200_sxm": B200_SXM,
    "gb200":    GB200,
    "gb300":    GB300,
    "l40s":     L40S,
    "b60":      B60,
    "a100-80gb":  A100_SXM,
    "h100-80gb":  H100_SXM,
    "a10g":       L40S,
}


def get(name: str) -> HardwareSpec:
    """Look up a HardwareSpec by name (case-insensitive, hyphens/underscores ok)."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in _CATALOG:
        raise KeyError(
            f"Unknown hardware '{name}'. Available: {list_names()}"
        )
    return _CATALOG[key]


def list_names() -> list:
    """Return the canonical name for every hardware spec in the catalog."""
    seen, names = set(), []
    for spec in _CATALOG.values():
        if spec.name not in seen:
            seen.add(spec.name)
            names.append(spec.name)
    return sorted(names)
