"""inference-fleet-sim: Fleet-level LLM inference simulator.

Provides:
  - Fleet/Pool/Instance DES engine
  - Pluggable routing algorithms (LengthRouter, C&R, LeastLoaded, Random)
  - Workload generators (Poisson+CDF, real trace replay)
  - Fleet optimizer (aggregated M/G/c + DES sizing)
  - Disaggregated fleet optimizer (prefill/decode pool rate-matching)
  - Model-aware GPU profiles (ComputedProfile from HardwareSpec + ModelSpec)
  - Hardware catalog: A100, H100, H200, B200, GB200, GB300, L40S, B60
  - Model catalog: Llama-3.1, Qwen3, DeepSeek-V3
"""
# ── Core simulation engine ────────────────────────────────────────────────────
from .core import Fleet, FleetConfig, PoolConfig, Request

# ── GPU profiles ──────────────────────────────────────────────────────────────
from .gpu_profiles import (
    GpuProfile,
    ManualProfile, ComputedProfile, DecodeEfficiencyPoint,
    ProfileBuilder, ServingConfig,
    A100_80GB, H100_80GB, A10G, CUSTOM,
)

# ── Hardware catalog ──────────────────────────────────────────────────────────
from .hardware import (
    HardwareSpec,
    A100_SXM, H100_SXM, H200_SXM, B200_SXM, GB200, GB300, L40S, B60,
    get_hardware, list_hardware,
)

# ── Model catalog ─────────────────────────────────────────────────────────────
from .models import (
    ModelSpec,
    LLAMA_3_1_8B, LLAMA_3_1_70B, LLAMA_3_1_405B,
    QWEN3_8B, QWEN3_32B, QWEN3_235B_A22B, QWEN3_30B_A3B,
    DEEPSEEK_V3,
    get_model, list_models,
)

# ── Optimizers ────────────────────────────────────────────────────────────────
from .optimizer import (
    FleetOptimizer, SweepResult,
    DisaggFleetOptimizer, DisaggResult, DisaggSweepPoint,
    ALPHA_PRE, ALPHA_DEC, BETA_TTFT,
    node_availability,
    A100_AVAIL_RSC1_FAST, A100_AVAIL_RSC1_SLOW, H100_AVAIL_5PCT,
    GridFlexPoint, grid_flex_analysis, print_grid_flex_table,
    TpwPoint, FleetTpwResult,
    tpw_analysis, fleet_tpw_analysis,
    print_tpw_table, print_fleet_tpw,
    _split_cdf,
)

from importlib.metadata import version as _pkg_version, PackageNotFoundError
try:
    __version__ = _pkg_version("inference-fleet-sim")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    # Core
    "Fleet", "FleetConfig", "PoolConfig", "Request",
    # Profiles
    "GpuProfile", "ManualProfile", "ComputedProfile", "DecodeEfficiencyPoint",
    "ProfileBuilder", "ServingConfig",
    "A100_80GB", "H100_80GB", "A10G", "CUSTOM",
    # Hardware
    "HardwareSpec",
    "A100_SXM", "H100_SXM", "H200_SXM", "B200_SXM", "GB200", "GB300",
    "L40S", "B60", "get_hardware", "list_hardware",
    # Models
    "ModelSpec",
    "LLAMA_3_1_8B", "LLAMA_3_1_70B", "LLAMA_3_1_405B",
    "QWEN3_8B", "QWEN3_32B", "QWEN3_235B_A22B", "QWEN3_30B_A3B",
    "DEEPSEEK_V3", "get_model", "list_models",
    # Optimizers
    "FleetOptimizer", "SweepResult",
    "DisaggFleetOptimizer", "DisaggResult", "DisaggSweepPoint",
    "ALPHA_PRE", "ALPHA_DEC", "BETA_TTFT",
    # Grid flexibility
    "GridFlexPoint", "grid_flex_analysis", "print_grid_flex_table",
    # Tokens-per-watt
    "TpwPoint", "FleetTpwResult",
    "tpw_analysis", "fleet_tpw_analysis",
    "print_tpw_table", "print_fleet_tpw",
    "_split_cdf",
]
