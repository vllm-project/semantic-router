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
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .core import Fleet, FleetConfig, PoolConfig, Request

# ── GPU profiles ──────────────────────────────────────────────────────────────
from .gpu_profiles import (
    A10G,
    A100_80GB,
    CUSTOM,
    H100_80GB,
    ComputedProfile,
    DecodeEfficiencyPoint,
    GpuProfile,
    ManualProfile,
    ProfileBuilder,
    ServingConfig,
)

# ── Hardware catalog ──────────────────────────────────────────────────────────
from .hardware import (
    A100_SXM,
    B60,
    B200_SXM,
    GB200,
    GB300,
    H100_SXM,
    H200_SXM,
    L40S,
    HardwareSpec,
    get_hardware,
    list_hardware,
)

# ── Model catalog ─────────────────────────────────────────────────────────────
from .models import (
    DEEPSEEK_V3,
    LLAMA_3_1_8B,
    LLAMA_3_1_70B,
    LLAMA_3_1_405B,
    QWEN3_8B,
    QWEN3_30B_A3B,
    QWEN3_32B,
    QWEN3_235B_A22B,
    ModelSpec,
    get_model,
    list_models,
)

# ── Optimizers ────────────────────────────────────────────────────────────────
from .optimizer import (
    A100_AVAIL_RSC1_FAST,  # noqa: F401
    A100_AVAIL_RSC1_SLOW,  # noqa: F401
    ALPHA_DEC,
    ALPHA_PRE,
    BETA_TTFT,
    H100_AVAIL_5PCT,  # noqa: F401
    DisaggFleetOptimizer,
    DisaggResult,
    DisaggSweepPoint,
    FleetOptimizer,
    FleetTpwResult,
    GridFlexPoint,
    SweepResult,
    TpwPoint,
    _split_cdf,
    fleet_tpw_analysis,
    grid_flex_analysis,
    node_availability,  # noqa: F401
    print_fleet_tpw,
    print_grid_flex_table,
    print_tpw_table,
    tpw_analysis,
)

try:
    __version__ = _pkg_version("inference-fleet-sim")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "A10G",
    "A100_80GB",
    "A100_SXM",
    "ALPHA_DEC",
    "ALPHA_PRE",
    "B60",
    "B200_SXM",
    "BETA_TTFT",
    "CUSTOM",
    "DEEPSEEK_V3",
    "GB200",
    "GB300",
    "H100_80GB",
    "H100_SXM",
    "H200_SXM",
    "L40S",
    "LLAMA_3_1_8B",
    "LLAMA_3_1_70B",
    "LLAMA_3_1_405B",
    "QWEN3_8B",
    "QWEN3_30B_A3B",
    "QWEN3_32B",
    "QWEN3_235B_A22B",
    "ComputedProfile",
    "DecodeEfficiencyPoint",
    "DisaggFleetOptimizer",
    "DisaggResult",
    "DisaggSweepPoint",
    # Core
    "Fleet",
    "FleetConfig",
    # Optimizers
    "FleetOptimizer",
    "FleetTpwResult",
    # Profiles
    "GpuProfile",
    # Grid flexibility
    "GridFlexPoint",
    # Hardware
    "HardwareSpec",
    "ManualProfile",
    # Models
    "ModelSpec",
    "PoolConfig",
    "ProfileBuilder",
    "Request",
    "ServingConfig",
    "SweepResult",
    # Tokens-per-watt
    "TpwPoint",
    "_split_cdf",
    "fleet_tpw_analysis",
    "get_hardware",
    "get_model",
    "grid_flex_analysis",
    "list_hardware",
    "list_models",
    "print_fleet_tpw",
    "print_grid_flex_table",
    "print_tpw_table",
    "tpw_analysis",
]
