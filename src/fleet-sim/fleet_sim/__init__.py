"""vllm-sr-sim: Fleet-level LLM inference simulator.

Provides:
  - Fleet/Pool/Instance DES engine
  - Pluggable routing algorithms (LengthRouter, C&R, LeastLoaded, Random)
  - Workload generators (Poisson+CDF, real trace replay)
  - Fleet optimizer (aggregated M/G/c + DES sizing)
  - Disaggregated fleet optimizer (prefill/decode pool rate-matching)
  - Model-aware GPU profiles (ComputedProfile from HardwareSpec + ModelSpec)
  - Hardware catalog: A100, H100, H200, B200, GB200, GB300, L40S, B60
  - Model catalog: Llama-3.1, Mistral/Mixtral, Qwen3, DeepSeek-V3
"""

# ── Core simulation engine ────────────────────────────────────────────────────
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from . import optimizer as _optimizer
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
    MISTRAL_7B,
    MIXTRAL_8X7B,
    QWEN3_8B,
    QWEN3_30B_A3B,
    QWEN3_32B,
    QWEN3_235B_A22B,
    ModelSpec,
    get_model,
    list_models,
)

# Keep root-level optimizer exports owned by fleet_sim.optimizer.
for _optimizer_export in _optimizer.ROOT_PUBLIC_EXPORTS:
    globals()[_optimizer_export] = getattr(_optimizer, _optimizer_export)

del _optimizer_export

try:
    __version__ = _pkg_version("vllm-sr-sim")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    # Core
    "Fleet",
    "FleetConfig",
    "PoolConfig",
    "Request",
    # Profiles
    "GpuProfile",
    "ManualProfile",
    "ComputedProfile",
    "DecodeEfficiencyPoint",
    "ProfileBuilder",
    "ServingConfig",
    "A100_80GB",
    "H100_80GB",
    "A10G",
    "CUSTOM",
    # Hardware
    "HardwareSpec",
    "A100_SXM",
    "H100_SXM",
    "H200_SXM",
    "B200_SXM",
    "GB200",
    "GB300",
    "L40S",
    "B60",
    "get_hardware",
    "list_hardware",
    # Models
    "ModelSpec",
    "LLAMA_3_1_8B",
    "LLAMA_3_1_70B",
    "LLAMA_3_1_405B",
    "MISTRAL_7B",
    "MIXTRAL_8X7B",
    "QWEN3_8B",
    "QWEN3_32B",
    "QWEN3_235B_A22B",
    "QWEN3_30B_A3B",
    "DEEPSEEK_V3",
    "get_model",
    "list_models",
]
__all__.extend(_optimizer.ROOT_PUBLIC_EXPORTS)
