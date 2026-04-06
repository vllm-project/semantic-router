"""Fleet optimizer — aggregated and disaggregated serving modes."""

from .base import (
    A100_AVAIL_RSC1_FAST,
    A100_AVAIL_RSC1_SLOW,
    H100_AVAIL_5PCT,
    FleetOptimizer,
    SweepResult,
    node_availability,
)
from .disagg import (
    ALPHA_DEC,
    ALPHA_PRE,
    BETA_TTFT,
    DisaggFleetOptimizer,
    DisaggResult,
    DisaggSweepPoint,
)
from .grid_flex import GridFlexPoint, grid_flex_analysis, print_grid_flex_table
from .pareto import ThresholdResult, print_threshold_pareto, threshold_pareto
from .tpw import (
    FleetTpwResult,
    TpwPoint,
    _split_cdf,
    fleet_tpw_analysis,
    print_fleet_tpw,
    print_tpw_table,
    tpw_analysis,
)

ROOT_PUBLIC_EXPORTS = (
    "FleetOptimizer",
    "SweepResult",
    "DisaggFleetOptimizer",
    "DisaggResult",
    "DisaggSweepPoint",
    "ALPHA_PRE",
    "ALPHA_DEC",
    "BETA_TTFT",
    "GridFlexPoint",
    "grid_flex_analysis",
    "print_grid_flex_table",
    "TpwPoint",
    "FleetTpwResult",
    "tpw_analysis",
    "fleet_tpw_analysis",
    "print_tpw_table",
    "print_fleet_tpw",
)

__all__ = [
    # Aggregated (existing)
    "FleetOptimizer",
    "SweepResult",
    "ThresholdResult",
    "threshold_pareto",
    "print_threshold_pareto",
    "node_availability",
    "A100_AVAIL_RSC1_FAST",
    "A100_AVAIL_RSC1_SLOW",
    "H100_AVAIL_5PCT",
    # Grid flexibility
    "GridFlexPoint",
    "grid_flex_analysis",
    "print_grid_flex_table",
    # Tokens-per-watt
    "TpwPoint",
    "FleetTpwResult",
    "tpw_analysis",
    "fleet_tpw_analysis",
    "print_tpw_table",
    "print_fleet_tpw",
    "_split_cdf",
    # Disaggregated (new)
    "DisaggFleetOptimizer",
    "DisaggResult",
    "DisaggSweepPoint",
    "ALPHA_PRE",
    "ALPHA_DEC",
    "BETA_TTFT",
]
