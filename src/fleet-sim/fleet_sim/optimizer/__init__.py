"""Fleet optimizer — aggregated and disaggregated serving modes."""

from .base import (
    A100_AVAIL_RSC1_FAST,
    A100_AVAIL_RSC1_SLOW,
    H100_AVAIL_5PCT,
    FleetOptimizer,
    FleetTpwResult,
    GridFlexPoint,
    SweepResult,
    ThresholdResult,
    TpwPoint,
    _split_cdf,
    fleet_tpw_analysis,
    grid_flex_analysis,
    node_availability,
    print_fleet_tpw,
    print_grid_flex_table,
    print_threshold_pareto,
    print_tpw_table,
    threshold_pareto,
    tpw_analysis,
)
from .disagg import (
    ALPHA_DEC,
    ALPHA_PRE,
    BETA_TTFT,
    DisaggFleetOptimizer,
    DisaggResult,
    DisaggSweepPoint,
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
