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
    "A100_AVAIL_RSC1_FAST",
    "A100_AVAIL_RSC1_SLOW",
    "ALPHA_DEC",
    "ALPHA_PRE",
    "BETA_TTFT",
    "H100_AVAIL_5PCT",
    # Disaggregated (new)
    "DisaggFleetOptimizer",
    "DisaggResult",
    "DisaggSweepPoint",
    # Aggregated (existing)
    "FleetOptimizer",
    "FleetTpwResult",
    # Grid flexibility
    "GridFlexPoint",
    "SweepResult",
    "ThresholdResult",
    # Tokens-per-watt
    "TpwPoint",
    "_split_cdf",
    "fleet_tpw_analysis",
    "grid_flex_analysis",
    "node_availability",
    "print_fleet_tpw",
    "print_grid_flex_table",
    "print_threshold_pareto",
    "print_tpw_table",
    "threshold_pareto",
    "tpw_analysis",
]
