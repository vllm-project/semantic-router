"""Fleet optimizer — aggregated and disaggregated serving modes."""
from .base import (
    FleetOptimizer, SweepResult,
    ThresholdResult, threshold_pareto, print_threshold_pareto,
    node_availability,
    A100_AVAIL_RSC1_FAST, A100_AVAIL_RSC1_SLOW, H100_AVAIL_5PCT,
    GridFlexPoint, grid_flex_analysis, print_grid_flex_table,
    TpwPoint, FleetTpwResult,
    tpw_analysis, fleet_tpw_analysis,
    print_tpw_table, print_fleet_tpw,
    _split_cdf,
)
from .disagg import (
    DisaggFleetOptimizer, DisaggResult, DisaggSweepPoint,
    ALPHA_PRE, ALPHA_DEC, BETA_TTFT,
)

__all__ = [
    # Aggregated (existing)
    "FleetOptimizer", "SweepResult",
    "ThresholdResult", "threshold_pareto", "print_threshold_pareto",
    "node_availability",
    "A100_AVAIL_RSC1_FAST", "A100_AVAIL_RSC1_SLOW", "H100_AVAIL_5PCT",
    # Grid flexibility
    "GridFlexPoint", "grid_flex_analysis", "print_grid_flex_table",
    # Tokens-per-watt
    "TpwPoint", "FleetTpwResult",
    "tpw_analysis", "fleet_tpw_analysis",
    "print_tpw_table", "print_fleet_tpw",
    "_split_cdf",
    # Disaggregated (new)
    "DisaggFleetOptimizer", "DisaggResult", "DisaggSweepPoint",
    "ALPHA_PRE", "ALPHA_DEC", "BETA_TTFT",
]
