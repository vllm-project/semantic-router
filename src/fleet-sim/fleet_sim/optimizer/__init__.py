"""Fleet optimizer — aggregated and disaggregated serving modes."""

from .grid_flex import (
    GridFlexPoint,
    grid_flex_analysis,
    print_grid_flex_table,
)

from .tpw import (
    FleetTpwResult,
    TpwPoint,
    _split_cdf,
    fleet_tpw_analysis,
    print_fleet_tpw,
    print_tpw_table,
    tpw_analysis,
)

from .base import (
    A100_AVAIL_RSC1_FAST,
    A100_AVAIL_RSC1_SLOW,
    H100_AVAIL_5PCT,
    FleetOptimizer,
    SweepResult,
    ThresholdResult,
    node_availability,
    print_threshold_pareto,
    threshold_pareto,
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
