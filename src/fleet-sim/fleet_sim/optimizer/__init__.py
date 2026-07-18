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
from .forecast import evaluate_forecast_backtest
from .forecast_models import (
    ForecastActuationRecord,
    ForecastBacktestError,
    ForecastBacktestReport,
    ForecastCapacityRecommendation,
    ForecastMethodBacktest,
    ForecastWindowBacktest,
)
from .grid_flex import (
    GridFlexPoint,
    grid_flex_analysis,
    print_grid_flex_table,
)
from .mixture import (
    MixtureCaseResult,
    MixtureOptimizationError,
    MixtureOptimizationReport,
    MixtureStressCase,
    RobustMixtureRecommendation,
    aggregate_mixture_cdf,
    evaluate_mixture_scenario,
)
from .threshold import ThresholdResult, print_threshold_pareto, threshold_pareto
from .tpw import (
    FleetTpwResult,
    TpwPoint,
    _split_cdf,
    fleet_tpw_analysis,
    print_fleet_tpw,
    print_tpw_table,
    tpw_analysis,
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
    # Workload mixtures
    "MixtureCaseResult",
    "MixtureOptimizationError",
    "MixtureOptimizationReport",
    "MixtureStressCase",
    "RobustMixtureRecommendation",
    "aggregate_mixture_cdf",
    "evaluate_mixture_scenario",
    # Workload forecast backtests
    "ForecastActuationRecord",
    "ForecastBacktestError",
    "ForecastBacktestReport",
    "ForecastCapacityRecommendation",
    "ForecastMethodBacktest",
    "ForecastWindowBacktest",
    "evaluate_forecast_backtest",
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
