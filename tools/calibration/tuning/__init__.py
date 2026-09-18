"""DSL Tuning Framework — extensible analytical optimization for semantic router configs.

Core modules:
  client     — RouterClient for eval/config/reload HTTP interactions
  probes     — probe loading and result persistence
  engine     — trace-based analytical engine (diagnosis, decomposition, fixes, regression)
  analyzer   — offline threshold optimization on collected data
  scenario   — Scenario ABC + TuningLoop for pluggable tuning pipelines

Scenario plugins live in tuning.scenarios.
"""

from .analyzer import OfflineAnalyzer
from .client import RouterClient
from .probes import load_probes, save_results
from .scenario import Scenario, TuningLoop

__all__ = [
    "OfflineAnalyzer",
    "RouterClient",
    "Scenario",
    "TuningLoop",
    "load_probes",
    "save_results",
]
