"""DSL Tuning Framework — extensible analytical optimization for semantic router configs.

Core modules:
  client     — read-only RouterClient for immutable-runtime evaluation
  probes     — probe loading and result persistence
  engine     — trace-based analytical engine (diagnosis, decomposition, fixes, regression)
  analyzer   — offline threshold optimization on collected data
  scenario   — Scenario ABC + CandidateTuner for offline candidate generation

Scenario plugins live in tuning.scenarios.
"""

from .analyzer import OfflineAnalyzer
from .client import RouterClient
from .probes import load_probes, save_results
from .scenario import CandidateTuner, Scenario

__all__ = [
    "OfflineAnalyzer",
    "RouterClient",
    "CandidateTuner",
    "Scenario",
    "load_probes",
    "save_results",
]
