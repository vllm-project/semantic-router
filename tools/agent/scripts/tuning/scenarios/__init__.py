"""Built-in scenario plugins for the DSL tuning framework.

Each scenario defines what to tune and how to score it:
  - privacy:      privacy-aware routing policy repair
  - calibration:  domain-conditional model escalation tuning
  - confidence:   per-category confidence threshold optimization (offline)
"""

from .calibration import CalibrationScenario
from .privacy import PrivacyScenario

__all__ = ["CalibrationScenario", "PrivacyScenario"]
