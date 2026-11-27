"""recipe-lab: closed-loop experiments for semantic-router routing recipes.

Design doc: maintained outside the repository (umbrella + sub-issue structure).

Modules:
- contract: experiment spec with the oracle policy (YAML, v1alpha1)
- attribution: misroute attribution E1/E2/E3 over a labeled frozen dataset
- cli: recipe-lab attribute / plan / collect-replays
"""

__version__ = "0.1.0"

API_VERSION = "recipe-lab/v1alpha1"
