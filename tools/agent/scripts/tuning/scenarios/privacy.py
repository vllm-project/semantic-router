"""Privacy Routing Scenario — severity-weighted repair of routing policy.

Targets a four-lane policy (security, privacy, frontier reasoning, standard)
where misrouting a query to the wrong lane has differential severity impact.

Usage:
    from tuning import CandidateTuner, RouterClient
    from tuning.scenarios import PrivacyScenario

    scenario = PrivacyScenario()
    tuner = CandidateTuner(
        scenario=scenario,
        router=RouterClient(),
        config_path=Path("config.yaml"),
        probes_path=Path("probes.yaml"),
        candidate_path=Path("candidate.yaml"),
    )
    output = tuner.run()
"""

from __future__ import annotations

from ..engine import probe_severity
from ..scenario import Scenario


class PrivacyScenario(Scenario):

    @property
    def name(self) -> str:
        return "privacy_routing_tuning"

    def severity(self, probe: dict) -> int:
        return probe_severity(probe)
