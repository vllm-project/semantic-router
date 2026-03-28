# DSL Tuning Framework

Analytical optimization toolkit for semantic router DSL configurations.
Runs an **observe → diagnose → fix → verify** loop that traces routing
failures back to specific DSL parameters and computes minimal, regression-checked
fixes — no gradient descent, no black-box search.

## Quick start

```bash
# From the repository root:
PYTHONPATH=tools/agent/scripts:$PYTHONPATH \
python3 -m tuning.cli <scenario> \
  --endpoint http://localhost:8080 \
  --config path/to/config.yaml \
  --probes path/to/probes.yaml \
  --router-pid <PID> \
  --max-iter 10
```

## Router API dependencies

The framework interacts with these semantic router HTTP endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/eval?trace=true` | POST | Evaluate a query and return the full decision trace |
| `/config/hash` | GET | SHA256 of the active config file — used to confirm hot-reloads |
| `/config/router` | GET | Read the current router config |
| `/config/router` | PUT | Replace the config (creates a versioned backup) |
| `/config/router/rollback` | POST | Rollback to a named backup version |
| `/config/router/versions` | GET | List available backup versions |

Config changes written to the YAML file are picked up automatically via
fsnotify (file-system hot-reload). The `RouterClient.hot_reload()` helper
sends `SIGHUP` and polls `/config/hash` to confirm the new config is live.

## Package layout

```
tuning/
├── cli.py               # Entry point — resolves scenario by name, parses args
├── client.py            # RouterClient — eval, config hash, hot-reload HTTP calls
├── engine.py            # Analytical core — data structures, trace walking, score
│                        #   decomposition, fix computation, regression checking
├── engine_selection.py  # Fix selection, priority conflict analysis, config mutation
│                        #   (split from engine.py for structural line-count limits)
├── analyzer.py          # OfflineAnalyzer — threshold sweeps on pre-collected data
├── probes.py            # YAML probe loader and result persistence
├── scenario.py          # Scenario ABC + TuningLoop orchestrator
├── scenarios/
│   ├── privacy.py       # Privacy routing repair (parametric + structural)
│   ├── calibration.py   # Category escalation calibration (batch structural)
│   └── confidence.py    # Per-category confidence thresholds (offline)
├── tests/
│   └── test_framework.py  # Unit tests (45 tests, no live router needed)
└── verify_results/
    ├── run_confidence_verification.py  # Reproduce Case Study 3 offline
    ├── confidence_verification.json    # Expected output for Case Study 3
    ├── privacy_verification.json       # Expected output for Case Study 1
    └── calibration_verification.json   # Expected output for Case Study 2
```

`engine.py` and `engine_selection.py` form the analytical core. All public
symbols are re-exported from `engine.py` for backward compatibility, so
existing code using `from tuning.engine import select_fix` continues to work.

## Scenario plugins

Three scenario plugins correspond to the paper's three case studies:

### Case Study 1: Privacy Routing Repair (`privacy`)

Live tuning against the router. Diagnoses parametric failures (threshold
mismatch) and structural failures (missing signal coverage), then applies
severity-weighted threshold fixes with regression checking.

- 16 probes across 4 routing lanes (including Chinese cross-lingual probes)
- Severity-weighted loss decreases 31% in one iteration
- Residual structural failures correctly identified as outside the parameter
  control surface

### Case Study 2: Confidence Calibration (`calibration`)

Live tuning with batch structural fixes. Starts from an "always escalate"
configuration and prunes non-beneficial categories from the escalation
decision using regression-aware batch analysis.

- 140 probes (10 per category × 14 MMLU-Pro categories)
- One batch fix removes 5 categories, reducing severity-weighted loss by 45%
- Correctly refuses to remove economics/history due to cross-category
  classifier confusion that would regress higher-severity probes

### Case Study 3: Per-Category Confidence Thresholds (`confidence`)

Offline analysis — no live router needed. Uses `OfflineAnalyzer` to classify
14 MMLU-Pro categories into ESCALATE/SELECTIVE/AVOID strategies and compute
per-category optimal confidence thresholds.

Run it:

```bash
PYTHONPATH=tools/agent/scripts:$PYTHONPATH \
python3 tools/agent/scripts/tuning/verify_results/run_confidence_verification.py
```

Expected output (reproduced from 350 MMLU-Pro questions):

```
Strategy distribution: ESCALATE=5, SELECTIVE=7, AVOID=2
Overall accuracy: 61.7%  (7B baseline: 46.6%, 72B baseline: 59.7%)
Overall escalation: 60.3%
Router queries: 0
```

The tuned policy achieves 61.7% accuracy — 2.0 pp above always-72B — at
61% escalation cost, compared to 100% for always-72B. All optimization is
analytical; no router queries are needed beyond the initial data collection.

## Adding a new scenario

1. Create `tuning/scenarios/my_scenario.py`:

```python
from tuning.scenario import Scenario
from tuning.engine import probe_severity  # or write your own

class MyScenario(Scenario):

    @property
    def name(self) -> str:
        return "my_scenario"

    def severity(self, probe: dict) -> int:
        """Return an integer weight for this probe's failure cost."""
        return probe_severity(probe)  # default: uses probe["severity"]
```

2. Register it in `tuning/cli.py`'s `BUILTIN_SCENARIOS` dict:

```python
BUILTIN_SCENARIOS = {
    "privacy": "tuning.scenarios.privacy:PrivacyScenario",
    "calibration": "tuning.scenarios.calibration:CalibrationScenario",
    "my_scenario": "tuning.scenarios.my_scenario:MyScenario",
}
```

Alternatively, skip registration and pass the full module path directly:

```bash
python3 -m tuning.cli tuning.scenarios.my_scenario:MyScenario --config ...
```

3. Write a probes file (`my_scenario.probes.yaml`):

```yaml
probes:
  - id: example_probe
    query: "What is the capital of France?"
    expected_decision: standard_route
    tags: [baseline]
```

### Optional overrides

| Method | Purpose |
|---|---|
| `adapt_result(probe, resp)` | Custom parsing of router response → result dict |
| `display_iteration(...)` | Extra per-iteration console output |
| `build_output(base)` | Enrich the final JSON with scenario-specific fields |

For offline scenarios (no live router), use `OfflineAnalyzer` directly —
see `scenarios/confidence.py` for an example.

## Running tests

```bash
PYTHONPATH=tools/agent/scripts:$PYTHONPATH \
python3 -m pytest tools/agent/scripts/tuning/tests/test_framework.py -v
```

All 45 tests run without a live router and validate the core analytical engine
(trace walking, score decomposition, fix computation, regression checking,
fix selection, config mutation) and the scenario/CLI integration contracts.
