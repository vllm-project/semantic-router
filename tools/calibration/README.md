# Routing calibration

Calibration tools share this directory while retaining their separate execution
contracts:

- `recipe/`: maintained recipe probes, conformance checks, deployment validation,
  and evidence reports;
- `tuning/`: trace analysis and bounded configuration changes, plus offline
  confidence analysis;
- `image-routing/`: image-routing classifier calibration.

The maintained recipe workflow uses
`recipe/router_calibration_loop.py` and the versioned probe schema in
`config/schemas/recipe-probes-v1.schema.json`. See the repository's
`routing-calibration` skill for live verification requirements.

Probe decisions and individual variants may declare `expected_signal_errors`,
an exact mapping from runtime signal keys to error codes. For example, a probe
that deliberately exceeds complete-input evaluation limits can require:

```yaml
expected_signal_errors:
  reask:repeat: reask_evaluation_failed
  projection:recovery: projection_input_failed
  projection:retry: projection_input_failed
```

Omitting the field requires an empty error map. A variant inherits its decision's
map unless it supplies a replacement; an explicit `{}` restores the no-error
requirement. Missing errors, extra errors, and different codes all fail in both
policy and deployment evaluation. This field never waives routing, selection,
plugin, signal-match, or raw-value assertions: an errored signal cannot satisfy
an `expected_signal_values` bound. Reports retain both expected and observed
maps, including diagnostics for failed HTTP requests, which always fail the probe.

Run the Python unit suites from the repository root:

```bash
PYTHONPATH=tools/calibration \
python -m pytest tools/calibration/recipe tools/calibration/tuning/tests
```

Inputs, generated reports, and config snapshots belong in the caller's chosen
artifact directory. Keep credentials and private endpoints out of committed
evidence.
