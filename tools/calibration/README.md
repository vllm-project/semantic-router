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

Run the Python unit suites from the repository root:

```bash
PYTHONPATH=tools/calibration \
python -m pytest tools/calibration/recipe tools/calibration/tuning/tests
```

Inputs, generated reports, and config snapshots belong in the caller's chosen
artifact directory. Keep credentials and private endpoints out of committed
evidence.
