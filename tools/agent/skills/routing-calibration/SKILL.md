---
name: routing-calibration
description: Use when calibrating or validating a maintained routing recipe against live model backends, including probe manifests and evidence reports.
---

# Routing calibration

Use `tools/dev/router-calibration/router_calibration_loop.py` and the versioned
probe schema in `config/schemas/recipe-probes-v1.schema.json`. Keep recipe
changes in the local checkout; remote systems are validation mirrors only.

Treat endpoint credentials, hostnames, and private fleet details as secrets.
Before a live run, validate the probe manifest and identify the exact recipe,
backend, and rollback point. Record the evaluated inputs, outputs, and observed
quality rather than inferring success from process health.

Use `make recipe-conformance-live-cpu RECIPE_CONFORMANCE_RECIPES=<recipe>`
for the recipe under calibration. Run `make recipe-conformance-live-cpu-all`
when the change warrants maintained-recipe regression coverage.
