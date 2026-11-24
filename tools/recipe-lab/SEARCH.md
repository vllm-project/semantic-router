# Search and evaluation

Run these commands from `tools/recipe-lab` with the same workspace and installed configuration as the development loop. The benchmark adapter runs the stock agent, obtains independent strict grades and exports evidence. The Lab chooses assignments and verifies the receipt files. The [worked scenarios](SCENARIOS.md) introduce the development loop. Python, PyYAML and pytest are sufficient to run its synthetic regression tests offline.

## Freeze a development objective

Before opening development scores, save a `recipe-lab-objective-v1` JSON and its digest. Add `"objective": {"file": "objective.json", "sha256": "…"}` to the development matrix manifest passed to `headroom`. The declaration fixes the study, arm order and policy hashes, per-task controls, target and guard tasks, hashes of held-out task IDs, repeats, baseline, threshold values and any external checks. The report distinguishes thresholds measurable from the matrix and checks that require other evidence. Older matrices remain readable. The [objective fixture](tests/test_headroom.py) gives a complete example.

The cross-method search sections (skill-replay readbacks and the matched-budget three-method comparison) join this page with the optimizer-evaluation slice.
