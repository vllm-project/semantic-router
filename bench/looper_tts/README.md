# Looper fixed-budget experiment contracts

This package implements the offline foundation of [issue #2858](https://github.com/vllm-project/semantic-router/issues/2858).
It validates experiment inputs and freezes a reproducible matrix. It does not
call providers, download datasets, execute algorithms, compute prices, or grade
answers. Its fixtures are synthetic contract examples, never benchmark evidence.

## Run from the repository root

Only Python 3.8+ and the standard library are required:

```bash
python -m bench.looper_tts validate --config bench/looper_tts/testdata/synthetic.json
python -m bench.looper_tts plan --config bench/looper_tts/testdata/synthetic.json --code-revision <evaluated-revision> --output /tmp/looper-tts-plan
python -m unittest bench.looper_tts.test_contracts -v
```

Replace `<evaluated-revision>` with the exact evaluated code revision; the
planner records this supplied value without claiming to verify a clean checkout.
The output directory receives `manifest.json`; an existing manifest is never
overwritten. No credentials or provider endpoints belong in these artifacts.
After installing the `bench` package, use `looper-tts` or
`python -m looper_tts` with the same arguments.

## Version 1 input contract

`testdata/synthetic.json` is the complete minimal example. Objects use exact
field sets: missing or unknown fields, duplicate JSON keys, unknown versions,
nonfinite numbers, and booleans used as numbers are rejected. Public validation
functions raise `ContractError`; CLI validation errors exit with status 2.

| Object | Contract |
| --- | --- |
| Experiment | `schema_version`, unique local `id`, dataset, models, arms, budgets, scorer, nonempty unique integer seeds |
| Dataset | ID and revision, `synthetic` or `benchmark` evidence kind, evaluation items, disjoint calibration IDs |
| Item | Unique ID, exact rendered prompt, lowercase SHA-256 of its UTF-8 bytes |
| Model | Local ID, provider/model/revision, temperature and top-p, dated USD input/output prices per million tokens |
| Arm | Unique ID, algorithm, ordered model IDs, algorithm-specific parameters |
| Budget | Unique ID, positive call and total-token ceilings, `stop_and_record` exhaustion policy |
| Scorer | ID, revision, SHA-256 of the external scoring protocol |

The matrix requires Direct, Confidence, ReMoM and Fusion, plus at least two
distinct budget ceilings. Every arm crosses every budget and seed. Direct has
one model; Confidence records threshold and calibration revision; ReMoM records
positive round breadths; Fusion names panel, judge and synthesis models. Fusion
stage references must exactly cover its declared model IDs. Model IDs cannot
reference undeclared models. Repeated sampling and voting will extend the arm
contract alongside their future executors.

Prompts are embedded so a saved manifest contains the exact evaluated inputs.
Dataset and model revisions and scorer protocol digests are declared provenance;
this offline tool does not resolve or authenticate external versions. Calibration
overlap is checked by supplied IDs within this dataset, not semantic similarity.
No secret or restricted dataset material should be placed in committed examples.

Ceilings describe planned constraints, not proof of runtime enforcement or equal
actual spend. The executor phase must reserve concurrent budgets and charge all
generation, verification, judge, synthesis and retry calls. Configuration
validation deliberately does not assert that any real algorithm fits a budget.

## Frozen identity

`manifest.json` contains the complete config, config digest, evaluated revision,
planner source digest, experiment ID, reproduction command argument vector and
expanded matrix. Each cell identifies an arm, budget and seed and lists its
evaluation items. Status is always `planned`.

Hashes use SHA-256 of UTF-8 canonical JSON (sorted object keys, compact separators,
Unicode preserved, no NaN). Prompt hashes use raw UTF-8 bytes instead. Object key
order does not matter; array order and numeric representation do matter.
Experiment identity includes the config, supplied code revision and hashes of
the package's top-level Python source, including its contract tests. Commands
and output paths are recorded but excluded from experiment identity. There is
no timestamp, so identical inputs and source generate identical plans.

## Evidence contract for future executors

`validate_records(bundle, plan)` validates a complete normalized JSON bundle:

- `schema_version`, `experiment_id`, `evidence_kind`, `calls`, `results`;
- calls have unique IDs, experiment/cell/item coordinates, stage, model ID,
  one-based attempt number, status, usage, latency, raw-output reference, error
  and cache identity;
- results have unique IDs and coordinates, terminal status, final answer,
  normalized score, scorer ID, call references, candidate scores, optional panel
  digest, budget status and error.

Call stages are `generate`, `verify`, `select`, `judge`, `synthesize`; statuses
are `success`, `error`, `cached`. Result statuses are `success`, `error`,
`budget_exhausted`. A successful result may remain ungraded (`score: null`).
Scores are normalized to [0, 1]; benchmark-native outputs remain in raw artifacts.
Failed results retain null scores; future reports must declare failure treatment
and retain the full planned cohort rather than silently dropping those rows.

There must be exactly one terminal result per planned cell/item and every call
must belong to exactly one matching result. Missing, orphaned or cross-cell calls
are rejected. A failed request can have no output, and a budget-exhausted result
can have no calls. Retries are separate call records with their attempt number.

Unknown usage, prices, latency and ungraded scores use JSON `null`, never zero.
Provider total tokens are preserved, not silently recomputed from input/output.
Cached calls retain original candidate usage and a cache artifact SHA-256 but
have null live latency. Cache identity is distinct from the optional whole-panel
digest. Runtime writers must save raw artifacts; this structural validator does
not dereference or verify their existence. The fixture uses `fixture://` references.

Later accounting must separately report actual replay expenditure and complete
algorithm cost, including original candidate generation; cached usage must not
be counted as newly purchased tokens. Live latency and cached-policy latency are
also separate measurements. Price calculations and budget enforcement are outside
this phase.

## Deterministic fixture

`fixture_records(plan)` accepts only a synthetic plan and creates examples for
all cells: success, provider error, exhausted budget, unknown usage, zero output
tokens and cached candidates. These are hand-authored states, not a simulated
implementation of each algorithm. Unit tests round-trip and validate the bundle,
check bad references and states, and exercise both CLI commands without network
access. The tests are also discoverable by pytest.
