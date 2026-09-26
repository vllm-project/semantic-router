# Looper fixed-budget experiment contracts

This package implements the fixed-budget benchmark harness for [issue #2858](https://github.com/vllm-project/semantic-router/issues/2858).
It validates experiment inputs, freezes a reproducible matrix, executes the
four maintained algorithm families through an OpenAI-compatible provider, and
writes normalized per-call evidence. Candidate replay, repeated sampling,
native scoring, and reports remain later phases. Its deterministic provider is
synthetic smoke evidence, never a benchmark claim.

Fake execution rejects manifests whose dataset evidence kind is not
`synthetic`, before dispatch or artifact creation. Blank content and reasoning
produce terminal error results while retaining paid call records and receipts.

## Run from the repository root

Python 3.8+ and the benchmark package dependencies are required:

```bash
python -m bench.looper_tts validate --config bench/looper_tts/testdata/synthetic.json
python -m bench.looper_tts plan --config bench/looper_tts/testdata/synthetic.json --code-revision <evaluated-revision> --output /tmp/looper-tts-plan
python -m unittest bench.looper_tts.test_contracts bench.looper_tts.test_evidence_integrity -v

# Offline PR2 smoke (writes records.json, runtime_receipt.json and raw/)
python -m bench.looper_tts execute --manifest /tmp/looper-tts-plan/manifest.json --output /tmp/looper-tts-run --fake

# Live execution (one-attempt provider calls; retries are recorded separately)
python -m bench.looper_tts execute --manifest /tmp/looper-tts-plan/manifest.json --output /tmp/looper-tts-run --endpoint http://localhost:8000/v1 --api-key "$MODEL_API_KEY" --retries 1
```

Builds that include the native Looper bindings can run the same manifest
through the production Go algorithms. The native endpoint is the complete
`/chat/completions` URL; the command reads the key from an environment
variable and writes normalized `records.json` plus a runtime receipt with budget
events:

Every native dispatch applies the target model's declared temperature and
top-p, including verification and synthesis calls. Native arms that assign
conflicting sampling settings to the same provider model slug are rejected,
because the production client cannot distinguish those aliases at dispatch.

```bash
cd src/semantic-router && go build -o ../../bin/looper-tts ./cmd/looper-tts
../../bin/looper-tts --manifest /tmp/looper-tts-plan/manifest.json --output /tmp/looper-tts-native --endpoint http://localhost:8000/v1/chat/completions
```

Replace `<evaluated-revision>` with the exact evaluated code revision; the
planner records this supplied value without claiming to verify a clean checkout.
The plan output directory receives `manifest.json`; an existing manifest is
never overwritten. Execution writes `records.json`, `runtime_receipt.json`, and
raw response artifacts. No credentials are stored in those artifacts.
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

Ceilings describe planned constraints and are enforced independently for every
cell/item execution. The executor reserves the prompt estimate plus its
deterministic output cap before dispatch, then settles against provider usage.
Concurrent Fusion and ReMoM calls reserve before entering their worker pool.
Generation, verification, judge, synthesis, failed, unusable, and retry calls
all consume a call slot. If usage is missing or lacks a complete billable
token pair, the reservation is charged and the receipt marks the call
`usage_source: reservation`.

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

Evidence validation first revalidates the saved config, config digest, experiment
identity and complete matrix, including cell IDs and item lists. The saved planner
digest participates in identity; it is not replaced by the current checkout's
digest. These checks establish internal consistency, not authenticity: a trusted
external experiment ID is still needed to detect replacement of an entire plan
and its evidence with a newly hashed experiment.

## Evidence contract for future executors

`validate_records(bundle, plan)` validates a complete normalized JSON bundle:

- `schema_version`, `experiment_id`, `evidence_kind`, `calls`, `results`;
- calls have unique IDs, experiment/cell/item coordinates, stage, model ID,
  one-based attempt number, status, usage, latency, raw-output reference, error
  and cache identity;
- results have unique IDs and coordinates, terminal status, final answer,
  normalized score, scorer ID, call references, candidate scores, optional panel
  digest, budget status and error.

Allowed stages and models are bound to the declared algorithm:

| Algorithm | Stage and model roles |
| --- | --- |
| Direct | `generate` using its single model |
| Confidence | `generate` or `verify` using its declared model pool |
| ReMoM | `generate` or `synthesize` using its declared model pool |
| Fusion | `generate` using panel models, `judge` using the judge model, `synthesize` using the synthesis model |

`select` has no declared model role in this version and is rejected. Call statuses
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
When all three usage fields are known, the total must be at least the sum of
prompt and completion tokens; additional provider token classes remain valid.
Cached calls retain original candidate usage and a cache artifact SHA-256 but
have null live latency. Cache identity is distinct from the optional whole-panel
digest. Runtime writers must save raw artifacts; this structural validator does
not dereference or verify their existence. The fixture uses `fixture://` references.

Later accounting must separately report actual replay expenditure and complete
algorithm cost, including original candidate generation; cached usage must not
be counted as newly purchased tokens. Live latency and cached-policy latency are
also separate measurements. PR2's `runtime_receipt.json` reports complete
algorithm cost when both billable usage fields and model prices are known; the
normalized `records.json` keeps the stable v1 evidence fields.

## Runtime behavior

`execute` expands each matrix cell/item as follows:

| Algorithm | Calls |
| --- | --- |
| Direct | one `generate` call |
| Confidence | `generate`, then `verify` for each candidate until the threshold is met |
| ReMoM | parallel `generate` rounds from `breadth`, followed by a final `synthesize` round |
| Fusion | parallel panel `generate`, `judge`, then `synthesize` |

The output cap defaults to `min(1024, floor(max_total_tokens / max_calls))` and
can be fixed with `--max-output-tokens`. The chosen cap and retry policy are
recorded in the runtime receipt. Provider model slugs come from the manifest's
`model` field while evidence rows use the stable local `model_id`.

The Go Looper client exposes the same accounting seam through
`looper.WithCallObserver`: native callers can reserve before dispatch and
settle after transport or parse errors. It also propagates the four stage names
to Base, Confidence, ReMoM, and Fusion calls, and preserves explicit usage
field presence for streaming and non-streaming responses.

## Deterministic fixture

`fixture_records(plan)` accepts only a synthetic plan and creates examples for
all cells: success, provider error, exhausted budget, unknown usage, zero output
tokens and cached candidates. These are hand-authored states, not a simulated
implementation of each algorithm. Unit tests round-trip and validate the bundle,
check bad references and states, and exercise both CLI commands without network
access. The tests are also discoverable by pytest.
