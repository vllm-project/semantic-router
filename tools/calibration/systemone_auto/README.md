# System One auto research pilot

Compare an authored confidence cascade with a learned escalation policy on the
same independently labelled requests. Every request starts with Decision 2.0
Kai and may use one additional Decision 2.0 model. Qwen3.8-Flash-Next is measured
separately as a typed-answer baseline; that comparison does not measure an LLM
judge that sees a previous answer.

The pilot supports Choice, Noul and Score. It does not presume that cascading
improves quality, latency or cost. Preserve negative results alongside gains.

## Prepare the dataset

Use the repository environment and keep downloaded data and results outside
tracked source:

```bash
make harness-bootstrap
export PYTHONPATH=tools/calibration
.venv-agent/bin/python -m systemone_auto download \
  --directory .agent-harness/systemone-auto/data
.venv-agent/bin/python -m systemone_auto prepare \
  --data-dir .agent-harness/systemone-auto/data \
  --output .agent-harness/systemone-auto/pilot.json
```

The default contains 384 independent public source groups: 128 each from
Banking77, BoolQ and DynaSent. They are partitioned into 192 training, 96
calibration and 96 held-out groups. DynaSent bundles express multiple views of
one human annotation, not independent examples. Another 32 deterministic
arithmetic requests form a diagnostic cohort, excluded from training and the
primary aggregate. This is a small pilot, not evidence for a one-percentage-point
quality guarantee or broad task coverage.

Source URLs, immutable Git revisions where available, content hashes,
attribution and licenses are retained in `sources.json`. Banking77 and DynaSent
use CC BY 4.0; BoolQ uses CC BY-SA 3.0. Preserve their attribution and applicable
license terms when sharing a derived dataset. Pass an earlier receipt to
`download --lock sources.json` to require the same source bytes.

## Collect real observations

Create a private `targets.json` array. Each target has a distinct `name`, a
`protocol` (`systemone` or `chat`), the public `model_id`, its immutable 40-character
`revision`, and the full `endpoint` URL. Set `served_model_id` if the service uses
a different concrete name. Set `api_key_env` to the name of an environment
variable when authentication is required; credentials and endpoint URLs are
not copied into result manifests.

The native model set is Decision 2.0 Kai, Eos, Sol, Nox, Lux and Vega. The only
Chat baseline is `Qwen/Qwen3.8-Flash-Next`. Use concrete model names, not an auto
entrypoint, when collecting the paired reference matrix. Freeze runtime image,
model revision, dtype, context limits and resource allocation in the deployment
receipt before comparing results.

```bash
.venv-agent/bin/python -m systemone_auto collect \
  --dataset .agent-harness/systemone-auto/pilot.json \
  --targets .agent-harness/systemone-auto/targets.json \
  --output-dir .agent-harness/systemone-auto/collection \
  --target kai --limit 3
```

Inspect the three responses, coverage and model identity before removing
`--limit`. Repeat with `--target` for each provisioned model when capacity requires
sequential deployment. Keep the complete target array unchanged throughout the
run. Collection resumes existing records without retrying failed requests or
silently changing their costs. A timeout, HTTP error or malformed answer remains
in the dataset. An incomplete matrix cannot be used for the final paired replay.

The collector uses serial requests without retries. Its observed elapsed time
includes transport and server queuing; it is neither GPU compute time nor an
online cascade latency measurement. Sequential model deployments also introduce
time-of-run effects. Use a separately controlled serving experiment to validate
end-to-end latency and provisioned-resource cost.

## Compare the two policies

```bash
.venv-agent/bin/python -m systemone_auto replay \
  --dataset .agent-harness/systemone-auto/pilot.json \
  --collection-dir .agent-harness/systemone-auto/collection \
  --output-dir .agent-harness/systemone-auto/results \
  --base kai
```

Both methods share a two-call maximum and the same calibration cost budgets.
The default cost metric is native `meta.compute_ms`, validated against the
pinned model revision and a consistent runtime profile. This is reported
compute elapsed time, not GPU-active time or billable GPU usage. Missing timing
does not become zero: compute-based replay refuses that matrix. You can run
`--cost-metric client_elapsed_ms` as a separate comparison that retains failures
and their observed client cost. The metric and its source are recorded in the
policy artifact. Neither metric establishes routed end-to-end latency.
The cascade selects one fixed escalation model and a confidence threshold.
The learned method fits lightweight ridge heads to predict whether each
alternative would correct or damage the entire bundle, then subtracts a cost
penalty. Neither method fits on held-out labels. An isotonic calibration maps
the first model's minimum top probability to estimated bundle correctness; it
does not improve confidence ranking or certify a risk bound.

Selection minimizes whole-bundle error, with typed loss and observed call cost
as tie-breakers. A Score answer is correct when rounding its predicted level
to the nearest integer recovers the gold level; normalized absolute error is
also reported. Missing or invalid answers count as failures. Missing
probabilities do not erase an otherwise valid point answer, but cannot supply
confidence evidence. The common native acceptance floor requires usable
probability evidence for every answer. A learned policy retains a previously
accepted first answer if its upgrade fails. In an authored cascade, escalation
means the first answer failed its additional acceptance gate, so a failed second
call leaves the whole request unresolved. Unresolved requests stay in the error
denominator. Direct-model quality baselines preserve valid point answers even
when confidence evidence is absent; they do not claim cascade deliverability.

Outputs include the data-only `policy.json`, quality-calibration estimates,
held-out per-request actions and `replay.json`. The latter contains realized
cost, escalation rates, whole-bundle accuracy, typed losses and a paired group
bootstrap interval. Held-out cost may differ from the calibration budget and
is reported as observed, never retrospectively constrained using test labels.
The exported policy includes only native-to-native actions. It is not an LLM
judge policy or a complete online performance result.

The report also includes direct-model and always-escalate controls selected
within the same calibration cost ceiling. A random control preserves the
selected policy's per-model action counts while shuffling which sources receive
them, using twenty predeclared seeds. Its group interval treats the original
source groups as the sample units; repetitions do not enlarge the labelled test
set. Realized random-control cost may differ because inputs have different
execution costs. The label-aware native oracle is only a diagnostic upper bound.
Rescue/harm tables use training and calibration examples for research exploration;
held-out errors must not become a tuning set.

Before deploying a learned artifact, bind its measured actions to the YAML
stage and provider aliases. For example, a binding entry is
`"kai": {"stage": "fast", "model": "decision-fast"}`. Supply every measured
action in the binding JSON; removing candidates requires a separately declared
experiment and validation.

```bash
.venv-agent/bin/python -m systemone_auto export-policy \
  --policy .agent-harness/systemone-auto/results/policy.json \
  --bindings .agent-harness/systemone-auto/bindings.json \
  --output .agent-harness/systemone-auto/deployment-policy.json
```

The command reports the artifact SHA256 for `algorithm.policy.sha256`. Each
action binds its provider alias to the observed model revision, content digest
and runtime metadata. Renaming stages preserves coefficients and all measured
candidates; it does not authorize repointing an alias to a different model.
The pilot quality-calibration file is exploratory evidence, not a certified
acceptance artifact for `evaluation.calibrations`.

## Validate the harness

```bash
make test-calibration
```

Hermetic loopback tests cover authenticated collection, resume identity,
failure accounting, grouped partitions and held-out-label isolation. The shared
`tests/fixtures/native-features.json` fixture keeps Python training features and
Go serving features consistent.

## Measure an actual terminal judge

A direct Qwen answer is not evidence for a judge that reviews native answers.
Keep those experiments in separate collections. Generate label-free JSONL inputs
with the serving implementation:

```bash
make build-systemone-judge-requests
bin/systemone-judge-requests --model qwen/qwen3.8-flash-next \
  < native-candidate-inputs.jsonl > judge-requests.jsonl
```

Each source record contains its `record_id`, `group_id`, `cohort`, `split`, the
original `request`, and `candidates` with `stage`, `model` and the actual native
response `body`. Unknown fields, including ground-truth labels, are rejected.
The helper uses `systemone.JudgeRequest`; its shared golden verifies exact prompt
text bytes, candidate order, constrained selection schema and the 128-token
output limit. It selects one whole native candidate or abstains. It cannot
synthesize a corrected answer when both candidates are wrong.

Freeze the generated payload digests, input manifest and deployment receipt
before collecting. The judge collector validates their identities, accepts only
the pinned FlashNext model, and requires a receipt with the server-wide
`default_chat_template_kwargs: {enable_thinking: false}` setting. Authentication
and endpoint values stay in the private target object.

```bash
.venv-agent/bin/python -m systemone_auto collect-judge \
  --requests judge-inputs/requests.jsonl --manifest judge-inputs/manifest.json \
  --target judge-target.json --deployment-receipt judge-deployment.json \
  --output-dir judge-observations --limit 3
```

Remove `--limit` after checking the smoke responses. Requests run once without
retries. An abstention, timeout, invalid selection or incomplete response remains
an unresolved bundle in `score-judge`. This measures the selector component;
measure a fully resident cascade separately before claiming end-to-end latency.

A smaller native pool is also a separate experiment. Declare its pool, base,
dataset digest, two-call cap and all five cost ceilings in a frozen protocol,
then pass `replay --protocol protocol.json`. Its artifact binds that protocol and
contains only the declared actions. Do not trim a full-pool artifact after
looking at held-out results.
