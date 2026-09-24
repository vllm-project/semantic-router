# Decision HTTP performance comparisons

This directory has two independently labeled measurements. `run` is the
byte-identical, single-state SystemOne comparison below. `semantic` measures
synthetic logical workflows with mixed questions, concurrent requests, and
multiple states. For multiple states it compares old `/v1/systemone` fan-out
with new `/v1/decision/batches`; those are different HTTP protocols and wire
bytes. Neither mode measures decision quality.

## Synthetic semantic workloads

Run each of the six catalog models independently on two isolated, ready
services. The old service needs `/v1/systemone`; the new service needs both
`/v1/systemone` and `/v1/decision/batches`. The runner derives the new batch
URL from `--new-url`. Use the exact deployed source and weight revisions,
public-safe hardware labels, and the serving configuration's actual physical
batch capacity. The example ports are placeholders. Credentials belong only
in environment variables named by `--old-token-env` and `--new-token-env`.

```bash
.venv-agent/bin/python -m bench.decision_runtime semantic \
  --model llm-semantic-router/Decision-1.0-Kai-0.6B \
  --old-url 'http://127.0.0.1:<old-port>/v1/systemone' \
  --new-url 'http://127.0.0.1:<new-port>/v1/systemone' \
  --old-source-ref <old-source-commit> --new-source-ref <new-source-commit> \
  --old-model-revision <old-model-commit> \
  --new-model-revision <new-model-commit> \
  --old-hardware 'MI300X x1' --new-hardware 'MI300X x1' \
  --old-network-scope loopback --new-network-scope loopback \
  --old-physical-batch-size 1 --new-physical-batch-size 32 \
  --new-metrics-url 'http://127.0.0.1:<new-port>/metrics' \
  --question-counts 1,8,32 --state-counts 1,8,32 \
  --concurrencies 1,8,32 --variants 4 --seed 17 \
  --warmup 2 --latency-workflows 16 \
  --throughput-workflows 32 --rounds 2 \
  --output-dir .agent-harness/decision-semantic/kai
```

The generator constructs a deterministic cohort for every question/state
shape. Question types cycle through Noul, Choice, and Score, and counts of at
least three contain all three. States and wording vary by seed; these synthetic
fixtures have no accuracy labels. Before warmup or measurement, the runner
sends **each generated case once per arm, untimed**: one old single per state
and one new single or batch. It verifies the same logical state and question
contents across arms (allowing only the declared model ID and batch envelope
differences), validates each response contract, and checks per-state input-token
totals, answer IDs/types, Noul probabilities, Choice labels, probability
distributions, and Score expectations. Input tokens and Choice labels must
match exactly. A Noul crossing at 0.5 is recorded as a diagnostic, not an
eligibility gate, because callers choose the Noul action threshold. Each
probability may differ by at most `0.01` absolute by
default; Score may differ by at most that tolerance times the rubric range
(`level_count - 1`). The default accommodates observed small batching drift
without hiding the raw maximum and p50/p95/p99 probability deltas. Set
`--probability-tolerance` explicitly for a justified different bound. Every
request is validated against the runtime's strict contract before traffic.
The batch decision limit is 1,024,
so each question count multiplied by each state count must fit it. Repeat the
same shape grid, seed, variants, phase counts, concurrency levels, timeout, and
declared serving policy across all six models. Reconfigure the actual serving
physical batch capacity between runs when studying that axis; the CLI records
the declared capacity but cannot verify it from the server. `--concurrencies`
bounds in-flight HTTP requests per arm. The server's observed physical batches
can be captured from its own telemetry.

For a one-state shape, both arms send the same single-request bytes when they
use the same model ID. For a multi-state shape, the old arm sends one single
request per state and the new arm sends one shared-question batch request. A
logical workflow is complete only when every state returns a conforming answer.
The old fan-out is bounded by the selected HTTP concurrency. Each arm runs in
its own wave; throughput wave order alternates by round. Latency workflow order
alternates per item. There are no automatic retries or simultaneous old/new
inference waves.

Optional `--old-model-id <slug>` adapts only the old request's model ID before
timing. It requires the old response to echo that ID and still pass strict
request-relative validation. The receipt records the mapping and labels even
the one-state comparison as nonidentical wire bytes. No generic response
envelope rewrite is supplied; a service with another envelope needs a specific
audited adapter and explicit protocol labeling before measurement.

The historical Decision preview service returns a diagnostic envelope and
defines Choice/Score `confidence` as `max(probabilities)`. For that **specific**
old shape, add `--old-response-mode legacy_preview` (usually together with
`--old-model-id decision-nano-preview` or the matching old model slug). The
runner checks the old model ID, answer identities, input-token accounting, and
max-probability statistic, then projects the old answers onto the current
Decision-owned confidence formula solely for common response validation.
This projection happens **after** the complete old HTTP body is timed and its
raw hash recorded. The receipt identifies the adapter and does not claim
identical wire bytes or confidence semantics. Other old envelopes fail closed;
do not use this flag for a current strict `drun` service.

`--parity-policy require` is the default. A contract, token, label, or numeric
parity failure writes `audit.jsonl` and an audit-only `receipt.json`, exits 1,
and sends no warmup or measured requests. `--parity-policy report` is an
explicit exploratory override: it continues timing after the audit, but all
old/new performance and telemetry ratios are null, every comparison is marked
ineligible, and the receipt states whether the audit passed or failed. The
old preview projector is used only to validate its contract; the audit retains
raw old per-answer `input_tokens` and new per-state `usage.input_tokens`.
Output-token totals are recorded but not a gate: the old preview and current
API account for them differently. It never compares old and new numeric
confidence because their formulas differ.
Equal logical requests and token totals do **not** establish identical
server-side prompt rendering, and synthetic agreement is not an accuracy or
quality evaluation. If model-side prompts differ, report performance and
accuracy parity separately; do not call the comparison a validated co-design
gain.

`audit.jsonl` records each untimed case's old/new request and response hashes,
per-state input/output-token totals, old preview per-answer input-token counts
when present,
answer values and categorical outcomes, numeric deltas, and mismatch codes.
The receipt's audit summary includes the source and model-revision provenance,
cohort hash, tolerances, mismatch counts, and global delta quantiles. The
untimed calls are excluded from all latency, throughput, and metrics windows.
`samples.jsonl` records every timed HTTP attempt with request and response hashes,
status, error code, decision count, and timing offsets. `workflows.jsonl`
records each logical workflow's first-send to last-complete interval and full
success/failure. `receipt.json` records source, harness, generator, and cohort
hashes; declared deployment provenance; all settings; per-shape and per-arm
error counts; complete-workflow p50/p95/p99; HTTP p50/p95/p99; and successful
and attempted decisions per second. Percentiles use nearest rank over complete
conforming workflows. Throughput divides complete-workflow decisions by the
sum of each wave's first-send to last-completion window; partial old fan-out
does not earn successful decisions. Parsing and strict response validation are
outside the HTTP timing interval. The default files contain no URLs,
credentials, request bodies, response bodies, or answer text. Review all
metadata before publication.

Protected paired release runs additionally pass `--timed-semantic-evidence`.
This writes `timed-semantic.jsonl.gz` with the exact new-arm HTTP request and
response bodies from every throughput wave at concurrency 1, 8, and 32, linked to
each timed sample by case, round, sequence, request hash, and response hash.
The release gate recomputes both body hashes, validates the response against
the request criteria, and compares every answer with the sealed old-arm audit
at the same probability, Choice, Score, and input-token policy. New-arm
output-token counts must remain equal to its formal audit; the old preview's
different output-token accounting is not treated as parity. The archive has an
120 MiB decompressed limit, a 24 MiB compressed limit, a 1 MiB per-request limit,
and a 2 MiB per-response limit. These are synthetic cases, but the optional
archive does contain request and response bodies and should be reviewed before
publication.
The protected c1/c8/c32 grid has three captured concurrency cells instead of
two; its 120/24 MiB archive caps are the prior 80/16 MiB caps scaled by 3/2.
An oversized archive fails closed rather than skipping responses.

Optional `--old-metrics-url` and `--new-metrics-url` accept each service's
`GET /metrics` address. For every throughput wave, the runner takes a snapshot
immediately before and after the HTTP work, outside the timed interval. It
selects the request model's cumulative row-preparation seconds, preparation
count, physical batch count, physical batch row count, and fixed-size histogram
buckets. Sol graph capture, replay, and fallback counters are also selected;
missing event series count as zero. These metrics require the bounded `model`
label, and histogram buckets use `le`. `metrics.jsonl` stores selected
before/after values, response hashes,
deltas, and generic error codes; it does not store the raw metrics body or URL.
The receipt normalizes preparation seconds and observed physical rows per
complete decision, plus mean rows per physical batch. It shows an internal
old/new ratio only when both services expose complete counters. An old service
without these counters may omit `--old-metrics-url`; its internal baseline
remains unavailable. A requested metrics stream that fails or resets makes the
run exit 1 after writing the receipt. Isolate the services from other model
traffic so their process-wide counter deltas belong to these waves. Neither a
declared physical batch capacity nor an observed histogram is a capability
gate based on a particular model revision; record each actual deployment and
weight snapshot in every receipt.

Ratios appear only when the required parity audit passes, model revision,
hardware, and network-scope declarations match, warmup and measured responses
all conform, and both arms have successful
latency and throughput samples. Multi-state ratios are explicitly labeled
`single_fanout_vs_batch_protocol_workflow` and do not imply equivalent HTTP
payloads. Equal metadata alone does not prove equal deployment conditions.
Small fake-service or smoke runs demonstrate the harness, not a ROCm speedup.

The command exits 1 after writing receipts if the required audit or any
workflow fails. It exits 2
for invalid input or an existing output directory; it never overwrites a run.
After all six model runs, collate their receipts with `semantic-matrix`:

```bash
.venv-agent/bin/python -m bench.decision_runtime semantic-matrix \
  --receipts .agent-harness/decision-semantic/{kai,lex,eos,sol,nox,lux}/receipt.json \
  --output .agent-harness/decision-semantic/matrix.json
```

The matrix requires one receipt per catalog model with the same harness source,
case IDs, and settings. It preserves each model's own cohort hash, serving
metadata, and per-shape results. Run the fake-service checks with
`python -m unittest bench.decision_runtime.test_semantic -v` in the approved
validation environment.

## Byte-identical SystemOne comparison

This runner measures the six Decision 1.0 models one at a time against an old
SystemOne HTTP service and the integrated runtime. It measures performance and
response conformance. It does not run `deval`, score decision quality, or infer
accuracy from the synthetic cases.

Both services must accept the same canonical model ID at `POST /v1/systemone`.
The runner validates every request with the repository's `SystemOneRequest`
contract and sends the **same serialized bytes** to each arm. Responses must
pass `SystemOneResponse` validation and the request-relative answer checks.
It stores response hashes, not response content. A legacy direct model server
with a different path or wire shape needs a separately reviewed adapter; its
timings are not interchangeable with this HTTP protocol.

## Run one model

Use the source checkout's Python environment. Supply deployed source commit
SHAs or content digests, exact model weight commits, and public-safe hardware
labels. The example ports are placeholders; discover and configure the actual isolated
services before running. Put credentials in environment variables, never in
URLs or command arguments.

```bash
.venv-agent/bin/python -m bench.decision_runtime run \
  --model llm-semantic-router/Decision-1.0-Kai-0.6B \
  --old-url 'http://127.0.0.1:<old-port>/v1/systemone' \
  --new-url 'http://127.0.0.1:<new-port>/v1/systemone' \
  --old-source-ref <old-source-commit> \
  --new-source-ref <new-source-commit> \
  --old-model-revision <old-model-commit> \
  --new-model-revision <new-model-commit> \
  --old-hardware 'MI300X x1' --new-hardware 'MI300X x1' \
  --old-network-scope loopback --new-network-scope loopback \
  --warmup 4 --latency-pairs 32 \
  --throughput-requests 64 --rounds 2 --concurrency 4 \
  --output-dir .agent-harness/decision-http/kai
```

Add `--old-token-env ENV_NAME` or `--new-token-env ENV_NAME` only when that
service requires a bearer token. The variable names and values are not written
to receipts. Run each of Kai, Lex, Eos, Sol, Nox, and Lux into a separate new
output directory. `--model` lists the exact IDs. `--cases` can select another
reviewed JSONL cohort with `id`, `state`, and `questions` on each line.

The default four cases are synthetic Noul, Choice, Score, and mixed-question
requests. They are protocol/performance fixtures, not an evaluation dataset.
Keep the same case file, seed, warmup, latency pair count, throughput request
count, rounds, concurrency, and timeout for all six models. Report the case
cohort and sample count when publishing a result; a small smoke run is not a
steady-state performance study.

## Measurement and artifacts

For each paired latency request, the arm order alternates. Throughput runs use
separate waves so the old and new services do not actively infer at the same
time; wave order alternates across rounds. Each wave sends the same ordered
cases to one service. A new HTTP connection is used for every request in both
arms. There are no automatic retries.

Every request starts timing immediately before the HTTP send and stops after
the complete response body is read, including failed HTTP responses. Local
request construction, response JSON parsing, and conformance validation occur
outside the timed interval. Latency p50/p95/p99 use the nearest-rank rule on
successful requests; failures retain their own latency and error code in
`samples.jsonl`. Throughput is successful requests divided by the sum of each
wave's first-start-to-last-completion window. Attempted throughput and per-wave
windows are also reported.

`receipt.json` records model, declared old/new source and weight revisions,
declared hardware and network scope, fixture and harness hashes, settings,
latency percentiles, throughput, and error counts. The runner does not record
URLs, tokens, hostnames, raw HTTP bodies, or response text. The caller must use
public-safe source and hardware labels and review artifacts before publishing.
The runner trusts the caller's metadata; it cannot verify endpoint deployment
identity or hardware remotely.

The receipt gives a relative p50 latency and throughput ratio only when the two
arms declare the same model revision, hardware, and network scope, every warmup
and measured request conforms, and both phases have successful samples. A
public endpoint and a loopback endpoint therefore retain separate measurements
without a speedup claim. Equal network-scope labels alone do not prove equal
network paths; review deployment topology and load before interpreting a ratio.

The run exits 1 if any request fails or violates the response contract, after
writing the full receipt. It exits 2 for invalid inputs or an output directory
that already exists. It never overwrites a prior run.

After six runs, collate their receipts:

```bash
.venv-agent/bin/python -m bench.decision_runtime matrix \
  --receipts \
    .agent-harness/decision-http/kai/receipt.json \
    .agent-harness/decision-http/lex/receipt.json \
    .agent-harness/decision-http/eos/receipt.json \
    .agent-harness/decision-http/sol/receipt.json \
    .agent-harness/decision-http/nox/receipt.json \
    .agent-harness/decision-http/lux/receipt.json \
  --output .agent-harness/decision-http/matrix.json
```

`matrix` requires all six distinct models with the same fixture bytes, harness
source, and measurement settings. It preserves separate per-model baselines and
does not average ratios across model sizes.

## Focused validation

Run the fake-service contract checks on the approved validation machine:

```bash
.venv-agent/bin/python -m unittest bench.decision_runtime.test_harness -v
```

For a live smoke, use `--warmup 1 --latency-pairs 4
--throughput-requests 4 --rounds 2 --concurrency 2` against two isolated,
ready services. Inspect both `samples.jsonl` and `receipt.json` before increasing
the run size.
