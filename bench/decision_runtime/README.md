# Decision SystemOne HTTP comparison

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
