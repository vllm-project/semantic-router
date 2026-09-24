# Decision ROCm image qualification

`decision_rocm_qualify.py` is the protected hardware qualification producer for
`decision-runtime-rocm`. It consumes an already-published **staging image by
immutable OCI digest**; it never builds or promotes an image. Run it from an
exact, clean source checkout on a trusted ROCm runner with `vllm-sr` installed
from that checkout, a Docker-compatible runtime, `skopeo`, registry read access,
and access to the six selected Hugging Face model snapshots. Supply credentials
through the runner environment, never as command-line arguments or evidence.

```bash
python3 tools/ci/decision_rocm_qualify.py \
  --owner <github-owner> \
  --candidate-ref 'ghcr.io/<github-owner>/semantic-router/decision-runtime-rocm-staging@sha256:<digest>' \
  --output-dir "$RUNNER_TEMP/decision-rocm-qualification" \
  --port <free-loopback-port> \
  --gpu-device <rocm-device-index>
```

The output directory must not exist before the run. Keep it private because
raw requests, responses, logs, artifact revisions, and timing observations are
included. The producer checks the candidate digest and OCI source/backend
labels against the checkout, then starts each of Kai, Lex, Eos, Sol, Nox, and
Lux through `drun` from that exact image on the selected GPU. It stops each
instance before starting the next. A failed check exits nonzero and does not
write the final `qualification.json`.

For every model the producer checks the detached `drun` launch identity,
`/ready`, live `/api/status` model and artifact identity plus scheduler limits
(concurrency 8, queue 32, active-row credits 4096), and response contracts
for a mixed Noul/Choice/Score single request, a two-state batch, a 32-question
single request, and an eight-state batch. It also sends eight concurrent
eight-question requests and records end-to-end p50/p95 latency, wall time,
decision throughput, and before/after physical-forward counters. The live
runtime's ROCm loader requires an available HIP GPU; the observed physical
forward metrics must cover all 64 concurrency-probe decisions. These are
measured observations, not caller-provided `passed` flags. The forward count
must be lower than the row count, confirming that at least some rows actually
coalesced into physical microbatches.

On success, `qualification.json` identifies the candidate and contains exactly
six model rows. Each row points to `models/<slug>.json` by SHA-256 digest; each
model evidence file records identities, checks, measured timings and counters,
and SHA-256 hashes of its raw request/response/status/metrics/launch files under
`raw/<slug>/`. The producer re-reads those files before finalizing its receipt
and validates the receipt with `decision_rocm_promotion.py`. Promotion must
consume the producer artifact from the protected runner, not an operator-edited
or externally supplied JSON file.

This receipt proves live compatibility and reports workload measurements for
the candidate. It does **not** establish a speedup over an earlier runtime;
that claim needs a separate paired benchmark on the same hardware, artifacts,
and request distribution. It also does not replace model-quality evaluation.
