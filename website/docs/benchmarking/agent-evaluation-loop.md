---
title: Agent Evaluation Loop
description: Validate, route, probe, benchmark, and optimize vLLM Semantic Router without depending on Dashboard.
---

# Agent Evaluation Loop

An agent works directly with two runtime contracts: the Router management API
for configuration and routing inspection, and the Envoy listener for real model
requests. Dashboard is an optional viewer and is never part of the execution
path.

```text
canonical YAML
    │
    ├── validate → plan → compare-and-swap apply       Router :8080
    │
    ├── route preview                                  Router :8080
    │       └── signals + decision; no backend call
    │
    ├── route probe                                    Envoy listener
    │       └── real response + route receipt
    │
    └── benchmark
            ├── routing workload → recipe behavior and system outcome
            └── Intelligence 1.0 → physical or virtual model quality
```

## 1. Discover and change configuration

Discover only the contract needed for the current edit:

```bash
curl -sS 'http://localhost:8080/api/v1?audience=agent&visibility=primary'
curl -sS 'http://localhost:8080/openapi.json?capability=config&audience=agent'
vllm-sr config schema --endpoint http://localhost:8080 \
  --surface algorithm:multi_factor
```

YAML is the only operator-authored configuration format. The CLI and HTTP API
are equivalent transports over the same Router validators:

```bash
vllm-sr config validate --config candidate.yaml \
  --endpoint http://localhost:8080
vllm-sr config plan --config candidate.yaml --mode replace \
  --endpoint http://localhost:8080
vllm-sr config apply --config candidate.yaml --mode replace \
  --endpoint http://localhost:8080
```

`plan` executes the same parse, normalization, semantic validation, and
hot-reload feasibility checks as mutation without writing. `apply` plans again
and uses the returned ETag as its compare-and-swap precondition. A plan that
changes listeners or provider backend topology returns `RESTART_REQUIRED`
because those fields are rendered into Envoy; activate that candidate through
the deployment workflow instead of the Router mutation API. For local Docker,
ask before replacing the running stack, then use
`vllm-sr serve --config candidate.yaml --replace-active-config`. Ordinary
`serve` preserves Dashboard-edited active state.

## 2. Verify routing in two stages

Preview checks routing policy without spending model tokens:

```bash
vllm-sr route preview \
  --endpoint http://localhost:8080 \
  --model vllm-sr/auto \
  --prompt 'Implement a lock-free queue' \
  --json
```

Probe then sends a real request through Envoy and asserts the resulting route:

```bash
vllm-sr route probe \
  --base-url http://localhost:8899/v1 \
  --model vllm-sr/auto \
  --prompt 'Implement a lock-free queue' \
  --expect-recipe balanced \
  --expect-decision coding \
  --expect-algorithm multi_factor \
  --expect-selected-model qwen \
  --expect-response-model Qwen/Qwen3.8-Flash-Next
```

The probe emits a machine-readable receipt with HTTP status, latency, routing
headers, response, and assertions. `--expect-selected-model` checks the Router
receipt; `--expect-response-model` checks the upstream OpenAI response body.
Use the latter when that backend exposes a stable top-level `model` value. A
failed assertion exits with code `2`.
The base URL may be either the Envoy listener origin or the standard OpenAI
root ending in `/v1`.
Preview success proves decision behavior only. A selected-model header proves
the Router's choice but not which backend answered; response-model evidence
closes that gap when available. One probe still does not substitute for a
benchmark.

## 3. Run comparable benchmarks

Use `vllm-sr benchmark` for immutable routing workloads. Use the fixed
Intelligence 1.0 harness for standalone or virtual model quality:

```bash
vllm-sr benchmark intelligence list
vllm-sr benchmark intelligence plan \
  --model vllm-sr/quality \
  --base-url http://localhost:8899 \
  --source-root .vllm-sr/benchmark-sources \
  --output .vllm-sr/benchmark-results/quality-1
```

The six 1.0 leaves are MMLU-Pro, GPQA Diamond, HLE 1.0 text-only,
LiveCodeBench v6, SciCode, and Terminal-Bench 2.1. HLE is always the frozen
2,158-question text-only subset. The harness verifies clean runner revisions;
attests the Hugging Face revisions used by AIPerf before execution; and uses
Inspect Evals' checksum-pinned SciCode problems and numeric test asset. It
records those identities and execution conditions in private, secret-free
receipts. `--sample-limit` is smoke evidence and cannot enter the index.

A physical model name and a virtual model name use the same `--model` field and
the same endpoint contract. A virtual score is measured end to end; it is never
assembled from the scores of its member models.

## 4. Optimize from evidence

Join benchmark results with route receipts, replay decisions, outcome feedback,
latency, token use, failures, and cost. Change one reviewed policy at a time,
rerun the same frozen workload for baseline and candidate, and retain the
candidate only when its declared quality, cost, reliability, and safety gates
pass.

Credentials belong in environment variables named by `--token-env` or
`--api-key-env`. Do not put credential values in YAML, URLs, command arguments,
logs, or receipts.
