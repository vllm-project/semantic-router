# Routing and model evaluation loop

## 1. Route preview

Preview is deterministic control-plane evidence and makes no generation call:

```bash
vllm-sr route preview \
  --endpoint http://router-management:8080 \
  --model vllm-sr/auto \
  --prompt 'Implement a lock-free queue'
```

Use `--json` when an agent needs the complete signal and decision payload.

## 2. Routed probe

Probe sends a real OpenAI-compatible request through the inference listener:

```bash
export OPENAI_API_KEY='...'

vllm-sr route probe \
  --base-url http://router-inference:8899/v1 \
  --model vllm-sr/auto \
  --prompt 'Implement a lock-free queue' \
  --expect-recipe balanced \
  --expect-decision coding \
  --expect-algorithm multi_factor \
  --expect-selected-model qwen \
  --expect-response-model Qwen/Qwen3.8-Flash-Next
```

The JSON receipt includes status, latency, routing headers, body, and every
assertion. `--expect-selected-model` verifies the Router receipt, while
`--expect-response-model` verifies the upstream response body's top-level
`model` field when the backend exposes one. The base URL accepts either the
listener origin or its OpenAI `/v1` root. Exit code `2` means an assertion
failed.

## 3. Versioned workloads

Use `vllm-sr benchmark catalog` to discover routing/model-pool/joint workloads,
then validate and run an immutable manifest. Compare paired baseline and
candidate runs and gate the candidate before applying it broadly.

## 4. Intelligence 1.0

The full suite is MMLU-Pro, GPQA Diamond, HLE 1.0 text-only, LiveCodeBench v6,
SciCode, and Terminal-Bench 2.1. First inspect its exact sources:

```bash
vllm-sr benchmark intelligence list
vllm-sr benchmark intelligence plan \
  --model vllm-sr/quality \
  --base-url http://router-inference:8899 \
  --source-root .vllm-sr/benchmark-sources \
  --output .vllm-sr/benchmark-results/quality-1
```

Materialize each listed repository under its `source.cache_key`, check out the
exact revision, and keep it clean. Run the same command with `run` in place of
`plan`. The harness verifies source revisions, fails closed if an AIPerf-backed
Hugging Face dataset has moved, and relies on Inspect Evals' checksum-pinned
SciCode assets. It keeps runner output private and writes a secret-free
receipt. Set `HF_TOKEN` for gated GPQA/HLE data and `OPENROUTER_API_KEY` for the
HLE judges. HLE always passes `include_multi_modal=false`.

Physical and virtual models use the same command and scoring contract. The
`--model` value is the only subject identifier; the harness calls the routed
endpoint for every task. A partial run or any `--sample-limit` is useful for
smoke testing but not eligible for an Intelligence 1.0 score.

## Optimize and repeat

Join benchmark evidence with replay decisions, outcome feedback, backend
health, latency, tokens, and cost. Change one reviewed policy at a time. Re-run
the same frozen workload against baseline and candidate, then retain the
candidate only when its quality, cost, safety, and reliability gates pass.
