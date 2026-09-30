---
title: sr-bench-nano (temporary)
---

# sr-bench-nano (temporary, experimental)

:::warning Temporary scope
sr-bench-nano exists only on the `exp/sr-bench-nano` branch. It is not part of the
sr-bench 1.0 contract, may change or disappear without notice, and its scores are
not sr-bench scores.
:::

sr-bench-nano is a small, frozen five-benchmark scope for evaluating **your own
models**. It fixes only the questions, the graders and the run protocol. It ships
no model, recipe, endpoint or arm configuration: you register your own
OpenAI-compatible targets. Results stay with you in your local sr-bench store.
They are never committed to this repository and nano has no shared results
location.

## Composition

Every benchmark weighs 0.2. The nano score is the equal-weight mean of the five
per-benchmark accuracies. Each split is a contiguous window of sr-bench's
stratified-hash-v1 order (seed `20260918`) over the normalized population, so
`nano` and `nano-holdout` never overlap.

| Benchmark | nano | nano-holdout | Population | Stratified by |
| --- | ---: | ---: | --- | --- |
| MMLU-Pro | 150 | 500 | all 12,032 test tasks | category |
| SimpleQA Verified | 100 | 300 | all 1,000 tasks | topic |
| GPQA Diamond | 60 | 138 | all 198 tasks | subdomain |
| LiveCodeBench | 60 | 150 | 342 release v5+v6 problems | difficulty |
| HLE (exact-answer, text) | 60 | 150 | 849 judge-free tasks | category |

- The MMLU-Pro `nano` split is the first 150 tasks of sr-bench's quick set.
  `nano-holdout` is the first 500 tasks of sr-bench's standard holdout set.
- GPQA is split exhaustively. Its labels were seen in earlier project work, so
  treat it as a retest, not an unseen holdout.
- LiveCodeBench uses the problems added in releases v5 and v6 (`test5.jsonl` and
  `test6.jsonl`, contests from 2024-09-22 to 2025-04-06) at the revision sr-bench
  pins for `release_v6`.
- The HLE population keeps only text-only `exactMatch` items whose reference is
  numeric (an integer, decimal or `a/b`) or a plain string of at most 4 words and
  40 characters. Items whose answers are free-form and would need an LLM judge are
  excluded.
- `nano-holdout` is a frozen holdout. Run it only when it is explicitly requested,
  and never tune against it.

## Frozen id list v1

`cli/sr_bench/nano_ids_v1.json` records the seed, the dataset revisions and file
SHA-256s, and for every task its id, stratum and SHA-256s: `prompt_sha256` covers
the canonical messages, and `answer_sha256` or `source_record_sha256` plus
`content_sha256` cover the rest. It contains no dataset text. Hashes use sr-bench's
canonical JSON: sorted keys, compact separators, and `ensure_ascii=False`.

The list identity is its self-hash, `sha256` =
`576b027954761527a30e4878db2fc69b2aaee7e7b0c1522e106c523e520ef964`. The raw file
SHA-256 is `f6523e3d12d5f17d55682d1c5072844cda48a75bcd82f0d255758fc305d6711f`.
`vllm-sr benchmark nano show` re-verifies the self-hash and prints it with the
per-split `ids_sha256` values. `nano prepare` refuses any source file or task whose
hashes differ, and planning refuses any nano case that is not a frozen task of
its split.

## Run protocol

- **One generation per task per target.** Every benchmark, including GPQA, is
  run once per task and target. There are no repeated draws.
- **No default output cap.** Nano never sends `max_tokens` unless a target sets
  one. Omitting `max_tokens` does **not** guarantee uncapped output, because some
  servers and gateways apply their own small default when it is absent (1,024
  tokens has been observed), and vLLM's default depends on `--max-model-len`. Set
  an explicit large value per target, for example the model limit minus some
  headroom:
  `request_params: {max_tokens: 120000}`. The report records for each target
  whether `max_tokens` was sent and its value. It also lists
  `suspected_output_cap_cases`: cells that ended with `finish_reason=length` or
  whose output token count equals a round number such as 1,024 or 4,096.
- **Per-request wall-clock timeout.** Each model request has 1,800 s (30 min),
  with `total_timeout_s` and `idle_timeout_s` both 1,800. A case that exceeds it
  gets the result status `timeout`, counts as incorrect in the full denominator,
  and the run continues. Reports show `timeouts` separately from `failed`. Other
  transport or grader failures still stop the run, as in sr-bench 1.0.
- **Streaming.** Targets stream by default. For providers that truncate streamed
  responses, set `stream: false` on a single-model target. Nano never uses
  logprobs.
- **Cost.** Uncapped output cannot bound the cost of a request before it is sent,
  so nano uses `cost_policy: capability_only`. Token usage and any known costs are
  still reported.
- A repetition guard and an 8 MiB character guard protect memory. A degenerate
  repeated output is recorded as incorrect, not as a run failure.

## Graders

| Benchmark | Grader |
| --- | --- |
| MMLU-Pro, GPQA | sr-bench deterministic multiple-choice grader (`sr-bench-mcq-final-v2`) |
| HLE | `sr-bench-nano-hle-exact-v1`: HLE's official exact-answer system prompt is prepended to the question. The grader reads the last `Exact Answer:` line (falling back to the last `\boxed{}`); numeric answers must match within the reference's precision, and other answers must match as normalized strings. No LLM judge is used. |
| LiveCodeBench | sr-bench's sandboxed `lcb_runner` execution at the pinned LiveCodeBench revision (no network, read-only, all capabilities dropped) |
| SimpleQA Verified | one configurable LLM grader, described below |

### SimpleQA grader

Nano has **no default grader model**. You must configure the grader endpoint,
model, API-key environment variable and any optional extra headers. If you don't,
`nano manifest` and planning fail immediately with an explicit message. The
grader uses the official SimpleQA grading template, pinned by SHA-256 from
[openai/simple-evals](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/simpleqa_eval.py)
(MIT). Nano forces temperature 0 and maps the grader's `A`, `B` or `C` reply to
correct, incorrect or not attempted. A reply without a letter stops the run
rather than being guessed.

We recommend a strong, large open-weight model that is not a typical candidate
model: **DeepSeek-V4-Pro** (open weights, MIT license, widely available through
public APIs). Use it with the official template, temperature 0 and thinking
disabled. How you disable thinking depends on the provider. Use the provider's
non-thinking mode, or pass a request parameter such as
`--grader-param 'chat_template_kwargs={"thinking":false}'` on vLLM or SGLang
deployments. Any strong model can serve as the grader. The report's
`nano.graders` section records the grader's model, base URL, request parameters,
header names, template hash and grader version. Compare runs only when they used
the same grader.

## Commands

The steps below go from installing the branch to reading a report. Gated
datasets (GPQA and HLE) need a Hugging Face token whose account has accepted
their terms. LiveCodeBench grading needs Docker and `uv`.

```bash
# 1. Install from the branch.
pip install "vllm-sr[bench] @ git+https://github.com/vllm-project/semantic-router.git@exp/sr-bench-nano#subdirectory=src/vllm-sr"

# 2. Confirm you have the same frozen id list (sha256 576b0279...).
vllm-sr benchmark nano show

# 3. Install the pinned LiveCodeBench harness and its offline sandbox image.
vllm-sr benchmark setup --benchmark livecodebench --install --build-sandbox

# 4. Download the pinned sources, verify every hash and store the nano split.
export HF_TOKEN=...   # account with access to GPQA and HLE
vllm-sr benchmark nano prepare --split nano > nano-dataset.json
```

Describe your own targets in a file. Credentials are always references to
environment variables: `api_key_env` for a bearer key and `header_env` for extra
headers. Header values never appear in the manifest or the report.

```yaml
# my-targets.yaml
- id: my-model
  kind: single
  base_url: https://api.example.com/v1
  model: my-served-model
  api_key_env: MY_MODEL_API_KEY
  header_env:              # optional extra headers: header name -> env var
    X-Tenant: MY_TENANT_HEADER
  request_params:
    max_tokens: 120000     # strongly recommended; see "No default output cap"
    temperature: 0
  # stream: false          # only if the provider truncates streamed output
```

```bash
# 5. Point at your SimpleQA grader (flags or SR_BENCH_NANO_GRADER_* env vars).
export SR_BENCH_NANO_GRADER_BASE_URL=https://grader.example.com/v1
export SR_BENCH_NANO_GRADER_MODEL=deepseek-v4-pro
export SR_BENCH_NANO_GRADER_API_KEY_ENV=GRADER_API_KEY
export GRADER_API_KEY=...
vllm-sr benchmark nano manifest \
  --dataset nano-dataset.json \
  --targets my-targets.yaml \
  --output nano-run.json
# optional: --grader-header X-Tenant=GRADER_TENANT --grader-param 'chat_template_kwargs={"thinking":false}'
#           --grader-no-stream --concurrency 4 --lcb-sandbox-image sha256:...

# Optional first: a 2-task-per-benchmark smoke manifest.
# vllm-sr benchmark nano manifest --dataset nano-dataset.json \
#   --targets my-targets.yaml --sample 2 --output nano-smoke.json

# 6. Run once and read the report.
vllm-sr benchmark run --manifest nano-run.json --detach --idempotency-key nano-1
vllm-sr benchmark show RUN_ID
vllm-sr benchmark report RUN_ID --output nano-report.json
```

The sr-bench service makes every model and grader call, so it reads the
credential variables (`MY_MODEL_API_KEY`, `GRADER_API_KEY`, and the header
variables) from its own environment. Export them before the first
`benchmark run`, which starts the service automatically. If the service is
already running without them, or if the store already has earlier runs (after
that, the service no longer starts automatically), start it yourself in a
terminal that has the variables exported, and keep that terminal open:

```bash
vllm-sr benchmark serve
```

Each manifest file name and idempotency key can be used only once. To run
again, use a new `--output` name and a new `--idempotency-key`.

The service also writes the report to `<store>/runs/RUN_ID/report.json`. The
default store is `~/.local/share/vllm-sr/sr-bench` unless `--store`,
`SR_BENCH_STORE` or a managed local stack selects another one. Per target, the
report provides:

- `summary.targets[].nano_score`: the equal-weight score, set only when the full
  frozen split finished with every cell completed or timed out.
- `macro_accuracy`, `accuracy`, and `timeouts`.
- Per-benchmark rows.
- A `nano` section with the id-list hash, the graders, and the output-cap
  evidence.

For the frozen holdout, prepare with `--split nano-holdout` and repeat steps 5
and 6, but only when a holdout evaluation has been explicitly requested.

## Dashboard

Nano runs appear in the Dashboard evaluation pages as custom-profile runs. The
Dashboard's profile cards still offer only smoke, quick and standard, so prepare
nano datasets with the CLI.
