# Router Flow Real Eval Plan

This directory is the real benchmark path for Router Flow. The older
`bench/router_flow/flow_eval.py` suite is a one-prompt-per-family proxy smoke
test; it is useful for development, but it is not publishable benchmark data.

The target comparison table should have public reference columns copied from
`bench/router_flow/public_reference_scores.json`, and VSR columns generated
locally:

- `VSR 1.0`, served through `vllm-sr/auto` with a single open-model backend
  family for the benchmark-specific recipe
- `VSR 1.0 Pro`, served through `vllm-sr/auto` with the best closed-model recipe

The current GLM-5.2-only campaign focuses on `VSR 1.0` for HLE text and
SWE-Bench Pro. It does not rerun native GLM-5.2; native reference values are
joined from the official GLM-5.2 benchmark table in
`public_reference_scores.json`.

The Fugu technical report reports that SWE-bench Pro, TerminalBench 2.1,
LiveCodeBench, GPQA-Diamond, HLE, SciCode, and related rows are measured with
EvalScope v1.8.1 or benchmark-native harnesses. This directory therefore treats
EvalScope reports as the source of truth for vLLM-SR rows and keeps the older
proxy suite clearly labeled as non-publishable smoke data.

## Install

Use a dedicated virtualenv on the eval machine:

```bash
python -m venv .venv-eval
. .venv-eval/bin/activate
python -m pip install -e 'bench[real_eval]'
```

Some heavy rows need additional adapters:

- SWE/LiveCodeBench/SciCode need Docker or EvalScope sandbox support. The
  runner prebuilds suite-declared local sandbox images such as
  `scicode-benchmark:latest` before invoking EvalScope so the scorer does not
  try to pull local image names from Docker Hub.
- TerminalBench needs `evalscope[terminal_bench]` and a supported agent wrapper.
- Tau3 Banking needs the tau-bench/tau2 knowledge dependencies and simulator
  setup.

## What To Run First

Start with the EvalScope-backed benchmarks that map directly to the public
table and exercise different failure modes:

| Public row | EvalScope dataset | Why it is first |
| --- | --- | --- |
| GPQA-D | `gpqa_diamond` | Catches answer-format and science MCQ regressions. |
| Humanity's Last Exam | `hle` text-only | Expert QA with judge scoring; multimodal still needs a vision-capable router path. |
| LiveCodeBench | `live_code_bench` Jan-Apr 2025 | Code generation with executable tests; formal mode uses the public technical-report date window and five generation retries. |
| SciCode | `scicode` with background | Scientific code generation with sandboxed checks. |
| Long Context Reasoning | `aa_lcr` | Long-context reasoning. |
| MRCRv2 | `openai_mrcr` 8-needle <=128K | Long-context recall aligned with the public technical-report MRCR setting. |

After those are stable, add `swe_bench_verified_mini_agentic`,
`terminal_bench_v2_1`, and `tau3_bench` banking. CharXiv Reasoning needs a
custom multimodal adapter before the numbers are comparable.

## Run

Serve the router with a benchmark-specific auto config that exposes
`vllm-sr/auto`, then run a dry run:

```bash
python bench/router_flow/real_eval/run_evalscope_suite.py --dry-run
```

Run the default aligned-core smoke suite:

```bash
python bench/router_flow/real_eval/run_evalscope_suite.py \
  --api-url http://127.0.0.1:8899/v1 \
  --output-root bench/router_flow/results/evalscope-smoke
```

Collect the completed EvalScope reports into the Fugu-aligned table and charts:

```bash
python bench/router_flow/real_eval/collect_evalscope_results.py \
  --output-root bench/router_flow/results/evalscope-smoke \
  --output-dir bench/router_flow/results/evalscope-report \
  --require-complete
```

The collector reads only EvalScope `reports/{model_id}/{dataset}.json` files,
extracts the configured metric from `evalscope_suite.yaml`, normalizes 0-1
metrics to 0-100, joins `public_reference_scores.json`, and writes:

- `evalscope_scores.json`
- `benchmark_table.md`
- `benchmark_table.csv`
- `overall_bars.svg`
- `benchmark_bars.svg`

Run one benchmark while iterating on bugs:

```bash
python bench/router_flow/real_eval/run_evalscope_suite.py \
  --benchmark gpqa_d \
  --model auto \
  --limit 20 \
  --output-root bench/router_flow/results/gpqa-omni-smoke
```

Use `--limit-mode formal` only after the smoke suite passes and cost/latency are
under control.

## AMD Matrix Runner

On the AMD eval host, the publishable path exposes one public model API:
`vllm-sr/auto`. Use the matrix helper to sync the EvalScope suite, switch the
mounted router config to the benchmark-specific auto recipe, regenerate Envoy
from that recipe's `providers.models[].backend_refs`, restart router and Envoy,
run the selected benchmark adapters, collect the joined table, and optionally
pull the artifacts back:

```bash
export VLLM_SR_AMD_HOST=<ssh-target>
python bench/router_flow/real_eval/run_amd_eval_matrix.py \
  --benchmark gpqa_d \
  --benchmark live_code_bench \
  --limit 1 \
  --pull
```

For the Kimi K2.7 Code direct-versus-router LiveCodeBench comparison, use the
Kimi recipe set. It switches the router to
`bench/router_flow/configs/amd_auto_livecode_kimi_k27_code_omni.yaml`, runs
`auto` as `VSR 1.0` through the local router, and runs
`kimi_k27_code_native` directly against OpenRouter as `Kimi K2.7 Code`:

```bash
python bench/router_flow/real_eval/run_amd_eval_matrix.py \
  --host <ssh-target> \
  --recipe-set kimi_k27_code \
  --benchmark live_code_bench \
  --limit-mode formal \
  --output-root results/livecode-kimi-k27-code-vsr1-formal175 \
  --report-dir results/livecode-kimi-k27-code-vsr1-formal175-report \
  --require-complete \
  --pull
```

For the current GLM-5.2-only HLE/SWE campaign, use `glm52_hle_swe`. It runs
only `auto` as `VSR 1.0`; official GLM-5.2 native values are reference columns,
not local model arms:

```bash
python bench/router_flow/real_eval/run_amd_eval_matrix.py \
  --host <ssh-target> \
  --recipe-set glm52_hle_swe \
  --benchmark hle_text \
  --limit 24 \
  --output-root results/hle-glm52-vsr1-smoke-b24 \
  --report-dir results/hle-glm52-vsr1-smoke-b24-report \
  --require-complete \
  --pull
```

SWE-Bench Pro uses EvalScope's agentic Docker harness. The harness is not
safely batchable inside one process, so its suite keeps `eval_batch_size: 1`;
use outer resumable shards for higher throughput.

For the competitive HLE path, use `hle_hybrid`. This keeps the public API on
`vllm-sr/auto`, but it is not GLM-only: the recipe combines GLM-5.2 breadth
samples with closed verifier/finalizer roles and labels the internal row
`VSR Hybrid`:

```bash
python bench/router_flow/real_eval/run_amd_eval_matrix.py \
  --host <ssh-target> \
  --recipe-set hle_hybrid \
  --benchmark hle_text \
  --limit 24 \
  --output-root results/hle-hybrid-smoke-b24 \
  --report-dir results/hle-hybrid-smoke-b24-report \
  --require-complete \
  --pull
```

Run only one benchmark while debugging router behavior:

```bash
python bench/router_flow/real_eval/run_amd_eval_matrix.py \
  --host <ssh-target> \
  --benchmark live_code_bench \
  --limit 1 \
  --continue-on-error \
  --pull
```

Use `--dry-run` first when changing remote paths. The script does not include
private hostnames, API keys, or provider credentials; the remote router config
must resolve its backend credentials from the existing router container
environment, the remote key file, or the remote shell environment. The generated
Envoy file is runtime state only; do not commit generated Envoy configs because
they contain provider authorization headers.

TerminalBench runs are long and SSH sessions can be unreliable. For one-task
feasibility probes, start the remote loop under `nohup`, then poll the log and
pull artifacts after `collect_evalscope_results.py` completes. The next
TerminalBench path should add one `amd_auto_terminalbench_omni.yaml` recipe and
run it through the same `vllm-sr/auto` model key.

The TerminalBench suite passes `extra_params.terminus2_kwargs`, and the runner
patches EvalScope versions that do not yet forward those kwargs into Harbor's
Terminus2 agent. The current config sets
`proactive_summarization_threshold: 0` because the default proactive
summarization can summarize without the latest terminal observation, causing the
agent to lose command output before scoring.

## Current Artifacts

The current joined matrix is tracked in
`bench/router_flow/real_eval/current_eval_matrix.json`. It includes:

- GPQA-Diamond formal-50 represented as `VSR 1.0 Pro` through `vllm-sr/auto`.
- LiveCodeBench formal-175 represented as `VSR 1.0 Pro` through
  `vllm-sr/auto`.
- HLE, SciCode, TerminalBench, and SWE-bench are intentionally absent until they
  have benchmark-specific auto recipes and fresh EvalScope outputs.

Raw EvalScope/TerminalBench outputs are intentionally under
`bench/router_flow/results/`, which is ignored as a run-output directory. The
current scorecard specs and rendered SVG/PNG files that should travel with the
repo are copied into `bench/router_flow/scorecards/`.

## Heavy Rows

These rows are real but expensive:

- SWE-bench: use `swe_bench_verified_mini_agentic` first. It needs Docker and can
  take hours because EvalScope builds or pulls benchmark images.
- TerminalBench 2.1: needs `evalscope[terminal_bench]`, Python 3.12+, Docker,
  and a supported agent wrapper.
- Tau3 Banking: needs `tau2[knowledge]` and user-simulator configuration.
- Long Context Reasoning: use `aa_lcr` or full MRCR after the bounded MRCR slice
  is reliable.

## Current Known Gap

The first real GPQA-D smoke exposed a Router Flow bug: dynamic flow can lose the
original answer-format contract and planner plan-parse failures can turn one
sample into a hard request failure. The runtime now includes:

- final synthesis instructions that preserve constrained output formats exactly;
- `on_error: skip` fallback for invalid dynamic planner output.

Re-run GPQA-D after deploying the updated router image before trusting larger
numbers.

## Completion Criteria For Publishable Numbers

- Every vLLM-SR table cell must come from `collect_evalscope_results.py`, not
  `flow_eval.py`.
- `evalscope_scores.json` must have an empty `missing` list for the selected
  benchmark/model set.
- The public Fugu/Fugu Ultra columns remain copied from
  `public_reference_scores.json`; do not relabel them as locally reproduced.
- Use smoke limits only for engineering iteration. Blog claims should specify
  the subset/limit unless the formal limit was completed.
