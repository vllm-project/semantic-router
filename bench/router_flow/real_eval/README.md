# Router Flow EvalScope Runner

This directory runs maintained EvalScope benchmark adapters against Semantic
Router model aliases. Use it when a result needs benchmark-native data and
scoring. The proxy harness in the parent directory is for development smoke
tests only.

`evalscope_suite.yaml` is the source of truth for model aliases, benchmark IDs,
datasets, metrics, smoke and formal limits, generation settings, and sandbox
requirements. Variant suite files support specific recipe comparisons; select
one explicitly and retain it with the result.

## Install

From the repository root:

```bash
python3 -m venv .venv-eval
. .venv-eval/bin/activate
python -m pip install -e 'bench[real_eval]'
```

Some adapters require additional EvalScope extras, Docker images, datasets, or
simulators. Read the selected entry's `notes`, `sandbox`, and `dataset_args`
before starting a run. In particular:

- code and agentic benchmarks may execute untrusted generated code in Docker;
- TerminalBench requires its EvalScope extra, Python version, Docker, and an
  agent wrapper;
- Tau3 requires its knowledge and simulator dependencies;
- multimodal rows require a compatible router path and adapter.

Treat benchmark containers and generated code as untrusted workloads. Use an
isolated machine with constrained credentials and no unrelated host mounts.

## Inspect the run first

The runner prints the EvalScope commands without executing them in dry-run
mode:

```bash
python bench/router_flow/real_eval/run_evalscope_suite.py --dry-run
```

Use this to verify model aliases, output paths, dataset limits, and sandbox
preparation. A dry run is not benchmark evidence.

## Run and collect a smoke suite

First serve the benchmark-specific recipe at the API URL defined by the suite,
or override it explicitly:

```bash
python bench/router_flow/real_eval/run_evalscope_suite.py \
  --api-url http://127.0.0.1:8899/v1 \
  --limit-mode smoke \
  --output-root bench/router_flow/results/evalscope-smoke
```

To isolate one failure mode, select a model and benchmark and use a small
explicit limit:

```bash
python bench/router_flow/real_eval/run_evalscope_suite.py \
  --model auto \
  --benchmark gpqa_d \
  --limit 20 \
  --output-root bench/router_flow/results/gpqa-smoke
```

Collect only after EvalScope has written its report JSON files:

```bash
python bench/router_flow/real_eval/collect_evalscope_results.py \
  --output-root bench/router_flow/results/evalscope-smoke \
  --output-dir bench/router_flow/results/evalscope-report \
  --require-complete
```

The collector reads the metric named by the suite, normalizes metrics where
specified by the implementation, joins contextual values from
`public_reference_scores.json`, and writes:

- `evalscope_scores.json`;
- `benchmark_table.md` and `benchmark_table.csv`;
- `overall_bars.svg` and `benchmark_bars.svg`.

`--require-complete` exits non-zero when a selected model/benchmark cell is
missing. A complete smoke run is still a smoke result; use
`--limit-mode formal` only after the adapter, recipe, budget, and runtime are
stable.

## Heavy and adapter-dependent rows

Pass `--include-heavy` only when the required execution environment is ready.
Heavy rows deliberately default to serial or low-parallel execution when the
sandbox cannot safely batch work. `adapter_needed` entries are excluded unless
`--include-adapter-needed` is set and a compatible adapter has been supplied.

Prediction caches can reduce rerun cost, but they can also hide an endpoint or
recipe change. Use `--use-cache` only when the cached prediction identity
matches the current run; use `--rerun-review` when only judging or scoring must
be repeated.

## Optional remote matrix helper

`run_amd_eval_matrix.py` automates an opinionated SSH workflow: copy the suite
and recipes, switch the mounted router configuration, regenerate Envoy, restart
services, invoke EvalScope, collect the report, and optionally pull artifacts.
It assumes the remote layout and container names represented by its defaults.
It is not the general installation path.

Inspect every remote action first:

```bash
python bench/router_flow/real_eval/run_amd_eval_matrix.py \
  --host <ssh-target> \
  --recipe-set closed \
  --benchmark gpqa_d \
  --limit 1 \
  --dry-run
```

Override remote paths, container names, and image identity explicitly when the
host differs. Keep provider keys in the remote environment or a protected key
file; generated Envoy configuration may contain authorization headers and must
not be committed or attached to a public report.

## Evidence requirements

A result intended for comparison or publication should satisfy all of these:

- every local score comes from an EvalScope report consumed by
  `collect_evalscope_results.py`;
- `evalscope_scores.json` has no missing selected cells;
- public reference columns remain labeled as external references;
- the exact suite, recipe, source revision, immutable image, limits, overrides,
  and dependency versions are recorded;
- any smoke limit, subset, cache reuse, retry, or incomplete adapter coverage is
  stated alongside the number;
- raw prompts, responses, environment captures, and generated proxy configs
  have been reviewed for credentials and private data.

Generated results belong under `bench/router_flow/results/`, which is ignored
by Git. Curated scorecards should contain derived, reviewable data rather than
raw private run logs.
