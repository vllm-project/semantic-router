# Performance Microbenchmarks

The `perf/` module measures component-level Router hot paths with Go
benchmarks. Use it to compare allocations, bytes, and execution time for
classification, decision evaluation, cache operations, and Looper execution.
Looper microbenchmarks live beside the implementation and are included by the
repository Make targets.

These are not end-to-end quality or load benchmarks. For reasoning quality,
session routing, hallucination detection, and fusion evaluations, see
[`bench/`](../bench/README.md).

`make perf-test-unit` runs the parser, artifact identity, and regression contract
tests without downloading models or running benchmarks. `make check` selects
this target for performance changes. The dedicated performance CI job runs the
model measurements and reports numerical regressions as warnings without failing
CI. Benchmark execution, complete inventory, and matching model identities remain
required. `make verify DOMAIN=performance` runs the strict local `perf-check`.

## Run the benchmarks

From the repository root:

```bash
# Short local run
make perf-bench-quick

# Longer run with CPU and memory profiles
make perf-bench
```

Run one component family when iterating:

```bash
make perf-bench-classification
make perf-bench-decision
make perf-bench-cache
make perf-bench-looper
```

Classification and cache benchmarks serve their models through the model
runtime; install it once before running them:

```bash
make model-runtime-install
```

## Compare with the committed baselines

`perf-check` captures a fresh component and Looper run, parses the Go benchmark
output, compares it with `perf/testdata/baselines/`, and exits non-zero for a
blocking regression:

```bash
make perf-check
```

To inspect a comparison without applying the failure exit, first create the
raw input expected by `perf-compare`:

```bash
mkdir -p reports
make perf-bench-quick 2>&1 | tee reports/bench-output.txt
make perf-bench-looper 2>&1 | tee -a reports/bench-output.txt
make perf-compare
```

The parser writes `reports/current.json`; the comparison writes
`reports/comparison.json`.

Classification and cache benchmarks serve the router catalog's revision-pinned
Vela models through the model runtime, the way the Router does: one runtime
manager, the serving facade, and a runtime process per deployment, which
downloads its pinned model on first start (`VLLM_SRUN_CACHE_DIR` keeps
it). Failed downloads or inference fail the run. `VLLM_SR_DOMAIN_MODEL`,
`VLLM_SR_PII_MODEL`, `VLLM_SR_JAILBREAK_MODEL`, and `VLLM_SR_EMBEDDING_MODEL`
can point to local packages instead. No model IDs or revisions are maintained
separately in this module.

The previous classification and cache baseline files contain no measurements.
CI therefore measures those families against the PR base revision (or the prior
main revision for scheduled runs), using the current benchmark program and the
same pinned models on both revisions:

```bash
bash perf/scripts/compare-model-baseline.sh "$BASE_REF" reports
cd perf
go run ./cmd/perftest --compare-baseline=testdata/baselines \
  --current=../reports/current.json --model-baseline=../reports/model-baseline.json \
  --threshold-file=config/thresholds.yaml --output=../reports/comparison.json \
  --inventory=config/benchmark-inventory.json
```

This is the report-only comparison used by CI. Add `--fail-on-regression` to
request a strict local allocation check, as `make perf-check` does.

The base side runs its own model-runtime source on the installed dependencies.
An incompatible base fails with its compile/runtime output. A base revision that
predates the model runtime cannot run this harness at all: the helper then
writes a model baseline that records the reset, and the comparison reports the
model benchmarks as measured but not gated. Reports record the measured source
commit, registry revisions, the content hash of the package each runtime
loaded, device, precision, engine, and benchmark protocol. Missing measurements
or identity differences fail the comparison; Qwen3 and legacy classifier numbers
cannot become Vela baselines. Model correctness belongs to the runtime's golden
answers; these benchmarks do not publish accuracy from a missing optional dataset.

### Release comparisons

A release compares with its development cycle's base instead of its parent
revision:

```bash
make perf-check PERF_BASE_REF="$(python3 tools/ci/ci_plan.py performance-base --version X.Y.Z)"
```

The base is the previous release unless `PERFORMANCE_BASES` in
`tools/ci/ci_plan.py` declares one. v0.4.0 predates the model runtime, so 0.5.0
declares #4707 (`abae8ff99`): it pinned the classify benchmarks to the Vela 1.0
specialists, and its own Production Benchmarks job passed with this harness.
From 0.5.0 on, the previous release has the model runtime; when it can't run
the current harness, declare a reviewed `main` commit whose Production
Benchmarks job passed with it. Resolution refuses a previous release that
predates the model runtime rather than turning the comparison into a reset.

## Input-length measurement protocol

`BenchmarkClassifyInputLength` times the Vela Domain classifier on inputs of up
to 512, 2,048, and 8,192 tokens. Each input repeats the five fixture prompts
until one more repetition would exceed its size, and each result reports the
processed `tokens/op`. This classifier's deployment admits 8,192 tokens and
rejects longer input, while the other classification benchmarks share a
deployment that truncates at 512 tokens, so every size is one complete forward
pass.

## Cache measurement protocol

Each HNSW operation measures 100 public `LookupSimilarWithThreshold` requests
on each of five independently constructed graphs (500 requests/op). Linear
search uses one cache and 100 requests/op. Every graph contributes equally;
no graph or sample is retried or selected by its result. The eight scenarios retain their cache
sizes, HNSW/linear modes, 1/10/50 workers, similarity threshold, and Vela model.
A fixed corpus retains the original ten query topics and composed-query shape;
70% or 90% of lookup inputs reuse a base query. This is an input repetition
ratio, not an asserted model hit rate. The report records the observed hit rate.

Population uses unique request IDs and validates the actual stored entry count.
Population, embedding-memo warmup, worker startup, aggregation, and cleanup are
outside Go's timed/allocation interval. Each worker handles requests even in a
`-benchtime=1x` smoke run. Reports use batch `ns/op`, `B/op`, and `allocs/op`,
with explicit `requests/op` and `graphs/op`, pooled observed hit rate, and
request throughput. They do not
claim separate embedding/search phase timings. HNSW retains its production
randomized graph construction; the corpus and insertion order are fixed. The
five-graph ensemble reduces graph-dependent variation. Go `B.Loop` reuses the
prepared graphs throughout calibration so setup runs once per scenario.

The `cache=public-lookup-v3` protocol records these units and the warmed memo.
Both revisions run this exact harness and checkpoint; old standalone-harness
measurements, which included random setup amortized over Go's variable iteration
count, are not comparable to this protocol. Model inference is real during
preparation; these cases measure repeated cached lookups, while classification
benchmarks separately measure inference calls.

## Regression reports and required evidence

Thresholds in [`config/thresholds.yaml`](config/thresholds.yaml) are matched to
benchmark names in order; the first matching pattern wins. Unmatched names use
the `default` thresholds.

- CI reports `allocs/op`, `B/op`, and `ns/op` changes without failing on numerical
  regressions. Allocation regressions appear as workflow warnings and in the job
  summary; all comparison values remain in the uploaded artifacts.
- The explicit local `perf-check` and `--fail-on-regression` option still fail on
  `allocs/op` and `B/op` regressions. `ns/op` remains advisory because host speed
  and contention affect wall-clock measurements.
- Model benchmarks require measurements with the same artifact content and
  execution settings. Both measurements must cover the versioned inventory in
  `config/benchmark-inventory.json`; missing, extra, or duplicate workloads fail.

Go allocation metrics do not include the model runtime's allocations, process
RSS, or GPU memory. Benchmark commands exclude ordinary unit tests, which run in their own
checks. The former JSON/map microbenchmarks did not call production ExtProc and
are excluded from the production benchmark inventory.
Record the source revision, Go version, model artifacts, CPU, and benchmark
command whenever wall-clock results are shared.

## Profiling

`perf-bench` writes `reports/cpu.prof` and `reports/mem.prof`. Open them with:

```bash
make perf-profile-cpu
make perf-profile-mem
```

The profile targets start the Go pprof web interface on port 8080. To choose a
different address, run `go tool pprof` directly against the profile file.

## Benchmark families

| Family | Location | Measures |
| --- | --- | --- |
| Classification | `benchmarks/classification*_bench_test.go` | owned Vela Domain/PII/Guard batch inference, parallel calls, and Domain inference on short and up to 8K-token inputs |
| Decision | `benchmarks/decision_bench_test.go` | rule evaluation, priority selection, and parallel evaluation |
| Cache | `benchmarks/cache_bench_test.go` | cache sizes, search modes, concurrency, and hit-rate paths through the owned Vela Embedding provider |
| Looper | `../src/semantic-router/pkg/looper/*_bench_test.go` | Base, Fusion, ReMoM, and Flow helpers and execution |

The repository's reusable performance workflow runs these numeric comparisons
when the performance CI domain is selected. The workflow and `make perf-check`
use the same parser and thresholds, with advisory results in CI and strict
allocation checks in the local target. Model measurements also require a
same-checkpoint baseline passed to the comparator as shown above.

## Directory layout

```text
perf/
├── benchmarks/            Go component benchmarks
├── cmd/perftest/           benchmark parser, comparator, and report CLI
├── config/                 runner settings and comparison thresholds
├── pkg/benchmark/          parsing, comparison, and report implementation
├── pkg/profiler/           reusable pprof helper
├── scripts/                baseline and dataset utilities
└── testdata/
    ├── baselines/          committed comparison inputs
    └── examples/           illustrative, non-gating fixture files
```

## Update a baseline

Only refresh baselines after reviewing why allocation behavior changed:

```bash
make perf-baseline-update
git diff -- perf/testdata/baselines
```

Commit the baseline update with the code change that requires it. Do not use a
baseline refresh to hide an unexplained regression.

## Add a benchmark

1. Add a `BenchmarkXxx` function to the appropriate `perf/benchmarks` file, or
   beside the Looper implementation when it requires unexported Looper code.
2. Call `b.ReportAllocs()` and keep setup outside the timed region.
3. Add the narrowest suitable pattern to `config/thresholds.yaml` when the
   default thresholds are not appropriate.
4. Run the new benchmark directly before running the family target:

   ```bash
   cd perf
   go test -run '^$' -bench '^BenchmarkXxx$' -benchmem ./benchmarks/...
   ```

5. Update the applicable baseline only after the result and threshold have
   been reviewed.

Run the parser and comparator unit tests after changing benchmark tooling:

```bash
cd perf
go test ./pkg/benchmark/...
```
