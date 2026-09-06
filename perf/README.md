# Performance Microbenchmarks

The `perf/` module measures component-level Router hot paths with Go
benchmarks. Use it to compare allocations, bytes, and execution time for
classification, decision evaluation, cache operations, and ExtProc data
handling. Looper microbenchmarks live beside the Looper implementation and are
included by the repository Make targets.

These are not end-to-end quality or load benchmarks. For reasoning quality,
session routing, hallucination detection, and fusion evaluations, see
[`bench/`](../bench/README.md).

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

The component targets build the Router and set the native-library path before
running `go test -bench`. Classification benchmarks require the benchmark model
artifacts; download them when they are not already available:

```bash
make download-models-perf
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

## What is gated

Thresholds in [`config/thresholds.yaml`](config/thresholds.yaml) are matched to
benchmark names in order; the first matching pattern wins. Unmatched names use
the `default` thresholds.

- `allocs/op` and `B/op` are blocking metrics because they are comparatively
  stable for the same code and Go version.
- `ns/op` is advisory because host speed and contention affect wall-clock
  measurements.
- A benchmark missing from the baseline is reported but cannot be compared
  until a reviewed baseline is added.

Treat an allocation pass as one signal, not a general performance guarantee.
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
| Classification | `benchmarks/classification*_bench_test.go` | batch inference, parallel calls, CGO overhead, and intent accuracy setup |
| Decision | `benchmarks/decision_bench_test.go` | rule evaluation, priority selection, and parallel evaluation |
| Cache | `benchmarks/cache_bench_test.go` | cache sizes, search modes, concurrency, and hit-rate paths |
| ExtProc data handling | `benchmarks/extproc_bench_test.go` | JSON encoding, request-body parsing, and header manipulation |
| Looper | `../src/semantic-router/pkg/looper/*_bench_test.go` | Base, Fusion, ReMoM, and Flow helpers and execution |

The repository's reusable performance workflow runs these numeric regression
checks when the performance CI domain is selected. The workflow and
`make perf-check` use the same parser, thresholds, and committed baseline
directory.

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
