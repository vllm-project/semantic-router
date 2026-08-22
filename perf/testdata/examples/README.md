# Parser Example

`benchmark-output-example.txt` is a parser-oriented example of Go benchmark
text. It is not a baseline or evidence of a measured result.

The versioned result and report contracts live under
[`../../pkg/benchmark/`](../../pkg/benchmark/). Generate real JSON, Markdown,
HTML, and raw suite logs from the repository root:

Generate current local artifacts from the repository root:

```bash
make perf-check
```

That command writes the current benchmark, comparison, and report artifacts
under `reports/perf/cpu-ci/`. Review the exact command, source revision, Go
version, execution host, and suite dimensions before drawing conclusions.

See the [performance microbenchmark guide](../../README.md) for supported
commands and interpretation boundaries.
