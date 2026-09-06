# Illustrative Performance Outputs

The files in this directory are hand-maintained examples of report shapes.
Their benchmark names, numbers, timestamps, branches, recommendations, and CI
messages are fictional. They are not generated fixtures, test inputs,
baselines, or evidence that a workflow produced the illustrated result.

| File | Illustrates |
| --- | --- |
| `benchmark-output-example.txt` | Go benchmark-style text with additional annotations |
| `comparison-example.txt` | A human-oriented baseline comparison |
| `example-report.json` | A possible machine-readable comparison shape |
| `example-report.md` | A Markdown rendering of fictional comparison data |
| `example-report.html` | A standalone HTML report mock-up |
| `pprof-example.txt` | Annotated pprof-style output |
| `pr-comment-example.md` | A historical CI comment mock-up |

Do not build integrations against these files. The current parser and report
contracts live under [`../../pkg/benchmark/`](../../pkg/benchmark/), and the
committed comparison inputs live under [`../baselines/`](../baselines/).

Generate current local artifacts from the repository root:

```bash
make perf-check
```

That command writes the current benchmark and comparison data under
`reports/`. Review those artifacts, the exact command, source revision, Go
version, and execution host before drawing performance conclusions.

See the [performance microbenchmark guide](../../README.md) for supported
commands and interpretation boundaries.
