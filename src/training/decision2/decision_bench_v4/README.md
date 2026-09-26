# Decision Bench v4 public text-readable axis

This is an **independent** native typed-decision rerun of the public
[`atlanai/decision-bench`](https://github.com/atlanai/decision-bench) corpus at
commit `6fed2cd4c3608b070e649180796ccaef3d020a23`. It is separate from
JevArena and from the project's sealed final evaluation.

The upstream bench-v4 release has 1,071 cases, 35 tasks, and 11 categories.
The adapter evaluates 1,041 text-readable cases across 34 tasks and 10
categories. The 30 `DSN-1` icon cases have no text rendering and are reported
as **N/E**, never as model errors or zeroes. Of the 1,041 cases, 949 are
upstream `text`, 62 `text+image`, and 30 `image` with an upstream OCR/text
rendering. No image pixels are passed to the model. The modality breakdown is
therefore mandatory when interpreting scores; this track is not the upstream
vision-inclusive score.

The builder verifies the exact source corpus and manifest bytes as well as
the upstream canonical corpus digest. For each eligible row it uses the same
model-visible `state`, ordered option descriptions, and policy plus question
instructions as the upstream native TypeSafe adapter. Prompts contain no gold,
rationale, task/category, provenance, title, ask, or image asset. Gold and task
metadata are kept in a separate target file. The scorer checks every input
hash and model identity, counts missing and malformed answers as wrong, and
reports task and category macro accuracy, invalids, Brier score, log loss,
pmax ECE, point/argmax disagreement, and latency.

```bash
python3 -m decision_bench_v4.bench build \
  --upstream-root <pinned-upstream-checkout> \
  --output-dir <workspace>/decision-bench-v4-panel

# Collect <workspace>/decision-bench-v4-panel/prompts.jsonl using a native
# Decision collector, then score the resulting per-row prediction JSONL.
python3 -m decision_bench_v4.bench score \
  --panel-dir <workspace>/decision-bench-v4-panel \
  --predictions <workspace>/predictions.jsonl \
  --model-id <model-id> --model-revision <revision> \
  --output <workspace>/score.json
```

The repository's MIT code license does not replace the per-row source terms.
The upstream corpus retains individual source licenses, notices, and review
notes in its manifest and `data/SOURCES.md`; keep these with any redistributed
corpus or panel. This adapter adds no upstream cases or raw corpus files to
the research repository.
